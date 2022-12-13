
import torch
from torch import nn
from torch._C import set_flush_denormal
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import nibabel as nib
import numpy as np
import os
from CP_relation_graph import *
from torch.autograd import Variable
from parameters import *
import seaborn as sns
import matplotlib.pyplot as plt
from dataloader import *
from timm.models.vision_transformer import partial
from timm.models.layers.weight_init import trunc_normal_
from timm.models.vision_transformer import LayerScale
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg, named_apply, adapt_input_conv, checkpoint_seq
from timm.models.layers.helpers import to_2tuple
from timm.models.layers import Mlp
from utils import *

try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message


def init_weights_vit_moco(module: nn.Module, name: str = ''):
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()

def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()

def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()

def get_init_weights_vit(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cp_mask):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale * cp_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, cp_mask):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), cp_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size) # Batch * 3 * 224 *224 -> Batch * 768 * 14 * 14
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
            class_token=True, no_embed_class=False, fc_norm=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            weight_init='', embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x, cp_mask):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, cp_mask)
            #x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, cp_mask):
        x = self.forward_features(x, cp_mask)
        x = self.forward_head(x)
        return x


class CP_ViT(VisionTransformer):
    def __init__(self, Layer = None):
        super().__init__(patch_size=16, embed_dim= 384, depth=12, num_heads=6, num_classes=1000)
        if Layer is None:
            self.layer = 12
        else:
            self.layer = Layer 
    
    def forwar_with_cp_mask(self, x, cp_mask, redis_index = None, dete_patch = True):  
        x = self.patch_embed(x)
        B, num_patches = x.shape[0], x.shape[1]

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim = 1)
        x = self.pos_drop(x + self.pos_embed)
        for i in range(self.layer):
            x = self.blocks[i](x, cp_mask)
        x = self.norm(x)

        return self.head(x[:,0])



class CPAttention(nn.Module):
    def __init__(self, dim, cp_mask, patches_in_core_nodes, repeat_pattern, heads = 8, dim_head = 64, dropout = 0.0):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        #cp_mask = np.repeat(cp_mask, dim, axis=0)
        #self.cp_mask = np.repeat(cp_mask, dim, axis=1)
        self.cp_mask = cp_mask
        self.patches_in_core_nodes = patches_in_core_nodes
        self.repeat_pattern = repeat_pattern

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        #dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        #attn = self.attend(dots) 
        #out = torch.matmul(attn, v)
        #---------------------------------------------------
        #attention_score = abs(np.array(torch.matmul(q, k.transpose(-1, -2)).detach().cpu()))* self.scale
        #attention_score = (attention_score.sum(axis = 3)).sum(axis = 1)
        cp_mask_nonzeros_num = np.count_nonzero(np.array((self.cp_mask).cpu()), axis = 1 )
        attention_score = abs(np.array(torch.matmul(q, k.transpose(-1, -2)).detach().cpu() * (self.cp_mask.cpu()) ))* self.scale
        attention_score = (attention_score.sum(axis = 1)).sum(axis = 2) / cp_mask_nonzeros_num
        descend_indices = list(np.argsort(attention_score))  #upsend
        descend_indices.reverse()               #descend
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale * (self.cp_mask)
        attn = self.attend(dots) 
        out = torch.matmul(attn, v)
        #---------------------------------------------------

        #visualize attention score-----------
        '''
        temp =  (abs((torch.matmul(q, k.transpose(-1, -2)).detach().cpu()))* self.scale).sum(axis = 1)
        temp = temp / temp.max()
        temp = (torch.matmul( torch.matmul(self.repeat_pattern, temp), self.repeat_pattern.permute(0,2,1))).squeeze()
        fig = sns.heatmap(temp)
        plt.show()
        if(os.path.exists('./ResAnalysis/node_attention.jpg')):
           os.remove('./ResAnalysis/node_attention.jpg')
        heatmap = fig.get_figure()
        heatmap.savefig('./ResAnalysis/node_attention.jpg')
        '''
        #---------------------------------

        out = rearrange(out, 'b h n d -> b n (h d)')
        out_clone = out.clone()
        #re_postion_patches in batch---------------------------
        for i in range(1, self.patches_in_core_nodes+1):
            for j in range(len(descend_indices)):
                #print('j: {}, i: {}, indice: {}'.format(j,i,descend_indices[j][i]))
                #print('len(descend_indices) ', len(descend_indices))
                out[j,i,:] = out_clone[j,descend_indices[j][i],:]
                out[j,descend_indices[j][i],:] = out_clone[j, i, :]
                
        #---------------------------------------------
        return self.to_out(out)


class CPTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, cp_mask, dropout, patches_in_core_nodes, repeat_pattern ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, CPAttention(dim, cp_mask = cp_mask, patches_in_core_nodes = patches_in_core_nodes, repeat_pattern = repeat_pattern, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class CP_ViT_MC_Img(nn.Module):
    def __init__(self, *, image_size, patch_size, channels, num_classes, dim, depth, heads, mlp_dim, cp_mask, \
        pool = 'cls', dim_head = 64, dropout = 0.1, emb_dropout = 0.1, repeat_pattern, patches_in_core_nodes):
        super().__init__()  
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
    
        self.component_embedding = nn.Sequential(
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear( patch_dim, dim ),
        )  
        
        self.patches_in_node =repeat_pattern.unsqueeze(0)
        self.patches_in_core_nodes = patches_in_core_nodes
        #CP mask----------------------------------------------------------------------
        self.mask = cp_mask
        #-----------------------------------------------------------------------------

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) # 1 for cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = CPTransformer(dim, depth, heads, dim_head, mlp_dim,  self.mask, dropout, self.patches_in_core_nodes, self.patches_in_node)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)    #284 * 284, pathces within a node are sumed, then there will be 50 * 284
        )

    def forward(self, img):
        x = self.component_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n +1 )]  # 1 for cls token, 1 for time token
        x = self.dropout(x)
        
        x_transoformer = self.transformer(x)   

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        return self.mlp_head(x) 



if __name__ == '__main__':

    #get cp_mask for ViT----------------------------------------------------------------------
    cp_graph_path = opt.cp_graph_path
    graphs = os.listdir(cp_graph_path)
    for graph in graphs:
        node_num = graph.split('_')[1]
        core_num = graph.split('_')[3]
        if(not core_num == str(opt.core_nodes)):
            continue
        save_path = './Cls/'+graph.split('.')[0]
        graph_id = graph.split('_')[5]
        G = nx.read_gexf(cp_graph_path + '/' + graph)
        adjG = nx.to_numpy_array(G)
        cp_mask, node_mask, repeat_in, repeat_out = get_mask(opt.num_patches, opt.num_patches, adjG)
        break
    #plt.subplot(1,2,1)
    #ax1 = sns.heatmap(nx.to_numpy_array(G))
    #plt.subplot(1,2,2)
    #ax2 = sns.heatmap(cp_mask)
    #plt.show()

    temp = np.ones((1,196))
    cp_mask = np.concatenate( (temp, cp_mask), axis=0)
    temp = np.ones((197,1))
    cp_mask = np.concatenate( (temp, cp_mask), axis = 1)

    cp_mask = np.ones((197,197))

    patches_in_core_nodes = np.sum(repeat_in[0:opt.core_nodes])     #get the number of patches in core nodes
    cp_mask = torch.from_numpy(cp_mask).float()
    repeat_pattern = torch.zeros(int(node_num), opt.num_patches)   # for add the fmri signal embeddings within the same node
    index_indicator = 0
    for i in range(int(node_num)):
        temp = torch.zeros(1, opt.num_patches)
        temp[0,index_indicator:(index_indicator+repeat_in[i])] = 1
        repeat_pattern[i,:] = temp
        print(repeat_pattern[i,:])
        index_indicator += repeat_in[i]
    #--------------------------------------


    vit_net = CP_ViT_MC_Img( 
                image_size = 224,
                patch_size = 16,
                num_classes = 3,
                channels = 3,
                dim= 10,     
                depth = 6, 
                heads = 12, 
                dim_head = 64,    # dim_head * heads = dim
                mlp_dim = 40, 
                pool = 'cls', 
                dropout = 0.0, 
                emb_dropout = 0.0,
                cp_mask = cp_mask, 
                repeat_pattern = repeat_pattern,
                patches_in_core_nodes = patches_in_core_nodes
                )
    
    # loss function
    cls_crietion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,(vit_net.parameters() )), 
                                            lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay =0)


    data_training, data_testing = load_data(opt.datafolder)

    classes = ('normal', 'benign', 'malignant')
    label_ID_dict = label_structure()
    for epoch in range(opt.epochs):
        vit_net.train()
        
        correct_pred = {classname: 0 for classname in label_ID_dict}
        total_pred = {classname: 0 for classname in label_ID_dict}
        total = 0
        correct = 0
        tra_dataloader = get_loader(data_training, data_testing, train= 1, test = 0, batch_size=opt.batch_size, )
        for i, (tra_transformed_normalized_img, tra_labels) in enumerate(tra_dataloader):
            batchSize = tra_transformed_normalized_img.shape[0]
            print('current lr: ', optimizer.state_dict()['param_groups'][0]['lr'])

            #------------------------------------------------------
            tra_transformed_normalized_img = tra_transformed_normalized_img.float()

            outputs = vit_net(tra_transformed_normalized_img) 

            cls_loss = cls_crietion(outputs, tra_labels)

            #acc---------------------------------------------------------------------
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(tra_labels, predictions.cpu()):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

            # print accuracy for each class
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                print(f'{epoch}: Accuracy for class: {classname:5s} is {accuracy:.1f} %')

            total += tra_labels.size(0)
            correct += (predictions.cpu() == tra_labels).sum().item()
            #------------------------------------------------------------------------

            optimizer.zero_grad()
            cls_loss.backward()
            optimizer.step()
            break

        #with torch.no_grad():
        vit_net.eval()


        te_dataloader = get_loader(data_training, data_testing, train= 0, test = 1, batch_size=opt.batch_size, )
        correct_pred = {classname: 0 for classname in label_ID_dict}
        total_pred = {classname: 0 for classname in label_ID_dict}
        total = 0
        correct = 0
        for i,(te_transformed_normalized_img, te_labels) in enumerate(te_dataloader):



            te_transformed_normalized_img = te_transformed_normalized_img.float()

            outputs = vit_net(te_transformed_normalized_img) #inde_comps1,inde_comps2

            cls_loss = cls_crietion(outputs, te_labels)

            #acc---------------------------------------------------------------------
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(te_labels, predictions.cpu()):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

            # print accuracy for each class
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                print(f'{epoch}: Accuracy for class: {classname:5s} is {accuracy:.1f} %')

            total += te_labels.size(0)
            correct += (predictions.cpu() == te_labels).sum().item()
        #------------------------------------------------------------------------

 


    
