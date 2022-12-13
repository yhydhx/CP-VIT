import pickle
import random
import os
import torch
import numpy as np


def re_org_pathch_embeds(x, indice):
    x1 = x.clone()
    batch, patch_num, _  = x.size()
    patches_in_core_nodes = len(indice[0])
    for i in range(batch):
        index = (indice[i]).tolist()
        #print(index)
        swap_index = [i for i in range(patch_num)]
        for temp in index:
            swap_index.remove(int(temp))
        swap_index = index+swap_index
        #print(swap_index)
        x1[i,:,:] = x[i,swap_index,:]

    return x1

#patch_size = 16 * 16
#imgs: batch * 3 * 224 * 224
def re_org_img_bug(imgs, indice):
    imgs_clone = imgs.clone()
    patches_in_core_nodes = len(indice[0])
    #re_postion_patches in batch---------------------------
    for i in range(len(imgs)):
        for j in range(patches_in_core_nodes):
            div = torch.div(indice[i][j] , 14, rounding_mode='floor')      #226 * 224 (16*16) 14*14
            mod = indice[i][j] % 14      #226 * 224 (16*16) 14*14
            pix_range = range(div * 16 , (div+1)*16,1)
            pix_range2 = range(mod * 16 , (mod+1)*16,1)
            div1 = torch.div(j , 14, rounding_mode='floor')      #226 * 224 (16*16) 14*14
            mod1 = j % 14      #226 * 224 (16*16) 14*14
            pix_range3 = range(div1 * 16 , (div1+1)*16 ,1)
            pix_range4 = range(mod1 * 16 , (mod1+1)*16, 1)
            imgs[i,:,pix_range3,pix_range4] = imgs_clone[i,:,pix_range,pix_range2]
            imgs[i,:,pix_range,pix_range2] = imgs_clone[i,:, pix_range3, pix_range4]
    return imgs


def re_org_img(imgs, indice):
    imgs_clone = imgs.clone()
    patches_in_core_nodes = len(indice[0])

    for i in range(len(imgs)):
        swap_index = (indice[i]).tolist()
        for j in range(patches_in_core_nodes):
            if(indice[i][j] < patches_in_core_nodes):
                swap_index.remove(indice[i][j])
        pos_marker = 0
        for j in range(len(swap_index)):
            while(1):
                if(pos_marker not in indice[i]):
                    index = swap_index[j]
                    div = torch.div(index, 14, rounding_mode= 'floor')
                    mod = index % 14
                    div1 = torch.div(pos_marker, 14, rounding_mode='floor')
                    mod1 = pos_marker % 14
                    #print(imgs_clone[i,:,(div * 16):((div+1)* 16):1, (mod* 16):((mod+1)* 16):1])
                    #print(imgs_clone[i,:,(div1 * 16):((div1+1)* 16):1, (mod1* 16):((mod1+1)* 16):1])
                    imgs[i,:,(div1 * 16):((div1+1)* 16):1, (mod1* 16):((mod1+1)* 16):1] = imgs_clone[i,:,(div * 16):((div+1)* 16):1, (mod* 16):((mod+1)* 16):1]
                    imgs[i,:,(div * 16):((div+1)* 16):1, (mod* 16):((mod+1)* 16):1] = imgs_clone[i,:,(div1 * 16):((div1+1)* 16):1, (mod1* 16):((mod1+1)* 16):1]
                    pos_marker += 1
                    break
                pos_marker += 1
            
    return imgs


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    # pdb.set_trace()
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

if __name__ == '__main__':
    
    img = torch.from_numpy(np.random.rand(1,1,224,224))
    indice = np.random.randint(16,size=(1, 4))
    print(indice)
    img = re_org_img(img,indice)

    #
    proj = torch.nn.Conv2d(1, 1, kernel_size=2, stride=2)
    x = torch.from_numpy(np.random.rand(1,1,6,4)).float()
    print(x)
    x = proj(x)
    print(x)
    
    '''

    x = torch.Tensor(2, 8, 3)
    indice = torch.randint(0, 8, (2, 1, 2))
    print(x)
    print(indice)
    x = re_org_pathch_embeds(x, indice)
    print(x)
    '''
