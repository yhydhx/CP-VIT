
from CP_vit import *
import numpy as np
import itertools
import time
from parameters import *
import torch
from dataloader import *
from torch.utils.tensorboard import SummaryWriter
from utils import *
import timm
from timm.loss import LabelSmoothingCrossEntropy
import torch
from timm.scheduler.cosine_lr import CosineLRScheduler

#get cp_mask for ViT----------------------------------------------------------------------
cp_graph_path = opt.cp_graph_path
graphs = os.listdir(cp_graph_path)
for graph in graphs:
    print('graph is ', graph)
    core_num = graph.split('_')[3]
    node_num = graph.split('_')[1]
    SPL = graph.split('_')[5]
    CC = (graph.split('_')[7])[0:3]
    save_path = './Cls/'+graph.split('.')[0]+'.'+graph.split('.')[1]+'.'+graph.split('.')[2]
    if(not os.path.exists(save_path)):
        os.makedirs(save_path)
    else:
        continue
    G = nx.read_gexf(cp_graph_path + '/' + graph)
    adjG = nx.to_numpy_array(G)
    cp_mask, node_mask, repeat_in, repeat_out = get_mask(opt.num_patches, opt.num_patches, adjG)

    temp = np.ones((1,196))
    cp_mask = np.concatenate( (temp, cp_mask), axis=0)
    temp = np.ones((197,1))
    cp_mask = np.concatenate( (temp, cp_mask), axis = 1)
    #cp_mask = np.ones( (197, 197))
    patches_in_core_nodes = np.sum(repeat_in[0:int(core_num)])     #get the number of patches in core nodes
    cp_mask = torch.from_numpy(cp_mask).float().to(device)
    node_num = graph.split('_')[1]
    repeat_pattern = torch.zeros(int(node_num), opt.num_patches)   # for add the fmri signal embeddings within the same node
    index_indicator = 0
    for i in range(int(node_num)):
        temp = torch.zeros(1, opt.num_patches)
        temp[0,index_indicator:(index_indicator+repeat_in[i])] = 1
        repeat_pattern[i,:] = temp
        #print(repeat_pattern[i,:])
        index_indicator += repeat_in[i]
    #--------------------------------------
    writer = SummaryWriter(log_dir=save_path)

    cp_vit = CP_ViT(Layer=12)
    cp_vit.to(device)

    saved_models = os.listdir(opt.model_saved_path)
    flag = 1
    for saved_model in saved_models:
        if(node_num in saved_model and core_num in saved_model and SPL in saved_model and CC in saved_model):
            cp_vit.load_state_dict(torch.load(opt.model_saved_path+'/'+saved_model))
            temp = float(saved_model.split('_')[0])
            flag = 0
            break
    if(flag):
        vit_small_pretrain = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes = 1000)
        vit_small_pretrain_weight = vit_small_pretrain.state_dict()
        cp_vit.load_state_dict(vit_small_pretrain_weight)
        temp = 0.75   
    
    for k,v in cp_vit.named_parameters():
        if('attn' not in k and 'head' not in k and 'patch_embed' not in k):
            print('freeze layer:', k)
            v.requires_grad = False
    #print('model parameters:', sum(param.numel() for param in cp_vit.parameters()) /1e6)
    #for name, param in cp_vit.named_parameters():
    #    if(param.requires_grad):
    #        print(name)
    print('threshold: ', temp)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, cp_vit.parameters()), 
                                        lr=opt.lr, weight_decay=opt.weight_decay,)
    #optimizer = create_optimizer(opt, cp_vit)
    scheduler = CosineLRScheduler(optimizer, t_initial=opt.epochs, lr_min=1e-7, warmup_t=opt.warm_up)
    # loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction=args.cross_loss_para)
    loss_fn = LabelSmoothingCrossEntropy(0.1)

    start_t = time.time()
    #tra_dataloader, te_dataloader = get_loader(opt.imagenet_path)
    tra_dataloader, te_dataloader = get_cal_loader()
    for epoch in range(opt.epochs):
        cp_vit.train()

        batch_time = AverageMeter()  # forward prop. + back prop. time
        accs = AverageMeter()
        epoch_tra_loss = AverageMeter()  
        
        total = 0
        correct = 0

        def backward_hook(module, grad_in, grad_out):
            grad_block.append(grad_out[0].detach().cpu())
        hook = cp_vit.patch_embed.register_full_backward_hook(backward_hook)
        for i, (tra_transformed_normalized_img, tra_labels) in enumerate(tra_dataloader):
            grad_block = list()

            batchSize = tra_transformed_normalized_img.shape[0]
            #print('current lr: ', (optimizer.state_dict()['param_groups'][0]['lr']))

            #------------------------------------------------------
            tra_transformed_normalized_img = tra_transformed_normalized_img.float().to(device)

            outputs = cp_vit.forwar_with_cp_mask(tra_transformed_normalized_img, cp_mask) 
            cls_loss = loss_fn(outputs, tra_labels.to(device))
            optimizer.zero_grad()
            cls_loss.backward()
            #find important patches according to gradients-------------------------------
            a = grad_block[0].abs().sum(dim = 2)
            #print(a.shape)  #batch_size, patch_num
            values, index = torch.topk(a, dim = 1, k = patches_in_core_nodes)
            re_org_tra_transformed_normalized_img = re_org_img(tra_transformed_normalized_img, index)
            #----------------------------------------------------------------------------

            outputs = cp_vit.forwar_with_cp_mask(re_org_tra_transformed_normalized_img, cp_mask) 
            cls_loss = loss_fn(outputs, tra_labels.to(device))

            #acc---------------------------------------------------------------------
            _, predictions = torch.max(outputs, 1)

            total += tra_labels.size(0)
            correct += (predictions.cpu() == tra_labels).sum().item()
            #------------------------------------------------------------------------

            optimizer.zero_grad()
            cls_loss.backward()
            optimizer.step()
            scheduler.step(cls_loss)

            epoch_tra_loss.update(cls_loss.detach())
            accs.update(correct/total)
            batch_time.update(time.time() - start_t)

            # Print log info
            if i % opt.log_step == 0:
                # print('======================== print results \t' + time.asctime(time.localtime(time.time())) + '=============================')
                print('Train Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Classification_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(epoch, i, len(tra_dataloader),
                                                                                    batch_time=batch_time,
                                                                                    loss=epoch_tra_loss,
                                                                                    top5=accs,))
            
        hook.remove()

        writer.add_histogram('clssification_train_loss', epoch_tra_loss.avg, epoch)
        writer.add_histogram('train_acc', accs.avg, epoch)

        start_t = time.time()
        #############################################################################

        accs = AverageMeter()
        epoch_te_loss = AverageMeter()  

        total = 0
        correct = 0
        hook = cp_vit.patch_embed.register_full_backward_hook(backward_hook)

        for i,(te_transformed_normalized_img, te_labels) in enumerate(te_dataloader):
            grad_block = list()
            te_transformed_normalized_img = te_transformed_normalized_img.float().to(device)

            outputs = cp_vit.forwar_with_cp_mask(te_transformed_normalized_img, cp_mask) 
            cls_loss = loss_fn(outputs, te_labels.to(device))
            optimizer.zero_grad()
            cls_loss.backward(retain_graph=True)
            #find important patches according to gradients-------------------------------
            a = grad_block[0].abs().sum(dim = 2)
            #print(a.shape)  #batch_size, patch_num
            values, index = torch.topk(a, dim = 1, k = patches_in_core_nodes)  
            re_org_te_transformed_normalized_img = re_org_img(te_transformed_normalized_img, index)
            #----------------------------------------------------------------------------
            with torch.no_grad():
                cp_vit.eval()
                outputs = cp_vit.forwar_with_cp_mask(re_org_te_transformed_normalized_img, cp_mask) 
                cls_loss = loss_fn(outputs, te_labels.to(device))
            #acc---------------------------------------------------------------------
            _, predictions = torch.max(outputs, 1)


            total += te_labels.size(0)
            correct += (predictions.cpu() == te_labels).sum().item()
        #------------------------------------------------------------------------

            epoch_te_loss.update( cls_loss.detach() )
            accs.update(correct/total)

            # Print log info
            if i % 1 == 0:  #opt.log_step
                # print('======================== print results \t' + time.asctime(time.localtime(time.time())) + '=============================')
                print('Test Epoch: [{0}][{1}/{2}]\t'                 
                    'Classification_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(epoch, i, len(te_dataloader),
                                                                                    loss=epoch_te_loss,
                                                                                    top5=accs,))
        if( accs.avg > temp):
            temp = accs.avg
            torch.save(cp_vit.state_dict(),  opt.model_saved_path + '/'+str(temp)+'_'+graph.split('.')[0]+'.'+graph.split('.')[1]+'.'+graph.split('.')[2]+ '_cp_vit_img_finetue.pkl') 

        hook.remove()

        writer.add_histogram('clssification_test_loss', epoch_te_loss.avg, epoch)
        writer.add_histogram('test_acc', accs.avg, epoch)

    del cp_vit
    del optimizer
    del scheduler






