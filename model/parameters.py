import torch
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--num_patches', type=int, default=196, help='number of patches') # (224/16) * (224/16)
parser.add_argument('--epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--weight_decay', default=1e-2, type=float)
parser.add_argument('--opt', default='adamw', type=str)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--imagenet_path', type=str, default='/home/share/nas2/ImageNet/')
parser.add_argument('--model_saved_path', type=str, default='/home/hx/CP/Img_Cls_ImageNet/models_saved')
parser.add_argument('--log_step', type=int, default=2, help='log_step')
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--warm_up', default=8, type=int)
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--cp_graph_path', type=str, default='/home/hx/CP/cp_graphsV2')

opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device('cuda' if cuda else 'cpu')

