import sys,os
import boto3
import argparse
import torch
import json
from statistics import mean
import efficientnet_pytorch

import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

 
import glob
import os.path as osp
import torch.utils.data as data
from torchvision import models, transforms
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt

MEAN = [0.485, 0.456, 0.40]
STD = [0.229, 0.224, 0.225]

class ImageTransform():

    def __init__(self, resize):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),  
                transforms.RandomHorizontalFlip(),  
                transforms.ToTensor(),  
                transforms.Normalize(MEAN, STD)  
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),  
                transforms.CenterCrop(resize),  
                transforms.ToTensor(),  
                transforms.Normalize(MEAN, STD)  
            ])
        }

    def __call__(self, img, phase='train'):
       
        return self.data_transform[phase](img)


def make_datapath_list(phase,rpath):
    target_path = osp.join(rpath + '/**/' +'*.jpg')
    path_list = []  

    
    for tpath in glob.glob(target_path):
        path_list.append(tpath)

    return path_list




class Dataset(data.Dataset):

    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list  
        self.transform = transform  
        self.phase = phase  

    def __len__(self):
       
        return len(self.file_list)

    def __getitem__(self, index):

        

        img_path = self.file_list[index]
        img = Image.open(img_path) 
        img = img.convert('RGB')

        img_transformed = self.transform(img, self.phase) 
        for i in range(0,10):
            n="label" + str(i)
            if n in img_path:
                label = i
            else:
                pass
        
        return img_transformed, label

def CreateDataset(Dpath,resize,batch_size):
    
    dataloaders_dict = {}
    for phase in ['train','val']:
       phaselist = make_datapath_list(phase=phase,rpath =Dpath)
       dataset = Dataset(file_list=phaselist,transform=ImageTransform(resize=resize),phase=phase)
       shuffle = True if phase == 'train' else False
       dataloader = data.DataLoader(dataset, batch_size = batch_size, shuffle=shuffle)
       dataloaders_dict[phase] = dataloader

    return dataloaders_dict

def EffNet(out_class):  
    import torch.nn as nn
    from efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b7')
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs,out_class)
    
    params_to_update = []
    update_param_names = ["_blocks.54._expand_conv.weight",
                            "_blocks.54._bn0.weight",
                            "_blocks.54._bn0.bias",
                            "_blocks.54._depthwise_conv.weight",
                            "_blocks.54._bn1.weight",
                            "_blocks.54._bn1.bias",
                            "_blocks.54._se_reduce.weight",
                            "_blocks.54._se_reduce.bias",
                            "_blocks.54._se_expand.weight",
                            "_blocks.54._se_expand.bias",
                            "_blocks.54._project_conv.weight",
                            "_blocks.54._bn2.weight",
                            "_blocks.54._bn2.bias",
                            "_conv_head.weight",
                            "_bn1.weight",
                            "_bn1.bias",
                            "_fc.weight",
                            "_fc.bias"]
    for name, param in model.named_parameters():
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False


    return model ,params_to_update

def train_phase(model,dataloaders,device,criterion,optimizer):
    model.train()
    model.to(device)

    for i, (inputs,labels) in enumerate(tqdm(dataloaders)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1) 
        loss = criterion(outputs, labels)
        loss.backward()
        
        if (i+1)%10 == 0:
            optimizer.step()
            optimizer.zero_grad()


def  train(args):

    global DAP,MOP,LR,BS,NE,IS,ON
    DAP = args.train_dir
    MOP = args.model_dir
    LR = args.learning_rate
    BS =args.batch_size
    NE = args.num_epochs
    IS = args.img_size
    ON = args.output_num
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    model ,params= EffNet(ON)
    dataloaders_dict = CreateDataset(DAP,IS,BS)
    optimizer = optim.Adam(params,lr=LR)
    criterion = nn.CrossEntropyLoss()


    for epoch in range(1, NE + 1):
        print("Epoch {}/{}".format(epoch, NE))
        train_phase(model,dataloaders_dict['train'], device, criterion, optimizer)
        

    torch.save(model.state_dict(),os.path.join(MOP,"saved_model"))

    print('finish')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str,default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train-dir", type=str,default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--img-size', type=int, default=400)
    parser.add_argument('--output-num', type=int, default=2)
    
    args  = parser.parse_args()


    train(args)
