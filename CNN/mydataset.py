import os, sys, glob, shutil, json
import cv2

from PIL import Image
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label 
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        fname = self.img_path[index]
        img = cv2.imread(fname)
        img = self.BGR2RGB(img)
        if self.transform is not None:
            img = self.transform(img)
        
        # 原始SVHN中类别10为数字0
        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl)  + (6 - len(lbl)) * [10]
        return img, torch.from_numpy(np.array(lbl[:6]))

    def __len__(self):
        return len(self.img_path)
    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def get_dataset(root,mode):
    if mode=='train':
        train_json = json.load(open(os.path.join(root,'mchar_train.json')))
        train_label = [train_json[x]['label'] for x in train_json.keys()]
        train_path = [os.path.join(root,'mchar_train/') + x for x in train_json.keys()]
    elif mode=='vald':
        train_json = json.load(open(os.path.join(root,'mchar_val.json')))
        train_label = [train_json[x]['label'] for x in train_json.keys()]
        train_path = [os.path.join(root,'mchar_val/') + x for x in train_json.keys()]

    # resize the image to (64, 64)
    # linearly map [0, 1] to [-1, 1]
    if mode=='train':
        transform = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ColorJitter(0.3, 0.3, 0.2),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3) ] )
    else:
        transform = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3) ] )
    dataset = SVHNDataset(train_path,train_label, transform)
    return dataset     
if __name__ == '__main__':
    root='F:/test_mission/input'
    train_loader = torch.utils.data.DataLoader(get_dataset(root,'train'), 
        batch_size=10, # 每批样本个数
        shuffle=False, # 是否打乱顺序
        num_workers=10, # 读取的线程个数
    )
    for data in train_loader:
        print(data)
        break