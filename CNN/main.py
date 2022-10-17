import torch
import torch.nn as nn
import os
from torch.utils import data
from tqdm import tqdm

import mydataset 
from vggnet import SE_VGG


lr = 1e-3
batch = 20
epochs = 36
device = "cuda" if torch.cuda.is_available() else "cpu"
iou_thr = 0.5


def train_epoch(model, dataloader, criterion: dict, optimizer,
                scheduler, epoch, device):
    model.train()
    bar = tqdm(dataloader)
    bar.set_description(f'epoch {epoch:2}')
    correct, total = 0, 0
    for X, y in bar:

        x,y_cls=X.to(device),y.to(device)
        c1,c2,c3,c4,c5,c6=model(x)
        loss=3*criterion(c1,y_cls[:,0].long())+3*criterion(c2,y_cls[:,1].long())+3*criterion(c3,y_cls[:,2].long())+2*criterion(c4,y_cls[:,3].long())+2*criterion(c5,y_cls[:,4].long())+criterion(c6,y_cls[:,5].long())
        loss/=14
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct+=sum((torch.argmax(c1,axis=1)==y_cls[:,0]).cpu().detach().numpy()&(torch.argmax(c2,axis=1)==y_cls[:,1]).cpu().detach().numpy()&(torch.argmax(c3,axis=1)==y_cls[:,2]).cpu().detach().numpy()&(torch.argmax(c4,axis=1)==y_cls[:,3]).cpu().detach().numpy()&(torch.argmax(c5,axis=1)==y_cls[:,4]).cpu().detach().numpy()&(torch.argmax(c6,axis=1)==y_cls[:,5]).cpu().detach().numpy())
        total+=len(X)


        bar.set_postfix_str(f'lr={scheduler.get_last_lr()[0]:.4f} acc={correct / total * 100:.2f} loss={loss.item():.2f}')
    scheduler.step()


def test_epoch(model, dataloader, device, epoch):
    model.eval()
    with torch.no_grad():
        correct, correct_cls, total = 0, 0, 0
        for X, y in dataloader:


            x,y_cls=X.to(device),y.to(device)
            c1,c2,c3,c4,c5,c6=model(x)
            correct+=sum((torch.argmax(c1,axis=1)==y_cls[:,0]).cpu().detach().numpy()&(torch.argmax(c2,axis=1)==y_cls[:,1]).cpu().detach().numpy()&(torch.argmax(c3,axis=1)==y_cls[:,2]).cpu().detach().numpy()&(torch.argmax(c4,axis=1)==y_cls[:,3]).cpu().detach().numpy()&(torch.argmax(c5,axis=1)==y_cls[:,4]).cpu().detach().numpy()&(torch.argmax(c6,axis=1)==y_cls[:,5]).cpu().detach().numpy())
            total+=len(X)

        print(f' val acc: {correct / total * 100:.2f}')


def main():
    root='F:/test_mission/input'
    workspace_dir='F:/test_mission/src/CNN/models'
    trainloader = data.DataLoader(mydataset.get_dataset(root,mode='train'),#修改了下路径
                                  batch_size=batch, shuffle=True, num_workers=4)
    testloader = data.DataLoader(mydataset.get_dataset(root,mode='vald'),
                                 batch_size=batch, shuffle=True, num_workers=4)
    model = SE_VGG(num_classes=11).to(device)
    #model.load_state_dict(torch.load(os.path.join(workspace_dir, 'dcgan_g.pth')))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95,
                                                last_epoch=-1)
    criterion =nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_epoch(model, trainloader, criterion, optimizer,
                    scheduler, epoch, device)
        test_epoch(model, testloader, device, epoch)
        if (epoch+1) % 4 == 0:
            torch.save(model.state_dict(), os.path.join(workspace_dir, f'dcgan_g.pth'))
    


if __name__ == '__main__':
    main()
