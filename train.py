import os
os.environ["MKL_NUM_THREADS"] = '4'
os.environ["NUMEXPR_NUM_THREADS"] = '4'
os.environ["OMP_NUM_THREADS"] = '4'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
import sys
import shutil
import datetime
from utils import *
import torch
import random
import numpy as np
from torch import nn
import logging
from render import render, Layer
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from model import *
from tqdm import tqdm
from render import fabric_map
import argparse

num_epoch = 100                     # Maximum epoch.
batch_size = 32
lr = 0.0001                         # Learning rate.
test_epoch = 5                      # Report appearance under eval set per 5 epoches.
save_epoch = 50                     # Saving the model per 50 epoches.
trainset_prop = 0.9                 # Proportion of train set.


class myDataSet(Dataset):
    def __init__(self, data_paths):
        self.imgs = []
        self.params = []

        for data_path in data_paths:
            with open(os.path.join(data_path, "params.txt"), 'r') as f:
                lines = f.read().splitlines()

            pbar = tqdm(total=len(lines), desc="Loading dataset", file=sys.stdout)
            for s in lines:
                pbar.update(1)
                id, pname = eval(s)

                front = readimg(os.path.join(data_path, f"{id}_front.png")).permute([2, 0, 1])
                back = readimg(os.path.join(data_path, f"{id}_back.png")).permute([2, 0, 1])
                front = vggTrans(front)
                back = vggTrans(back)
                self.imgs.append((front, back))
                param = params2feature(Parameters().from_name(pname))
                self.params.append(param)

            pbar.close()
        print(f"Done. dataset len = {len(self.imgs)}")
        pass

    def __getitem__(self, idx):
        return self.imgs[idx][0], self.imgs[idx][1], self.params[idx]

    def __len__(self):
        return len(self.imgs)


L1Lossfn = nn.L1Loss().to(device)
CrossEntropyLossfn = nn.CrossEntropyLoss().to(device)
def mycriterion(f1, ftar):
    if len(f1.shape) == 1:
        tar_type = torch.argmax(ftar[:5])
        loss = CrossEntropyLossfn(f1[:5], tar_type) * 0.2 + L1Lossfn(f1[5:], ftar[5:])
    else:
        tar_type = torch.argmax(ftar[:, :5], dim=1)
        loss = CrossEntropyLossfn(f1[:, :5], tar_type) * 0.2 + L1Lossfn(f1[:, 5:], ftar[:, 5:])
    return loss


def eval_model(test_data_loader):
    loss_sum = 0
    net.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(test_data_loader), file=sys.stdout, desc="Eval epoch")
        f1s, f2s = [], []
        for idx, (front, back, params) in enumerate(test_data_loader):
            pbar.update(1)
            front = front.to(device)
            back = back.to(device)
            params = params.to(device)
            pred = net.forward2(front, back)
            loss = mycriterion(pred, params)
            loss_sum += loss.item()
            pbar.set_postfix({"Eval Loss": loss.item()})

            f1s.append(pred)
            f2s.append(params)
        loss_sum /= len(test_data_loader)
        pbar.set_postfix({"Eval Loss": loss_sum})
        pbar.close()

    f1s = torch.cat(f1s, dim=0)
    f2s = torch.cat(f2s, dim=0)
    return loss_sum, detailLoss(f1s, f2s)


def train(data_loader, test_data_loader, num_epoch):
    lr_show = []
    loss_show = []
    eval_loss_show = []
    eval_epoches = []
    detail_loss_show = []
    train_detail_loss_show = []

    eval_loss, detail_loss = eval_model(test_data_loader)
    eval_loss_show.append(eval_loss)
    detail_loss_show.append(detail_loss)
    eval_epoches.append(0)
    for cur_epoch in range(1, num_epoch + 1):
        loss_sum = 0
        net.train()
        pbar = tqdm(total=len(data_loader), file=sys.stdout, desc=f"Train epoch {cur_epoch}")

        f1s, f2s = [], []
        for idx, (front, back, parameters) in enumerate(data_loader):
            pbar.update(1)
            optimizer.zero_grad()
            front = front.to(device)
            back = back.to(device)
            parameters = parameters.to(device)
            pred = net.forward2(front, back)
            loss = mycriterion(pred, parameters)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            pbar.set_postfix({"Loss": loss.item()})

            f1s.append(pred)
            f2s.append(parameters)
        loss_sum /= len(data_loader)
        pbar.set_postfix({"Loss": loss_sum})
        pbar.close()

        f1s = torch.cat(f1s, dim=0)
        f2s = torch.cat(f2s, dim=0)
        train_detail_loss_show.append(detailLoss(f1s, f2s))
        with open(os.path.join(writePath, 'train_detail_loss.txt'), 'w') as f:
            for i in range(len(train_detail_loss_show)):
                f.writelines(str(train_detail_loss_show[i]) + '\n')

        scheduler.step()
        loss_show.append(loss_sum)
        lr_show.append(optimizer.param_groups[-1]['lr'])

        if cur_epoch % test_epoch == 0:
            eval_loss, detail_loss = eval_model(test_data_loader)
            eval_loss_show.append(eval_loss)
            detail_loss_show.append(detail_loss)
            print(detail_loss)
            eval_epoches.append(cur_epoch)
            plt.clf()
            plt.plot(range(1, len(lr_show) + 1), lr_show, label=str("lr"))
            plt.legend()
            plt.savefig(os.path.join(writePath + '/lr.png'))

            plt.clf()
            plt.plot(range(1, len(loss_show) + 1), loss_show, label=str(loss_show[-1]))
            plt.legend()
            plt.savefig(os.path.join(writePath + '/train_loss.png'))

            plt.clf()
            plt.plot(eval_epoches, eval_loss_show, label=str(eval_loss_show[-1]))
            plt.legend()
            plt.savefig(os.path.join(writePath + '/eval_loss.png'))

            plt.clf()
            for key in ['pattern_loss', 'uv_loss', 'diffuse_loss', 'thickness_loss', 'roughness_loss', 'weight2_loss']:
                ls = list(map(lambda dic: dic[key], detail_loss_show))
                plt.plot(eval_epoches, ls, label=key)
            plt.legend()
            plt.savefig(os.path.join(writePath + '/detail_loss.png'))
            with open(os.path.join(writePath, 'detail_loss.txt'), 'w') as f:
                for i in range(len(eval_epoches)):
                    f.writelines(str(eval_epoches[i]) + ": " + str(detail_loss_show[i]) + '\n')

        if cur_epoch % save_epoch == 0 or cur_epoch == num_epoch:
            torch.save({'fc_net_state_dict': net.fc_net.state_dict()},
                       writePath + '/model_' + str(cur_epoch) + '.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the parameters prediction network.')
    parser.add_argument('--path', type=str, default="synthetic",
                        help='Dataset path.')
    args = parser.parse_args()

    dataset = myDataSet([args.path])

    writePath = f"./checkpoint/train_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(writePath)
    shutil.copyfile('./model.py', os.path.join(writePath, 'model.py'))
    shutil.copyfile('./train.py', os.path.join(writePath, 'train.py'))
    shutil.copyfile('./utils.py', os.path.join(writePath, 'utils.py'))
    shutil.copyfile('./render.py', os.path.join(writePath, 'render.py'))

    net = paramPredNet().to(device)
    optimizer = torch.optim.Adam(net.fc_net.parameters(), lr=lr, eps=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     [50, 100, 130, 150, 170, 200, 250, 300, 350, 400, 450],
                                                     gamma=0.5,
                                                     last_epoch=-1)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [trainset_prop, 1 - trainset_prop])
    train_data_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    test_data_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    print("Start trainning...")

    train(train_data_loader, test_data_loader, num_epoch)



