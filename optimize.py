import os
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import shutil
import datetime
from utils import *
import torch
import random
import numpy as np
import json
from model import *
from torch import nn
from skimage.transform import resize
from render import render, Layer, FILM_SIZE, fabric_map
from matplotlib import pyplot as plt
import argparse
from argparse import RawTextHelpFormatter
from tqdm import tqdm


# calculate loss in sRGB space.
def optimize(gt_front, gt_back, params, writePath):
    params.noise = torch.tensor(0.0, device=device)
    params.to(device)
    params.set_requires_grad(True)

    layer0 = fabric_map[params.type]["layer0"]
    layer1 = fabric_map[params.type]["layer1"]
    downsize = (16, 16)

    gt_front_down = downsample(gt_front, downsize).to(device)
    gt_back_down = downsample(gt_back, downsize).to(device)
    gt_gram_front = net.gram(gt_front)
    gt_gram_back = net.gram(gt_back)

    num_epoch = 300
    lr = 0.01
    loss_weight = (1, 0.1, 0.001)
    criterion = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(params.opt_params(), lr=lr, eps=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     [50, 100, 150, 200, 250],
                                                     gamma=0.5,
                                                     last_epoch=-1)

    loss_rec = [[], [], [], []]
    params_rec = []

    print("Init params: ")
    print(params.to_name())

    disStep = [0.1, 0.1, 0.05, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    for epoch in range(num_epoch + 1):
        pred_front, pred_back = render(params, layer0, layer1)
        pred_front_down = downsample(pred_front, downsize).to(device)
        pred_back_down = downsample(pred_back, downsize).to(device)

        pred_gram_front = net.gram(pred_front)
        pred_gram_back = net.gram(pred_back)

        color_loss = criterion(gt_front_down, pred_front_down) + criterion(gt_back_down, pred_back_down)
        gram_loss = criterion(gt_gram_front, pred_gram_front) + criterion(gt_gram_back, pred_gram_back)
        prior_loss =  params.prior_loss()

        optimizer.zero_grad()
        loss = gram_loss * loss_weight[0] + color_loss * loss_weight[1] + prior_loss * loss_weight[2]
        loss_rec[0].append(gram_loss.item() * loss_weight[0])
        loss_rec[1].append(color_loss.item() * loss_weight[1])
        loss_rec[2].append(prior_loss.item() * loss_weight[2])
        loss_rec[3].append(loss.item())
        params_rec.append((epoch, params.to_name(), round(loss.item(), 4)))

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {loss}")
            print(params.to_name())
        if epoch % 50 == 0 or epoch == num_epoch:
            plt.clf()
            plt.plot(np.arange(epoch + 1), loss_rec[0], label="gram_loss")
            plt.plot(np.arange(epoch + 1), loss_rec[1], label="color_loss")
            plt.plot(np.arange(epoch + 1), loss_rec[2], label="prior_loss")
            plt.plot(np.arange(epoch + 1), loss_rec[3], label="total_loss")
            plt.legend()
            plt.savefig(os.path.join(writePath, 'loss.png'))

            with open(os.path.join(writePath, 'params.txt'), 'w') as f:
                for i in params_rec:
                    f.write(str(i) + '\n')
            if epoch == 0:
                pred_front, pred_back = render(params, layer0, layer1, spp=16)
            writeimg(os.path.join(writePath, str(epoch) + "_Tran.png"), pred_back)
            writeimg(os.path.join(writePath, str(epoch) + "_Refl.png"), pred_front)

        try:
            loss.backward()
        except Exception:
            print("Params before error: ", params.to_name())
            raise

        optimizer.step()
        scheduler.step()

        params.correct()

        del pred_back, pred_front, pred_back_down, pred_front_down, pred_gram_front, pred_gram_back
        torch.cuda.empty_cache()

        torch.set_grad_enabled(False)
        def loss_nw_params():
            pred_front, pred_back = render(params, layer0, layer1)
            pred_front_down = downsample(pred_front, downsize).to(device)
            pred_back_down = downsample(pred_back, downsize).to(device)
            pred_gram_front = net.gram(pred_front)
            pred_gram_back = net.gram(pred_back)

            color_loss = criterion(gt_front_down, pred_front_down) + criterion(gt_back_down, pred_back_down)
            gram_loss = criterion(gt_gram_front, pred_gram_front) + criterion(gt_gram_back, pred_gram_back)
            prior_loss = params.prior_loss()

            nw_loss = gram_loss * loss_weight[0] + color_loss * loss_weight[1] + prior_loss * loss_weight[2]

            del pred_back, pred_front, pred_back_down, pred_front_down, pred_gram_front, pred_gram_back
            torch.cuda.empty_cache()
            return nw_loss

        # region optimize discrete parameters
        discrete_idx = epoch // 50
        # uv
        if epoch % 5 == 0:
            origin_params = params.tilesUV[0].clone()
            origin_loss = loss
            f = random.randint(0, 1) * 2 - 1
            params.tilesUV[0] = torch.maximum(params.tilesUV[0] + disStep[discrete_idx] * f * 40,
                                                 torch.tensor(1))
            loss = loss_nw_params()
            if loss > origin_loss:
                loss = origin_loss
                params.tilesUV[0] = origin_params
        if epoch % 5 == 0:
            origin_params = params.tilesUV[1].clone()
            origin_loss = loss
            f = random.randint(0, 1) * 2 - 1
            params.tilesUV[1] = torch.maximum(params.tilesUV[1] + disStep[discrete_idx] * f * 40,
                                                 torch.tensor(1))
            loss = loss_nw_params()
            if loss > origin_loss:
                loss = origin_loss
                params.tilesUV[1] = origin_params
        # psi
        if epoch % 5 == 0:
            origin_params = params.psi[0].clone()
            origin_loss = loss
            f = random.randint(0, 1) * 2 - 1
            params.psi[0] += f * disStep[discrete_idx] * 5
            loss = loss_nw_params()
            if loss > origin_loss:
                loss = origin_loss
                params.psi[0] = origin_params
        if epoch % 5 == 0:
            origin_params = params.psi[1].clone()
            origin_loss = loss
            f = random.randint(0, 1) * 2 - 1
            params.psi[1] += f * disStep[discrete_idx] * 5
            loss = loss_nw_params()
            if loss > origin_loss:
                loss = origin_loss
                params.psi[1] = origin_params
        # gapScaling
        if epoch % 5 == 0:
            origin_params = params.gapScaling[0].clone()
            origin_loss = loss
            f = random.randint(0, 1) * 2 - 1
            params.gapScaling[0] = torch.clip(params.gapScaling[0] + f * disStep[discrete_idx] * 0.5, 0, 1)
            loss = loss_nw_params()
            if loss > origin_loss:
                loss = origin_loss
                params.gapScaling[0] = origin_params
        if epoch % 5 == 0:
            origin_params = params.gapScaling[1].clone()
            origin_loss = loss
            f = random.randint(0, 1) * 2 - 1
            params.gapScaling[1] = torch.clip(params.gapScaling[1] + f * disStep[discrete_idx] * 0.5, 0, 1)
            loss = loss_nw_params()
            if loss > origin_loss:
                loss = origin_loss
                params.gapScaling[1] = origin_params
        # endregion

        torch.set_grad_enabled(True)
        # ==============================================

    params.set_requires_grad(False)

    params.noise = torch.tensor(1.0, device=device)
    front, back = render(params, layer0, layer1, spp=16)
    writeimg(os.path.join(writePath, "optimized_Tran.png"), back)
    writeimg(os.path.join(writePath, "optimized_Refl.png"), front)

    print("\n=== End optimization ===\n")
    return params


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Optimize parameters with differentiable rendering.', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--targetR', type=str,
                        help='Input reflection image.')
    parser.add_argument('--targetT', type=str,
                        help='Input transmission image.')
    parser.add_argument('--init', type=str, default="checkpoint/model.pth",
                        help='Deliver a network path start with \'checkpoint/\' for network initialization.\n'
                             'Deliver a pattern name (can only be plain, satin0, satin1, twill0, or twill1) for random initialization using the prior of specified pattern.\n'
                             'Deliver detailed initial parameters string (e.g. \'twill1_R_0.6,0.66_S_0.75,1.53_T_1.33,0.39_N_0.0_UV_121.8,168.9_Kd_0.456,0.398,0.089,0.286,0.279,0.0_G_0.99,1.0_Rm_0.65,0.62_Tm_2.66,0.79_W_0.71,1.29_Ks_0.325,0.37,0.0,0.351,0.883,0.0_Psi_-30.5,-29.5\').')
    parser.add_argument('--save', type=str, default=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                        help='Result saving directory under \'optimized\'.')
    args = parser.parse_args()

    st = time.time()
    set_seed(0)

    net = paramPredNet().to(device).eval()

    pathR, pathT = args.targetR, args.targetT
    gt_front, gt_back = readimg(pathR), readimg(pathT)
    gt_front = torch.from_numpy(resize(gt_front, output_shape=(FILM_SIZE, FILM_SIZE))).to(device)
    gt_back = torch.from_numpy(resize(gt_back, output_shape=(FILM_SIZE, FILM_SIZE))).to(device)

    if args.init.startswith("checkpoint"):
        checkpoint = torch.load(args.init)
        net.fc_net.load_state_dict(checkpoint['fc_net_state_dict'])
        feature = net.predictHWC(gt_front, gt_back)
        init_params = feature2params(feature)
    elif args.init in ['plain', 'satin0', 'satin1', 'twill0', 'twill1']:
        init_params = Parameters().random_init(args.init)
    else:
        init_params = Parameters().from_name(args.init)

    writePath = './optimized/' + args.save
    os.makedirs(writePath)
    shutil.copyfile('./optimize.py', os.path.join(writePath, 'optimize.py'))
    shutil.copyfile('./render.py', os.path.join(writePath, 'render.py'))
    shutil.copyfile('./utils.py', os.path.join(writePath, 'utils.py'))
    writeimg(os.path.join(writePath, "target_Refl.png"), gt_front)
    writeimg(os.path.join(writePath, "target_Tran.png"), gt_back)

    params = optimize(gt_front, gt_back, init_params, writePath)

    print("cost time: ", time.time() - st)



