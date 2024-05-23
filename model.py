import torch

from utils import *
from torch import nn
from torchvision.models.vgg import *
from torchvision import models, transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'

vggTrans = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


FEATURE_LENGTH = 34


class textDescriptor(nn.Module):
    def __init__(self):
        super(textDescriptor, self).__init__()
        self.outputs = []
        self.net = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device)

        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False

        for i, x in enumerate(self.net):
            if isinstance(x, nn.MaxPool2d):
                self.net[i] = nn.AvgPool2d(kernel_size=2)

        def hook(module, input, output):
            self.outputs.append(output)

        for i in [4, 9, 18, 27]:
            self.net[i].register_forward_hook(hook)

        self.weights = [1, 2, 4, 8, 8]

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()


    # inputs: (b, c, h, w)
    # output: (b, 610304)
    def forward(self, x):
        self.outputs = []
        x = self.net(x)
        self.outputs.append(x)

        result = []
        batch = self.outputs[0].shape[0]

        for i in range(batch):
            temp_result = []
            for j, F in enumerate(self.outputs):
                F_slice = F[i, :, :, :]
                f, s1, s2 = F_slice.shape
                s = s1 * s2
                F_slice = F_slice.view((f, s))

                # Gram matrix
                G = torch.mm(F_slice, F_slice.t()) / s
                temp_result.append(G.flatten())
            temp_result = torch.cat(temp_result)

            result.append(temp_result)
        return torch.stack(result)


    # input: (h, w, c)
    # output: (610304)
    def gram(self, x):
        x = x.permute([2, 0, 1])
        x = x.unsqueeze(dim=0)
        x = vggTrans(x)
        return self.forward(x)[0]


class paramPredNet(nn.Module):
    def __init__(self):
        super(paramPredNet, self).__init__()
        self.leaky_relu_threshold = 0.2

        self.td = textDescriptor()
        self.td_length = 610304 * 2
        self.active_fc_num = 256

        self.multiplier = 3
        self.fc_max = 960

        self.fc_net = nn.Sequential(
            nn.Linear(self.td_length, self.active_fc_num),
            nn.LeakyReLU(self.leaky_relu_threshold, inplace=True),

            nn.Linear(self.active_fc_num, self.active_fc_num),
            nn.LeakyReLU(self.leaky_relu_threshold, inplace=True),

            nn.Linear(self.active_fc_num, self.active_fc_num),
            nn.LeakyReLU(self.leaky_relu_threshold, inplace=True),

            nn.Linear(self.active_fc_num, FEATURE_LENGTH),
        )


    # input: (b, c, h, w)
    # output: (b, parameter vector)
    def forward2(self, front, back):
        td_out = torch.cat([self.td(front), self.td(back)], dim=-1)
        fc_out = self.fc_net(td_out)
        fc_out[:, 5:] = torch.sigmoid(fc_out[:, 5:])
        return fc_out


    # input: (h, w, c)
    # output: (feature vector)
    def predictHWC(self, front, back):
        front = front.permute([2, 0, 1])
        front = vggTrans(front.unsqueeze(dim=0))
        back = back.permute([2, 0, 1])
        back = vggTrans(back.unsqueeze(dim=0))
        outp = self.forward2(front, back)
        return outp[0].detach()


    # input: (h, w, c)
    # output: (610304)
    def gram(self, x):
        return self.td.gram(x)


def feature2params(f):
    type = ('plain', 'twill0', 'twill1', 'satin0', 'satin1')[torch.argmax(f[:5])]
    psi = 0.0
    if type.__contains__('twill'):
        psi = -30

    params = Parameters()
    params.type = type
    params.noise = f[[5]] * 10
    params.tilesUV = f[[6, 7]] * 400
    params.roughness = f[[8, 9]]
    params.thickness = f[[10, 11]] * 5
    params.hfScaling = f[[12, 13]] * 2
    params.gapScaling = f[[14, 15]]
    params.roughness_m = f[[16, 17]]
    params.thickness_m = f[[18, 19]] * 5
    params.kd = f[20:26]
    params.ks = f[26:32]
    params.weights = f[[32, 33]]
    params.weights[1] = params.weights[1] * 2
    params.psi = torch.tensor([psi, psi], dtype=floattype)

    params.correct()
    return params


def params2feature(params):
    params.correct()
    y = torch.zeros(FEATURE_LENGTH)
    y[{'plain': 0,
       'twill0':1,
       'twill1':2,
       'satin0':3,
       'satin1':4,
       }[params.type]] = 1.0
    y[5] = params.noise / 10
    y[[6, 7]] = params.tilesUV / 400
    y[[8, 9]] = params.roughness
    y[[10, 11]] = params.thickness / 5
    y[[12, 13]] = params.hfScaling / 2
    y[[14, 15]] = params.gapScaling
    y[[16, 17]] = params.roughness_m
    y[[18, 19]] = params.thickness_m / 5
    y[20:26] = params.kd
    y[26:32] = params.ks
    y[32] = params.weights[0]
    y[33] = params.weights[1] / 2
    return y


def detailLoss(f1, f2):
    c = nn.L1Loss()
    c2 = nn.CrossEntropyLoss()
    with torch.no_grad():
        tar_type = torch.argmax(f2[:, :5], dim=1)
        loss_dic = {
            "pattern_loss": c2(f1[:, :5], tar_type).item() * 0.2,
            "noise_loss": c(f1[:,5], f2[:,5]).item(),
            "uv_loss": c(f1[:,6:8], f2[:,6:8]).item(),
            "roughness_loss": c(f1[:,[8, 9]], f2[:,[8, 9]]).item(),
            "thickness_loss": c(f1[:,[10, 11]], f2[:,[10, 11]]).item(),
            "hfScaling_loss": c(f1[:,[12, 13]], f2[:,[12, 13]]).item(),
            "gapScaling_loss": c(f1[:,[14, 15]], f2[:,[14, 15]]).item(),
            "roughness2_loss": c(f1[:,[16, 17]], f2[:,[16, 17]]).item(),
            "thickness2_loss": c(f1[:,[18, 19]], f2[:,[18, 19]]).item(),
            "diffuse_loss": c(f1[:, 20:26], f2[:, 20:26]).item(),
            "specular_loss": c(f1[:, 26:32], f2[:, 26:32]).item(),
            "weight1_loss": c(f1[:,32], f2[:,32]).item(),
            "weight2_loss": c(f1[:,33], f2[:,33]).item(),
        }
    return loss_dic
