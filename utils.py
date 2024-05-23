import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import math
import torch
import cv2
import random
import rawpy
import colorsys
from piqa.ssim import ssim
from piqa.utils.functional import gaussian_kernel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
floattype = torch.float32
torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(floattype)

PI = math.pi
INV_PI = 1.0 / PI
INV_2PI = 1.0 / (PI * 2)


# region image operators ================

# return shape: (h, w, c), in linear space.
def readexr(path):
    assert path[-3:] == 'exr'
    img = torch.from_numpy(cv2.imread(path, cv2.IMREAD_UNCHANGED)[..., [2, 1, 0]]).type(floattype)
    return img


# return shape: (h, w, c), in sRGB space.
def readimg(path):
    img = torch.from_numpy(cv2.imread(path)[..., [2, 1, 0]]).type(floattype)
    img = img / 255
    return img


# input shape: (h, w, c), in linear space.
def writeexr(path, img):
    assert path[-3:] == 'exr'
    img = img.type(floattype).cpu().numpy()
    img = img[..., [2, 1, 0]]
    cv2.imwrite(path, img)


# input shape: (h, w, c), in sRGB space.
def writeimg(path, img):
    img = torch.minimum(img, torch.tensor(1.0))
    img = (img * 255).type(torch.int)
    img = img.cpu().numpy()
    img = img[..., [2, 1, 0]]
    cv2.imwrite(path, img)


# input shape: (h, w, 3)
# return shape: (1, c, h, w)
def downsample(img, downsize):
    img = img.permute([2, 0, 1]).unsqueeze(0)
    res = torch.nn.functional.interpolate(img, size=downsize, mode='bilinear', align_corners=False)
    res = res.squeeze(0).permute([1, 2, 0])
    return res

# endregion


# region params ================

priors = {
    "plain": {
        "hfScaling": [1.0, 1.0],
        "gapScaling": [0.75, 0.1],
        "psi": 0.0,
        "uv": (173, 173),
    },
    "twill0": {
        "hfScaling": [1.0, 0.5],
        "gapScaling": [0.9, 0.05],
        "psi": -30.0,
        "uv": (122, 80),
    },
    "twill1": {
        "hfScaling": [1.0, 0.5],
        "gapScaling": [0.9, 0.05],
        "psi": -30.0,
        "uv": (80, 122),
    },
    "satin0": {
        "hfScaling": [0.1, 0.5],
        "gapScaling": [0.9, 0.05],
        "psi": 0.0,
        "uv": (80, 160),
    },
    "satin1": {
        "hfScaling": [0.1, 0.5],
        "gapScaling": [0.9, 0.05],
        "psi": 0.0,
        "uv": (160, 80),
    }
}

class Parameters:
    def __init__(self):
        pass
    def random_init(self, type=None):
        if type == None:
            self.type = ('plain', 'twill0', 'twill1', 'satin0', 'satin1')[random.randint(0, 4)]
        else:
            self.type= type

        self.noise = torch.tensor(1)

        tilesU, tilesV = priors[self.type]['uv']
        if self.type.__contains__('twill'):
            s = np.random.uniform(0.7, 1.8)
            tilesU *= s * np.random.uniform(0.9, 1.1)
            tilesV *= s * np.random.uniform(0.9, 1.1)
        elif self.type.__contains__('satin'):
            s = np.random.uniform(0.7, 1.5)
            tilesU *= s * np.random.uniform(0.9, 1.1)
            tilesV *= s * np.random.uniform(0.9, 1.1)
        elif self.type.__contains__('plain'):
            s = np.random.uniform(0.7, 1.3)
            tilesU *= s * np.random.uniform(0.9, 1.1)
            tilesV = tilesU

        self.tilesUV = torch.tensor([tilesU, tilesV])

        def getkd():
            pass
            r = random.randint(0, 100) / 100
            if random.uniform(0, 10) > 9 and r != 0:
                return np.round([r, r, r], 2)
            g = random.randint(0, 100) / 100
            b = random.randint(0, 100) / 100
            if r == 0 and g == 0 and b == 0:
                r = random.randint(1, 100) / 100
                g = random.randint(1, 100) / 100
                b = random.randint(1, 100) / 100
            kd = np.asarray([r, g, b]).round(2)
            return kd

        def getks(kd):
            return np.power(kd, random.uniform(0, 1)).round(2)

        km_warp = getkd()
        if random.randint(0, 100) > 70:
            km_weft = km_warp
        else:
            km_weft = getkd()
        ks_warp = getks(km_warp)
        ks_weft = getks(km_weft)
        kdR = (km_warp + km_weft + getkd()) / 3 / 2
        kdT = (km_warp + km_weft + getkd()) / 3 / 2

        psi = priors[self.type]['psi']

        if type.__contains__('twill'):
            thickness_warp = random.uniform(1.0, 4.0)
            thickness_weft = random.uniform(1.0, 4.0)
            hfScaling_warp = random.uniform(1.0, 2.0)
            hfScaling_weft = random.uniform(1.0, 2.0)
            roughness_warp = random.uniform(0.7, 1.0) ** 2
            roughness_weft = random.uniform(0.7, 1.0) ** 2
            gapScaling_warp = random.uniform(0.85, 0.95)
            gapScaling_weft = random.uniform(0.85, 0.95)
            if type == 'twill0':
                thickness_warp, thickness_weft = thickness_weft, thickness_warp
        elif type.__contains__('plain'):
            ks_warp = ks_weft
            km_warp = km_weft
            thickness_warp = thickness_weft = random.uniform(0.5, 3.0)
            hfScaling_warp = hfScaling_weft = random.uniform(0.2, 2.0)
            roughness_warp = roughness_weft = random.uniform(0.1, 0.7) ** 2
            gapScaling_warp = gapScaling_weft = random.uniform(0.6, 0.9)
        elif type.__contains__('satin'):
            thickness_warp = random.uniform(2, 4)
            thickness_weft = random.uniform(0.5, 2)
            hfScaling_warp = random.uniform(0.2, 1)
            hfScaling_weft = random.uniform(0.2, 1)
            roughness_warp = random.uniform(0.7, 1.0) ** 2
            roughness_weft = random.uniform(0.2, 0.7) ** 2
            gapScaling_warp = random.uniform(0.85, 0.95)
            gapScaling_weft = random.uniform(0.85, 0.95)
            if type == 'satin1':
                thickness_warp, thickness_weft = thickness_weft, thickness_warp
                roughness_warp, roughness_weft = roughness_weft, roughness_warp

        normal_weight = random.uniform(0.0, 1.0)
        multiple_weight = random.uniform(0.01, 2.0)

        roughness_m_scale = random.uniform(0.5, 1.5)
        thickness_m_scale = random.uniform(0.5, 1.5)

        self.roughness = torch.tensor([roughness_warp, roughness_weft], dtype=floattype)
        self.thickness = torch.tensor([thickness_warp, thickness_weft], dtype=floattype)
        self.hfScaling = torch.tensor([hfScaling_warp, hfScaling_weft], dtype=floattype)
        self.gapScaling = torch.tensor([gapScaling_warp, gapScaling_weft], dtype=floattype)
        self.roughness_m = torch.tensor([roughness_warp, roughness_weft], dtype=floattype) * roughness_m_scale
        self.thickness_m = torch.tensor([thickness_warp, thickness_weft], dtype=floattype) * thickness_m_scale
        self.kd = torch.tensor([kdR[0], kdR[1], kdR[2],
                                kdT[0], kdT[1], kdT[2]], dtype=floattype) # kdR, kdT
        self.ks = torch.tensor([ks_warp[0], ks_warp[1], ks_warp[2],
                                ks_weft[0], ks_weft[1], ks_weft[2]], dtype=floattype)
        self.weights = torch.tensor([normal_weight, multiple_weight], dtype=floattype)
        self.psi = torch.tensor([psi, psi], dtype=floattype)
        self.correct()
        return self

    def from_name(self, name):
        ls = name.split('_')
        dic = {}
        cur = ""
        for s in ls[1:]:
            if any(c.isalpha() for c in s):
                cur = s
            else:
                dic[cur] = eval(s)

        self.type = ls[0]
        self.noise = torch.tensor(dic['N'])
        self.tilesUV = torch.tensor(dic['UV'])

        self.roughness = torch.tensor([dic['R'][0], dic['R'][1]])
        self.thickness = torch.tensor([dic['T'][0], dic['T'][1]])
        self.gapScaling = torch.tensor([dic['G'][0], dic['G'][1]])
        self.hfScaling = torch.tensor([dic['S'][0], dic['S'][1]])
        self.roughness_m = torch.tensor([dic['Rm'][0], dic['Rm'][1]])
        self.thickness_m = torch.tensor([dic['Tm'][0], dic['Tm'][1]])
        self.kd = torch.tensor([
            dic['Kd'][0], dic['Kd'][1], dic['Kd'][2],
            dic['Kd'][3], dic['Kd'][4], dic['Kd'][5]])
        self.ks = torch.tensor([
            dic['Ks'][0], dic['Ks'][1], dic['Ks'][2],
            dic['Ks'][3], dic['Ks'][4], dic['Ks'][5],])
        self.weights = torch.tensor(dic['W'])
        self.psi = torch.tensor([dic['Psi'][0], dic['Psi'][1]])

        self.correct()
        return self

    def to_name(self):
        def rstr(x, rd=2):
            if len(x.shape) == 0:
                return round(x.item(), rd)
            else:
                s = ""
                for i in x:
                    s = s + f",{round(i.item(), rd)}"
                return s[1:]

        name = (f"{self.type}_R_{rstr(self.roughness)}_S_{rstr(self.hfScaling)}_T_{rstr(self.thickness)}_N_{rstr(self.noise, 0)}"
                f"_UV_{rstr(self.tilesUV, 1)}_Kd_{rstr(self.kd, 3)}_G_{rstr(self.gapScaling)}"
                f"_Rm_{rstr(self.roughness_m)}_Tm_{rstr(self.thickness_m)}_W_{rstr(self.weights)}"
                f"_Ks_{rstr(self.ks, 3)}_Psi_{rstr(self.psi)}")
        return name

    def flip(self):
        self.type = {
            "plain": "plain",
            "twill0": "twill1",
            "twill1": "twill0",
            "satin0": "satin1",
            "satin1": "satin0"
        }[self.type]
        self.tilesUV = self.tilesUV[[1, 0]]
        self.roughness = self.roughness[[1, 0]]
        self.roughness_m = self.roughness_m[[1, 0]]
        self.thickness = self.thickness[[1, 0]]
        self.thickness_m = self.thickness_m[[1, 0]]
        self.hfScaling = self.hfScaling[[1, 0]]
        self.gapScaling = self.gapScaling[[1, 0]]
        self.psi = self.psi[[1, 0]]
        self.ks = self.ks[[3, 4, 5, 0, 1, 2]]
        return self

    def prior_loss(self):
        loss = 0.0

        hfScalings = self.hfScaling
        mean, std = priors[self.type]["hfScaling"]
        loss += -torch.log(torch.exp(-(hfScalings - mean) ** 2 / (2 * std ** 2))).mean()

        gapScalings = self.gapScaling
        mean, std = priors[self.type]["gapScaling"]
        loss += -torch.log(torch.exp(-(gapScalings - mean) ** 2 / (2 * std ** 2))).mean()

        return loss

    def correct(self):
        with torch.no_grad():
            self.noise.data = torch.clip(self.noise, 0, 10)
            self.tilesUV.data = torch.clip(self.tilesUV, 1, 400)
            self.roughness.data = torch.clip(self.roughness, 0.01, 1)
            self.roughness_m.data = torch.clip(self.roughness_m, 0.01, 1)
            self.hfScaling.data = torch.clip(self.hfScaling, 0.01, 2)
            self.gapScaling.data = torch.clip(self.gapScaling, 0.01, 1)
            self.thickness.data = torch.clip(self.thickness, 0.01, 5)
            self.thickness_m.data = torch.clip(self.thickness_m, 0.01, 5)
            self.kd.data = torch.clip(self.kd, 0, 1)
            self.ks.data = torch.clip(self.ks, 0, 1)
            self.weights.data[0] = torch.clip(self.weights.data[0], 0.0, 1.0)
            self.weights.data[1] = torch.clip(self.weights.data[1], 0.01, 2.0)
        return self

    def to(self, device):
        self.noise = self.noise.to(device)
        self.tilesUV = self.tilesUV.to(device)
        self.roughness = self.roughness.to(device)
        self.roughness_m = self.roughness_m.to(device)
        self.hfScaling = self.hfScaling.to(device)
        self.gapScaling = self.gapScaling.to(device)
        self.thickness = self.thickness.to(device)
        self.thickness_m = self.thickness_m.to(device)
        self.psi = self.psi.to(device)
        self.kd = self.kd.to(device)
        self.ks = self.ks.to(device)
        self.weights = self.weights.to(device)
        return self

    def set_requires_grad(self, tf):
        ls = self.opt_params()
        for i in ls:
            i.requires_grad = tf

    def opt_params(self):
        return [self.roughness,
                self.roughness_m,
                self.hfScaling,
                self.thickness,
                self.thickness_m,
                self.kd,
                self.ks,
                self.weights]

# endregion


# region vector ================

# input: v(..., 3)
# returns: len(..., 1)
def length(v):
    v2 = v ** 2
    s = torch.sum(v2, dim=-1, keepdim=True)
    len = torch.sqrt(s)
    return len


# input: (..., 3)
def normalize3(v):
    v2 = torch.square(v)
    s = torch.sum(v2, dim=-1, keepdim=True)
    s[s == 0] = 1  # sqrt(0) is Non-differentiable
    len = torch.sqrt(s)
    return v / len


# input: (..., 3)
def dot(a, b):
    c = a * b
    c = torch.sum(c, dim=-1, keepdim=True)
    return c


# inputs: (..., 3)
def to_local(wi, s, t, n):
    x = torch.sum(wi * s, dim=len(wi.shape) - 1)
    y = torch.sum(wi * t, dim=len(wi.shape) - 1)
    z = torch.sum(wi * n, dim=len(wi.shape) - 1)
    wi = torch.stack((x, y, z), dim=len(wi.shape) - 1)
    return wi

# endregion


# region other ================
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def gaussian_mask(h, w, sigma):
    x = torch.arange(h)
    y = torch.arange(w)
    x, y = torch.meshgrid(x, y)
    x = x + 0.5 - h / 2
    y = y + 0.5 - w / 2
    d2 = x * x + y * y
    e = torch.exp(-d2 / sigma ** 2)
    e_mx = torch.max(e)
    e /= e_mx
    return e

# endregion