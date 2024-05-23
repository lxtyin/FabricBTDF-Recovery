from utils import *
import sggx
import cv2
import torch
import math
from noise import sampleNoise, perlinNoise
import numpy as np
import argparse

ID_GAP = 0
ID_WARP = 1
ID_WEFT = 2
FILM_SIZE = 512                 # Output resolution
PLANE_SIZE = 10                 # Entire fabric size
CAMERA_Z = 8                    # Camera position (0, 0, 8)
FOV = 53                        # Camera fov
LIGHT_F_DZ = 8
LIGHT_F_DY = 8                  # Front light postion (0, 8, 8)
LIGHT_F_INTENSITY = 300
LIGHT_B_Z = -9                  # Back light position (0, 0, -9)
LIGHT_B_INTENSITY = 300
SPP = 4
MSCATTER_TYPE = "asggx"         # Multiple scattering method.
USE_TWO_LAYER = True
GAUSSIAN_MASK = True
THICK_SCALE = True
THINLENS_DELTA = True
USE_DIFFUSE = True

SIGMA_LENMASK = 450
THENLENS_GLOW_SIGMA = 20
THINLENS_GLOW_RADIANCE = 8

V_PLANE_SIZE = 2 * CAMERA_Z * math.tan(FOV * PI / 360)    # Visible fabric size
assert (V_PLANE_SIZE <= PLANE_SIZE)

CAMERA_POS  = torch.tensor([V_PLANE_SIZE / 2, -V_PLANE_SIZE / 2, CAMERA_Z])
LIGHT_F_POS = torch.tensor([V_PLANE_SIZE / 2, -V_PLANE_SIZE / 2 + LIGHT_F_DY, LIGHT_F_DZ])
LIGHT_B_POS = torch.tensor([V_PLANE_SIZE / 2, -V_PLANE_SIZE / 2, LIGHT_B_Z])


# Intersect with the fabric plane.
# Same as mitsuba, wi points to the camera and wo points to the light.
# Return: wo_to_front_light, wo_to_back_light, 1.0/distance_to_front_light, 1.0/distance_to_back_light,
#         wi_to_camera, u, v, points. (all of shape(spp, height, width, channel))
def get_intersections(spp=SPP):
    assert (int(math.sqrt(spp)) ** 2 == spp)
    root = torch.sqrt(torch.as_tensor(spp, dtype=floattype))
    root = torch.round(root).int()
    interval = 1 / root
    points = []
    for i in range(root):
        for j in range(root):
            x_offset = i * interval + torch.rand([FILM_SIZE, FILM_SIZE]) / root
            y_offset = j * interval + torch.rand([FILM_SIZE, FILM_SIZE]) / root

            offset = torch.stack([x_offset, -y_offset], dim=2)
            dx = torch.arange(0, FILM_SIZE)
            dy = torch.arange(0, -FILM_SIZE, -1).reshape(FILM_SIZE, 1)
            offset[..., 0] += dx
            offset[..., 1] += dy
            offset *= V_PLANE_SIZE / FILM_SIZE
            pos = torch.concat([offset, torch.zeros(FILM_SIZE, FILM_SIZE, 1)], dim=2)
            points.append(pos)

    points = torch.stack(points)
    uvs = points[..., [0, 1]] / PLANE_SIZE
    uvs[..., 1] += 1
    wo_f = LIGHT_F_POS - points
    wo_b = LIGHT_B_POS - points
    idis_f = 1.0 / length(wo_f)
    idis_b = 1.0 / length(wo_b)
    wo_f = normalize3(wo_f)
    wo_b = normalize3(wo_b)
    wi = normalize3(CAMERA_POS - points)
    us = uvs[..., [0]]
    vs = uvs[..., [1]]
    return (wo_f.to(device), wo_b.to(device), idis_f.to(device), idis_b.to(device),
            wi.to(device), us.to(device), vs.to(device), points.to(device))


set_seed(0) # for default intersection
default_intersections = get_intersections()


class Layer:
    def __init__(self, path, tileWidth, tileHeight, umax_warp, umax_weft, isback):
        idimg = readexr(f"{path}/id.exr").to(device)
        self.id = torch.zeros(idimg.shape[0], idimg.shape[1], 1).to(device)
        self.id[idimg[..., [0]] > 0.1] = ID_WARP
        self.id[idimg[..., [1]] > 0.1] = ID_WEFT

        self.centerUV = readexr(f"{path}/centerUV.exr").to(device)
        self.centerUV = self.centerUV * 2 - 1
        self.centerUV[(self.id == 0).squeeze(-1)] = 1.0

        self.yarnSize = readexr(f"{path}/yarnSize.exr").to(device)
        self.yarnSize[(self.id == 0).squeeze(-1)] = 1.0 # avoid gradient problem

        self.tension = readexr(f"{path}/tension.exr").to(device)
        self.tension = self.tension[..., [0]]

        self.tileWidth = tileWidth
        self.tileHeight = tileHeight
        self.umax_warp = umax_warp
        self.umax_weft = umax_weft
        self.isback = isback


    # Generate normal and orientation map.
    def get_norori(self, _psi):
        tileWidth, tileHeight, yarnType, centerUV, yarnSize =\
            self.tileWidth, self.tileHeight, self.id, self.centerUV, self.yarnSize

        yarnType = yarnType.squeeze(-1)

        umax = torch.ones_like(yarnType)
        umax[yarnType == ID_WARP] = self.umax_warp * PI / 180
        umax[yarnType == ID_WEFT] = self.umax_weft * PI / 180

        psi = _psi.squeeze() * PI / 180

        map_h = yarnType.shape[0]
        map_w = yarnType.shape[1]

        x = (torch.linspace(0, map_h - 1, map_h) + 0.5) / FILM_SIZE
        y = (torch.linspace(0, map_w - 1, map_w) + 0.5) / FILM_SIZE

        x = x.to(device)
        y = y.to(device)

        v, u = torch.meshgrid(x, y, indexing="ij")

        x = u * tileWidth
        y = v * tileHeight

        center_x = (x.long() // tileWidth) * tileWidth + centerUV[..., 0] * tileWidth
        center_y = (y.long() // tileHeight) * tileHeight + (1.0 - centerUV[..., 1]) * tileHeight

        x = x - center_x
        y = -(y - center_y)

        weft_cond = yarnType == ID_WEFT

        tmp = x[weft_cond]
        x[weft_cond] = -y[weft_cond]
        y[weft_cond] = tmp

        u = y / (yarnSize[..., 1] / 2) * umax * (-1 if self.isback else 1)
        v = x * PI / yarnSize[..., 0]

        normal = torch.stack([torch.sin(v), torch.sin(u) * torch.cos(v), torch.cos(u) * torch.cos(v)], dim=-1)

        tmp = yarnSize[..., 0] * torch.sin(umax) >= yarnSize[..., 1]
        normal[tmp] = torch.tensor([0.0, 0.0, 1.0]).to(device)
        normal = normalize3(normal)

        normal[weft_cond] = torch.stack(
            [normal[weft_cond][:, 1], -normal[weft_cond][:, 0], normal[weft_cond][:, 2]], dim=1)

        orientation = torch.stack(
            [-torch.cos(v) * torch.sin(psi), torch.cos(u) * torch.cos(psi) + torch.sin(u) * torch.sin(v) * torch.sin(psi),
             -torch.sin(u) * torch.cos(psi) + torch.cos(u) * torch.sin(v) * torch.sin(psi)], dim=2)
        orientation[tmp] = torch.tensor([0.0, 0.0, 1.0]).to(device)
        orientation = normalize3(orientation)
        orientation[weft_cond] = torch.stack(
            [orientation[weft_cond][:, 1], -orientation[weft_cond][:, 0], orientation[weft_cond][:, 2]], dim=1)

        return normal, orientation


    # Generate normal and orientation map (not repeated) with gap scaling.
    def get_norori_wgap(self, _psi, gapScaling_warp = 1.0, gapScaling_weft = 1.0):
        tileWidth, tileHeight, yarnType, centerUV, _yarnSize = \
            self.tileWidth, self.tileHeight, self.id, self.centerUV, self.yarnSize

        yarnType = yarnType.squeeze(-1)

        umax = torch.ones_like(yarnType)
        umax[yarnType == ID_WARP] = self.umax_warp * PI / 180
        umax[yarnType == ID_WEFT] = self.umax_weft * PI / 180

        psi = _psi.squeeze() * PI / 180

        map_h = yarnType.shape[0]
        map_w = yarnType.shape[1]

        x = (torch.linspace(0, map_h - 1, map_h) + 0.5) / FILM_SIZE
        y = (torch.linspace(0, map_w - 1, map_w) + 0.5) / FILM_SIZE

        x = x.to(device)
        y = y.to(device)

        v, u = torch.meshgrid(x, y, indexing="ij")

        x = u * tileWidth
        y = v * tileHeight

        center_x = (x.long() // tileWidth) * tileWidth + centerUV[..., 0] * tileWidth
        center_y = (y.long() // tileHeight) * tileHeight + (1.0 - centerUV[..., 1]) * tileHeight

        x = x - center_x
        y = -(y - center_y)

        weft_cond = yarnType == ID_WEFT

        tmp = x[weft_cond]
        x[weft_cond] = -y[weft_cond]
        y[weft_cond] = tmp

        yarnSize = _yarnSize.clone()
        yarnSize[yarnType == ID_WEFT] *= torch.tensor([gapScaling_weft, 1, 1], device=device)
        yarnSize[yarnType == ID_WARP] *= torch.tensor([gapScaling_warp, 1, 1], device=device)
        gap_condition = torch.abs(x) >= yarnSize[..., 0] / 2

        u = y / (yarnSize[..., 1] / 2) * umax * (-1 if self.isback else 1)
        v = x * PI / yarnSize[..., 0]

        # u /= umax
        normal = torch.stack([torch.sin(v), torch.sin(u) * torch.cos(v), torch.cos(u) * torch.cos(v)], dim=-1)

        tmp = yarnSize[..., 0] * torch.sin(umax) >= yarnSize[..., 1]
        normal[tmp] = torch.tensor([0.0, 0.0, 1.0]).to(device)
        normal = normalize3(normal)

        normal[weft_cond] = torch.stack(
            [normal[weft_cond][:, 1], -normal[weft_cond][:, 0], normal[weft_cond][:, 2]], dim=1)

        orientation = torch.stack(
            [-torch.cos(v) * torch.sin(psi), torch.cos(u) * torch.cos(psi) + torch.sin(u) * torch.sin(v) * torch.sin(psi),
             -torch.sin(u) * torch.cos(psi) + torch.cos(u) * torch.sin(v) * torch.sin(psi)], dim=2)
        orientation[tmp] = torch.tensor([0.0, 0.0, 1.0]).to(device)
        orientation = normalize3(orientation)
        orientation[weft_cond] = torch.stack(
            [orientation[weft_cond][:, 1], -orientation[weft_cond][:, 0], orientation[weft_cond][:, 2]], dim=1)

        normal[gap_condition, :] = torch.tensor([0, 0, 1.0], device=device)
        orientation[gap_condition, :] = torch.tensor([0, 1.0, 0], device=device)
        return normal, orientation.detach(), gap_condition


    # Precompute data for each pixel, including repeating patterns, height field scaling, adding noise and twisting yarns.
    def precompute(self, params, us, vs):

        specularNoise = hfNoise = 0
        if params.type == 'plain': hfNoise = params.noise
        else: specularNoise = params.noise
        tilesUV = params.tilesUV

        roughness_warp = params.roughness[0]
        roughness_weft = params.roughness[1]
        hfScaling_warp = params.hfScaling[0]
        hfScaling_weft = params.hfScaling[1]
        gapScaling_warp = params.gapScaling[0]
        gapScaling_weft = params.gapScaling[1]
        thickness_warp = params.thickness[0]
        thickness_weft = params.thickness[1]
        roughness_m_warp = params.roughness_m[0]
        roughness_m_weft = params.roughness_m[1]
        thickness_m_warp = params.thickness_m[0]
        thickness_m_weft = params.thickness_m[1]
        weights = params.weights
        specular_warp = params.ks[0:3]
        specular_weft = params.ks[3:6]
        psi_warp = params.psi[0]
        psi_weft = params.psi[1]

        id = self.id.clone()
        centerUV = self.centerUV.clone()
        thickScaleMin = fabric_map[params.type]['sMin']
        thickScale = thickScaleMin + (1 - self.tension) * (1 - thickScaleMin)

        tileWidth = self.tileWidth
        tileHeight = self.tileHeight

        psi = torch.zeros_like(id).to(device)
        psi[id == ID_WARP] = psi_warp
        psi[id == ID_WEFT] = psi_weft
        normal, orientation, gap_condition = self.get_norori_wgap(psi, gapScaling_warp, gapScaling_weft)
        id[gap_condition] = ID_GAP

        u = us * tilesUV[0]
        v = vs * tilesUV[1]
        x = u * tileWidth
        y = v * tileHeight

        # eval from map
        xindexs = (u % 1 * FILM_SIZE).long()[..., 0]
        yindexs = ((1 - v % 1) * FILM_SIZE).long()[..., 0]
        xindexs[xindexs == FILM_SIZE] = FILM_SIZE - 1
        yindexs[yindexs == FILM_SIZE] = FILM_SIZE - 1

        normal = normal[yindexs, xindexs, :]
        orientation = orientation[yindexs, xindexs, :]

        id = id[yindexs, xindexs, :]
        centerUV = centerUV[yindexs, xindexs, :]
        thickScale = thickScale[yindexs, xindexs, :]
        del xindexs, yindexs

        specular = torch.ones_like(normal).to(device)
        specular[id.squeeze(-1) == ID_WARP, :] = specular_warp
        specular[id.squeeze(-1) == ID_WEFT, :] = specular_weft
        specular_m = torch.zeros_like(specular).to(device)
        specular_m[specular != 0] = torch.pow(specular[specular != 0], 1.0 / weights[1])
        hfScaling = torch.ones_like(id).to(device)
        hfScaling[id == ID_WARP] = hfScaling_warp
        hfScaling[id == ID_WEFT] = hfScaling_weft
        roughness = torch.ones_like(id).to(device)
        roughness[id == ID_WARP] = roughness_warp
        roughness[id == ID_WEFT] = roughness_weft
        roughness_m = torch.ones_like(id).to(device)
        roughness_m[id == ID_WARP] = roughness_m_warp
        roughness_m[id == ID_WEFT] = roughness_m_weft
        thickness = torch.ones_like(id).to(device)
        thickness[id == ID_WARP] = thickness_warp
        thickness[id == ID_WEFT] = thickness_weft
        thickness_m = torch.ones_like(id).to(device)
        thickness_m[id == ID_WARP] = thickness_m_warp
        thickness_m[id == ID_WEFT] = thickness_m_weft
        if THICK_SCALE:
            thickness *= thickScale
            thickness_m *= thickScale

        # noise
        if specularNoise != 0:
            fineness = 11 - specularNoise
            index1 = (x[..., 0] * fineness).long()
            index2 = (y[..., 0] * fineness).long()
            xi = sampleNoise(index1, index2)
            intensityVariation = torch.minimum(-torch.log(xi), torch.tensor(10))
            specular = specular * intensityVariation.unsqueeze(-1)
            specular_m = specular_m * intensityVariation.unsqueeze(-1)
            del xi, intensityVariation, index1, index2
            torch.cuda.empty_cache()
        if hfNoise != 0:
            center_x = (x[..., 0].long() // tileWidth) * tileWidth + centerUV[..., 0] * tileWidth
            center_y = (y[..., 0].long() // tileHeight) * tileHeight + (
                    torch.tensor(1.0).to(device) - centerUV[..., 1]) * tileHeight
            center = torch.stack([center_x, center_y], dim=len(center_x.shape))
            pos = center.long()

            point_x = (center_x * (
                        tileHeight * tilesUV[1] + sampleNoise(pos[..., 0], 2 * pos[..., 1])) + center_y) / 100
            point_y = (center_y * (
                        tileWidth * tilesUV[0] + sampleNoise(pos[..., 0], 2 * pos[..., 1] + 1)) + center_x) / 100

            random1 = perlinNoise(point_x, torch.zeros_like(point_x), torch.zeros_like(point_x))
            random2 = perlinNoise(point_y, torch.zeros_like(point_x), torch.zeros_like(point_x))

            noise = sampleNoise(random1 * 100, random2 * 100)

            hfScaling = hfScaling - 0.1 * hfNoise + 0.15 * hfNoise * noise.unsqueeze(-1)
            del point_x, point_y, random1, random2, pos, center, center_y, center_x
            torch.cuda.empty_cache()

        # hfScaling
        n = normal.clone()
        o = orientation.clone()
        n[..., [0, 1]] *= hfScaling
        n = normalize3(n)

        oscale = n[..., [0, 1]] * o[..., [0, 1]]
        oscale = torch.sum(oscale, dim=-1, keepdim=True)
        otmp = oscale == 0
        oscale[otmp] = 1
        oscale = (-n[..., [2]] * o[..., [2]]) / oscale
        oscale[otmp] = 1
        o[..., [0, 1]] *= oscale
        orientation = normalize3(o)
        normal = n
        del n, o
        # end precompute

        # all info our bsdf needs.
        return normal, orientation, id, roughness, thickness, specular, specular_m, roughness_m, thickness_m, weights


    # return shape: (spp, h, w, 3)
    def eval(self, wi, wo, precomputed_data):
        result = torch.zeros_like(wi)

        normal, orientation, id, roughness, thickness, specular, specular_m, roughness_m, thickness_m, weights = precomputed_data


        left = torch.cross(orientation, normal)
        t_wi = to_local(wi, normal, left, orientation)
        t_wo = to_local(wo, normal, left, orientation)
        t_half = normalize3(t_wi + t_wo)
        t_half[length(t_half)[..., 0] == 0] = torch.tensor([0, 0, 1.0], device=device)

        sigmaTi = sggx.sigmaT(t_wi, roughness)
        sigmaTo = sggx.sigmaT(t_wo, roughness)

        i_cosi = wi[..., [2]]
        i_coso = wo[..., [2]]
        i_cosi[i_cosi == 0] = 1
        i_coso[i_coso == 0] = 1

        i_cosi = 1.0 / i_cosi
        i_coso = 1.0 / i_coso

        Ai = sigmaTi * i_cosi
        Ao = sigmaTo * i_coso
        E0, E1 = torch.zeros_like(Ai), torch.zeros_like(Ao)
        E0[Ai <= 0] += Ai[Ai <= 0]
        E0[Ao <= 0] += Ao[Ao <= 0]
        E1[Ai > 0] += Ai[Ai > 0]
        E1[Ao > 0] += Ao[Ao > 0]
        iAio = Ai + Ao
        iAio[iAio == 0] = 1
        iAio = 1.0 / iAio

        D = sggx.D(t_half, roughness)
        G = (torch.exp(thickness * E0) - torch.exp(-thickness * E1)) * iAio

        condition_gap = (id != ID_GAP)[..., 0]

        result[condition_gap] += (specular * D * 0.25 * G * abs(i_cosi))[condition_gap]

        if MSCATTER_TYPE == "sggx":
            D = sggx.D(t_half, roughness_m)
            G = (torch.exp(thickness_m * E0) - torch.exp(-thickness_m * E1)) * iAio
            single2 = specular_m * D * 0.25 * G * abs(i_cosi)

            result[condition_gap] += single2[condition_gap]

        elif MSCATTER_TYPE == "asggx":
            theta_i = torch.atan2(t_wi[..., [2]], torch.sqrt(t_wi[..., [0]] ** 2 + t_wi[..., [1]] ** 2))
            theta_o = torch.atan2(t_wo[..., [2]], torch.sqrt(t_wo[..., [0]] ** 2 + t_wo[..., [1]] ** 2))
            theta_rt_h = (theta_i + theta_o) / 2
            rt_half = torch.cat([torch.cos(theta_rt_h), torch.zeros_like(theta_rt_h), torch.sin(theta_rt_h)], dim=-1)
            D2 = sggx.D(rt_half, roughness_m)
            G2 = (torch.exp(thickness_m * E0) - torch.exp(-thickness_m * E1)) * iAio

            single2 = specular_m * D2 * 0.5 * G2 * abs(i_cosi)

            result[condition_gap] += single2[condition_gap]


        if not torch.all(torch.isfinite(result)):
            raise
        return result


    # return shape: (spp, h, w, 1).
    def eval_attenuation(self, ws, precomputed_data):
        normal, orientation, id, roughness, thickness, specular, specular_m, roughness_m, thickness_m, weights = precomputed_data

        result = torch.zeros_like(roughness)

        left = torch.cross(orientation, normal)
        t_w = to_local(ws, normal, left, orientation)
        sigmaT = sggx.sigmaT(t_w, roughness)
        i_wdn = torch.abs(ws[..., [2]])
        condition = (i_wdn != 0)

        i_wdn[i_wdn == 0] = 1
        i_wdn = 1.0 / i_wdn

        result[condition] = torch.exp(-thickness * sigmaT * i_wdn)[condition]
        condition_gap = (id == ID_GAP)
        result[condition_gap] = 1.0

        if not torch.all(torch.isfinite(result)):
            raise
        return result


fabric_map = {
    "plain": {"layer0": Layer(f"maps/plain", 2, 2, 22, 16, False),
              "layer1": Layer(f"maps/plain_back", 2, 2, 22, 16, True),
              "sMin": 1.0},
    "twill0": {"layer0": Layer(f"maps/twill0", 4, 8, 24, 24, False),
               "layer1": Layer(f"maps/twill0_back", 4, 8, 24, 24, True),
               "sMin": 0.5},
    "twill1": {"layer0": Layer(f"maps/twill1", 8, 4, 24, 24, False),
               "layer1": Layer(f"maps/twill1_back", 8, 4, 24, 24, True),
               "sMin": 0.5},
    "satin0": {"layer0": Layer(f"maps/satin0", 10, 5, 60, 35, False),
               "layer1": Layer(f"maps/satin0_back", 10, 5, 60, 35, True),
               "sMin": 0.5},
    "satin1": {"layer0": Layer(f"maps/satin1", 5, 10, 35, 60, False),
               "layer1": Layer(f"maps/satin1_back", 5, 10, 35, 60, True),
               "sMin": 0.5},
}


def eval_diffuse(wo, prec0, prec1, albedoRT, normalWeight, isreflect):
    if not USE_DIFFUSE:
        return torch.zeros_like(wo)
    n0 = prec0[0]
    n1 = prec1[0]
    id0 = prec0[2]
    id1 = prec1[2]

    kd = albedoRT[0:3] if isreflect else albedoRT[3:6]
    product_cos = torch.ones_like(id0)

    if isreflect:
        product_cos[id0 != ID_GAP] *= torch.clip(dot(wo, n0), 0.0)[id0 != ID_GAP]
        product_cos[(id0 == ID_GAP) & (id1 != ID_GAP)] *= torch.clip(dot(wo, n1), 0.0)[(id0 == ID_GAP) & (id1 != ID_GAP)]
    else:
        product_cos[id0 != ID_GAP] *= torch.clip(dot(-wo, n0), 0.0)[id0 != ID_GAP]
        product_cos[id1 != ID_GAP] *= torch.clip(dot(-wo, n1), 0.0)[id1 != ID_GAP]

    lambertian = ((1 - normalWeight) * kd * abs(wo[..., [2]]) +
                  normalWeight * kd * product_cos) * INV_PI

    gap_condition = ((id0 == ID_GAP) & (id1 == ID_GAP))
    lambertian[gap_condition.squeeze(-1)] = 0.0

    return lambertian


def eval_diffuse_singlelayer(wo, prec0, albedoRT, normalWeight, isreflect):
    if not USE_DIFFUSE:
        return torch.zeros_like(wo)
    n0 = prec0[0]
    id0 = prec0[2]

    kd = albedoRT[0:3] if isreflect else albedoRT[3:6]
    product_cos = torch.ones_like(id0)

    if isreflect:
        product_cos[id0 != ID_GAP] *= torch.clip(dot(wo, n0), 0.0)[id0 != ID_GAP]
    else:
        product_cos[id0 != ID_GAP] *= torch.clip(dot(-wo, n0), 0.0)[id0 != ID_GAP]

    lambertian = ((1 - normalWeight) * kd * abs(wo[..., [2]]) +
                  normalWeight * kd * product_cos) * INV_PI

    gap_condition = id0 == ID_GAP
    lambertian[gap_condition.squeeze(-1)] = 0.0

    return lambertian


# Delta transmission for simulating out-of-focus effects.
def eval_thinlensdelta(wo, prec0, prec1):
    id0 = prec0[2]
    id1 = prec1[2]
    result = torch.zeros_like(wo)
    gap_condition = (id0 == ID_GAP) & (id1 == ID_GAP)
    result[gap_condition.squeeze(-1), :] = 1.0

    delta_mask = gaussian_mask(FILM_SIZE, FILM_SIZE, THENLENS_GLOW_SIGMA).unsqueeze(-1).unsqueeze(0).to(device)
    delta_mask = delta_mask.expand(result.shape[0], FILM_SIZE, FILM_SIZE, 3)
    result *= delta_mask
    return result


# Return both reflection and transmission render results.
def render(params, layer0, layer1, spp=None):
    if spp == None:
        spp = SPP
        WO_Fs, WO_Bs, IDIS_Fs, IDIS_Bs, WIs, Us, Vs, POINTs = default_intersections
    else:
        WO_Fs, WO_Bs, IDIS_Fs, IDIS_Bs, WIs, Us, Vs, POINTs = get_intersections(spp)

    if USE_TWO_LAYER:
        precompute_0 = layer0.precompute(params, Us, Vs)
        precompute_1 = layer1.precompute(params, Us, Vs)

        # front ===========================

        specular0 = layer0.eval(WIs, WO_Fs, precompute_0)
        specular1 = layer1.eval(WIs, WO_Fs, precompute_1)

        attenuation = (layer0.eval_attenuation(WO_Fs, precompute_0) *
                       layer0.eval_attenuation(WIs, precompute_0))

        diffuse = eval_diffuse(WO_Fs, precompute_0, precompute_1, params.kd, params.weights[0], isreflect=True)

        front = specular0 + specular1 * attenuation + diffuse

        front = front * LIGHT_F_INTENSITY * IDIS_Fs * IDIS_Fs
        front = torch.sum(front, dim=0) / spp
        del specular0, specular1, attenuation

        # back ===========================

        specular0 = layer0.eval(WIs, WO_Bs, precompute_0)
        specular1 = layer1.eval(WIs, WO_Bs, precompute_1)

        attenuation0 = layer1.eval_attenuation(WO_Bs, precompute_1)
        attenuation1 = layer0.eval_attenuation(WIs, precompute_0)

        diffuse = eval_diffuse(WO_Bs, precompute_0, precompute_1, params.kd, params.weights[0], isreflect=False)

        back = specular0 * attenuation0 + specular1 * attenuation1 + diffuse

        back = back * LIGHT_B_INTENSITY * IDIS_Bs * IDIS_Bs
        if THINLENS_DELTA:
            delta = eval_thinlensdelta(WO_Bs, precompute_0, precompute_1)
            back += delta * THINLENS_GLOW_RADIANCE
        back = torch.sum(back, dim=0) / spp
        del specular0, specular1, attenuation0, attenuation1
    else:
        # layer1 unused
        precompute_0 = layer0.precompute(params, Us, Vs)

        # front ===========================

        front = layer0.eval(WIs, WO_Fs, precompute_0)
        front += eval_diffuse_singlelayer(WO_Fs, precompute_0, params.kd, params.weights[0], isreflect=True)

        front = front * LIGHT_F_INTENSITY * IDIS_Fs * IDIS_Fs
        front = torch.sum(front, dim=0) / spp

        # back ===========================

        back = layer0.eval(WIs, WO_Bs, precompute_0)
        back += eval_diffuse_singlelayer(WO_Bs, precompute_0, params.kd, params.weights[0], isreflect=False)

        back = back * LIGHT_B_INTENSITY * IDIS_Bs * IDIS_Bs
        if THINLENS_DELTA:
            delta = eval_thinlensdelta(WO_Bs, precompute_0, precompute_0)
            back += delta * THINLENS_GLOW_RADIANCE
        back = torch.sum(back, dim=0) / spp

    front = torch.minimum(front, torch.tensor(1.0))
    back = torch.minimum(back, torch.tensor(1.0))
    # avoid gradient error.
    front[front != 0] = torch.pow(front[front != 0], 1.0 / 2.2)
    back[back != 0] = torch.pow(back[back != 0], 1.0 / 2.2)

    if GAUSSIAN_MASK:
        mask = gaussian_mask(FILM_SIZE, FILM_SIZE, SIGMA_LENMASK).unsqueeze(-1).to(device)
        front *= mask
        back *= mask

    if not torch.all(torch.isfinite(back)):
        raise
    return front, back



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Rendering the fabric plane using our BSDF.')
    parser.add_argument('--params', type=str,
                        help='Parameters string (e.g. \'twill1_R_0.6,0.66_S_0.75,1.53_T_1.33,0.39_N_0.0_UV_121.8,168.9_Kd_0.456,0.398,0.089,0.286,0.279,0.0_G_0.99,1.0_Rm_0.65,0.62_Tm_2.66,0.79_W_0.71,1.29_Ks_0.325,0.37,0.0,0.351,0.883,0.0_Psi_-30.5,-29.5\').')
    parser.add_argument('--save', type=str, default="./",
                        help='Saving path.')
    
    args = parser.parse_args()

    params = Parameters().from_name(args.params).to(device)
    layer0 = fabric_map[params.type]["layer0"]
    layer1 = fabric_map[params.type]["layer1"]
    front, back = render(params, layer0, layer1)

    writeimg(os.path.join(args.save, "rendered_Tran.png"), back)
    writeimg(os.path.join(args.save, "rendered_Refl.png"), front)


