
import numpy as np
import torch
import torch.nn as nn

device = 'cuda'


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def hash(coords, log2_hashmap_size=16):
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]

    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]

    return torch.tensor((1 << log2_hashmap_size) - 1).to(xor_result.device) & xor_result


def get_voxel_vertices(xy, bounding_box, resolution, log2_hashmap_size):
    box_min, box_max = bounding_box

    if not torch.all(xy <= box_max) or not torch.all(xy >= box_min):
        xy = torch.clamp(xy, min=0, max=10000)

    grid_size = (box_max - box_min) / resolution

    bottom_left_idx = torch.floor((xy - box_min) / grid_size).int()
    voxel_min_vertex = bottom_left_idx * grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0, 1.0], device=device) * grid_size

    hashed_voxel_indices = hash(bottom_left_idx, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices


class HashEmbedder(nn.Module):

    def __init__(self, bounding_box, log2_hashmap_size=16, base_resolution=10000, finest_resolution=10000):
        super(HashEmbedder, self).__init__()
        self.bounding_box = bounding_box

        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)

        self.embeddings = nn.Embedding(2 ** self.log2_hashmap_size, 1)
        nn.init.uniform_(self.embeddings.weight, a=0.0, b=1.0)

    def forward(self, x):
        resolution = torch.floor(self.base_resolution * 1.0)
        voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices = get_voxel_vertices(x, self.bounding_box, resolution, self.log2_hashmap_size)
        voxel_embedds = self.embeddings(hashed_voxel_indices)
        x_embedds = voxel_embedds.squeeze(-1)
        return x_embedds


bounding_box = torch.tensor([[0, 0], [20000, 20000]]).to(device) 
hashEmbedder = HashEmbedder(bounding_box=bounding_box, log2_hashmap_size=16, base_resolution=20000, finest_resolution=20000).to(device)


def sampleNoise(index1, index2):
    with torch.no_grad():
        pos = torch.stack([index1, index2], dim=-1).to(device)
        res = hashEmbedder(pos)
        return res


_GRAD3 = ((1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
          (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
          (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
          (1, 1, 0), (0, -1, 1), (-1, 1, 0), (0, -1, -1))
_GRAD3 = torch.as_tensor(_GRAD3).to(device)

permutation = (151, 160, 137, 91, 90, 15,
               131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23,
               190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33,
               88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166,
               77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244,
               102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196,
               135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123,
               5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42,
               223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
               129, 22, 39, 253, 9, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228,
               251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107,
               49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254,
               138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180)

period = len(permutation)

permutation = torch.as_tensor(permutation * 2).to(device)


def lerp(t, a, b):
    return a + t * (b - a)


def grad3(hash, x, y, z):
    g = _GRAD3[hash % 16]
    return x * g[..., 0] + y * g[..., 1] + z * g[..., 2]


def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlinNoise(x, y, z):
    X = torch.floor(x).long() % 255
    Y = torch.floor(y).long() % 255
    Z = torch.floor(z).long() % 255
    x -= torch.floor(x)
    y -= torch.floor(y)
    z -= torch.floor(z)

    u = x ** 3 * (x * (x * 6 - 15) + 10)
    v = y ** 3 * (y * (y * 6 - 15) + 10)
    w = z ** 3 * (z * (z * 6 - 15) + 10)

    perm = permutation

    A = perm[X] + Y
    AA = perm[A] + Z
    AB = perm[A + 1] + Z
    B = perm[X + 1] + Y
    BA = perm[B] + Z
    BB = perm[B + 1] + Z

    res = lerp(w, lerp(v, lerp(u, grad3(perm[AA], x, y, z),
                               grad3(perm[BA], x - 1, y, z)),
                       lerp(u, grad3(perm[AB], x, y - 1, z),
                            grad3(perm[BB], x - 1, y - 1, z))),
               lerp(v, lerp(u, grad3(perm[AA + 1], x, y, z - 1),
                            grad3(perm[BA + 1], x - 1, y, z - 1)),
                    lerp(u, grad3(perm[AB + 1], x, y - 1, z - 1),
                         grad3(perm[BB + 1], x - 1, y - 1, z - 1)))) + 0.5
    return res.squeeze(-1)
