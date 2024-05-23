
from utils import *
from render import fabric_map, render
from tqdm import tqdm
import argparse
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate dataset for training.')
    parser.add_argument('--path', type=str, default="synthetic",
                        help='Dataset saving path.')
    parser.add_argument('--numpp', type=int, default=1280,
                        help='Samples per pattern.')
    args = parser.parse_args()

    dataset_path = args.path
    os.mkdir(dataset_path)
    with open(f"{dataset_path}/params.txt", 'w') as f:
        tot = 0
        for type in ['plain', 'twill0', 'twill1', 'satin0', 'satin1']:
            for i in tqdm(range(1, args.numpp + 1), file=sys.stdout):
                tot += 1
                param = Parameters().random_init(type).to(device)
                front, back = render(param, fabric_map[param.type]['layer0'], fabric_map[param.type]['layer1'])
                writeimg(f'{dataset_path}/{tot}_front.png', front)
                writeimg(f'{dataset_path}/{tot}_back.png', back)
                f.write(str((tot, param.to_name())) + '\n')

