### Overview

Here is the code of our paper: **Woven Fabric Capture with a Reflection-Transmission Photo Pair [SIGGRAPH 2024, Conference]**.

Project homepage: https://wangningbei.github.io/2024/FabricBTDF.html



### Contents

The project's directory structure is as follows:

```
checkpoint/								- Saving trained network.
|	model.pth							- The network our paper used.
maps/									- Spatially-varying maps of each pattern.
|	plain/								- First layer of plain.
|	|	centerUV.exr
|	|	id.exr
|	|	...
|	plain_back/							- Second layer of plain.
|	...									- Other patterns.
optimized/								- Saving optimized results.
target/									- Real captured fabric photos.
|	satin1_R.exr						- Reflection photo of satin1.
|	satin1_T.exr						- Transmission photo of satin1.
|	...
gendataset.py							- Script to generate synthetic dataset.
model.py								- Code of our network.
noise.py								- Noise functions.
optimize.py								- Script to optimize parameters using differentiable rendering.
render.py								- Code for rendering the fabric plane using our BSDF.
sggx.py									- SGGX functions.
train.py								- Script to train the network.
utils.py								- Utils functions (image, vector operators, et al.)
requirements.txt						- Environment requirements file.
Readme.md								- Here!
```



### Environment

Use the following commands to set up the environment:

```bash
cd <project directory>
conda create -n fabricBTDF python=3.9
conda activate fabricBTDF
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

Other versions of torch may work as well, but they have not been fully tested.



### Usage

#### Rendering

Here's an example command to render the fabric plane using our BSDF:

```bash
python render.py --params="twill1_R_0.6,0.66_S_0.75,1.53_T_1.33,1.39_N_0.0_UV_121.8,168.9_Kd_0.456,0.398,0.089,0.286,0.279,0.0_G_0.99,1.0_Rm_0.65,0.62_Tm_2.66,0.79_W_0.71,1.29_Ks_0.325,0.37,0.0,0.351,0.883,0.0_Psi_-30.5,-29.5" --save="./"
```

The `--params` option describes the fabric parameters as a string, defined as follows:

```python
<pattern>_R_<roughness*>_S_<height field scaling*>_T_<thickness*>_N_<noiselevel>_UV_<tilesuv>_Kd_<diffuse reflection>,<diffuse transmission>_G_<gapscaling*>_Rm_<ASGGX roughness*>_Tm_<ASGGX thickness*>_W_<blending weight>,<multiple weight>_Ks_<specular albedo*>_Psi_<twist*>
```

The parameter with `*` has different values for warp and weft, described as `<warp value>,<weft value>` in the string. `<pattern>` can only be `plain`, `satin0`, `satin1`, `twill0`, or `twill1`, where `satin1` and `twill1` mean 90-degree rotations of the satin and twill.

Render results (including reflection and transmission) will be saved in the folder indicated by the `--save` option.

More render settings can be modified in the head of `render.py`. **Note that the render settings are also used for optimization and dataset generation.**



#### Recovery

Our method recovers fabric parameters from a reflection-transmission photo pair. It contains two passes: initialize the parameters using network prediction and further optimize the parameters using differentiable rendering.

Here's an example command to start the recovery pipeline, including both network prediction and optimization:

```bash
python optimize.py --targetT="target/twill1_T.png" --targetR="target/twill1_R.png" --init="checkpoint/model.pth" --save="twill1"
```

where `--targetT` indicates the captured transmission photo and `--targetR` indicates the reflection one. We provided some captured samples in `target/`, where transmission photos are named with the suffix `_T` and reflection photos are named with the suffix `_R`. 

`--init` indicates the initialization method. There are three types of them:

- Deliver a network path starting with `checkpoint/` for network initialization. We provided a pre-trained network `checkpoint/model.pth`.
- Deliver a pattern name (can only be `plain`, `satin0`, `satin1`, `twill0`, or `twill1`) for random initialization using the prior of specified pattern.
- Deliver detailed initial parameters string (e.g. `"twill1_R_0.6,0.66_S_0.75,1.53_T_1.33,0.39_N_0.0_UV_121.8,168.9_Kd_0.456,0.398,0.089,0.286,0.279,0.0_G_0.99,1.0_Rm_0.65,0.62_Tm_2.66,0.79_W_0.71,1.29_Ks_0.325,0.37,0.0,0.351,0.883,0.0_Psi_-30.5,-29.5"`).

Optimized parameters and rendered images will be saved in the folder indicated by the `--save` option.



#### Train

##### dataset

We use synthetic datasets to train the network. To render the dataset:

1. Change line 21 of `render.py` to `SPP = 16` for better rendering, and do not forget to change it back to `SPP = 4` when optimizing (otherwise the optimization would be much more slow).

2. Run the following command to generate a dataset:

   ```bash
   python gendataset.py --path="synthetic" --numpp=1280
   ```

   where `--path` indicates the dataset saving folder under the project directory. `--numpp` indicates the number of samples per pattern.

It will take about an hour (on NVIDIA 4060 Laptop GPU) to render 1280 samples per pattern. After rendering, you can see the dataset named `synthetic` under the project directory. It contains 12800 images (1280 samples per pattern, 5 patterns, and 2 images per sample) and a txt file contains parameters of each sample.



##### training

Run the following command to train a network using the previous dataset:

```bash
python train.py --path="synthetic"
```

where `--path` specifies the dataset path. More training settings can be modified at the head of `train.py`.

Training took about four hours on a single A40 GPU. The trained network will be saved in a subfolder (named with current datetime) of `checkpoint/`.



### BibTex

Please cite our paper for any usage of our code in your work by:

```tex
@inproceedings{Tang:2024:FabricBTDF,
  title={Woven Fabric Capture with a Reflection-Transmission Photo Pair},
  author={Yingjie Tang and Zixuan Li and Milo\v{s} Ha\v{s}an and Jian Yang and Beibei Wang},
  booktitle={Proceedings of SIGGRAPH 2024},
  year={2024}
}
```