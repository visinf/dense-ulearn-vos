# Dense Unsupervised Learning for Video Segmentation

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the official implementation of our paper:

**Dense Unsupervised Learning for Video Segmentation**<br>
[Nikita Araslanov](https://arnike.github.io), [Simone Schaub-Mayer](https://schaubsi.github.io) and [Stefan Roth](https://www.visinf.tu-darmstadt.de/visinf/team_members/sroth/sroth.en.jsp)<br>
To appear at NeurIPS*2021. [[paper](https://openreview.net/pdf?id=i8kfkuiCJCI)] [[supp](https://openreview.net/attachment?id=i8kfkuiCJCI&name=supplementary_material)] [[talk](https://youtu.be/tSBWZ6nYld0)] [[example results](https://youtu.be/BqVtZJSLOzg)] [[arXiv](https://arxiv.org/abs/2111.06265)]

| <img src="assets/examples.gif" alt="drawing" width="420"/><br> |
|:--:|
| <p align="left">We efficiently learn spatio-temporal correspondences  <br> without any supervision, and achieve state-of-the-art <br>accuracy of video object segmentation.</p> |


Contact: Nikita Araslanov *fname.lname* (at) visinf.tu-darmstadt.de


---

## Installation
**Requirements.** To reproduce our results, we recommend Python >=3.6, PyTorch >=1.4, CUDA >=10.0. At least one Titan X GPUs (12GB) or equivalent is required.
The code was primarily developed under PyTorch 1.8 on a single A100 GPU.

The following steps will set up a local copy of the repository.

1. Create conda environment:
```
conda create --name dense-ulearn-vos
source activate dense-ulearn-vos
```

2. Install PyTorch >=1.4 (see [PyTorch instructions](https://pytorch.org/get-started/locally/)). For example on Linux, run:

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

3. Install the dependencies:
```
pip install -r requirements.txt
```

4. Download the data:

| Dataset | Website | Target directory with video sequences |
|:-:|:-:|:--|
| YouTube-VOS | [Link](https://competitions.codalab.org/competitions/19544#participate-get-data) | `data/ytvos/train/JPEGImages/` |
| OxUvA | [Link](https://oxuva.github.io/long-term-tracking-benchmark/) | `data/OxUvA/images/dev/` |
| TrackingNet | [Link](https://github.com/SilvioGiancola/TrackingNet-devkit) | `data/tracking/train/jpegs/` |
| Kinetics-400 | [Link](https://deepmind.com/research/open-source/kinetics) | `data/kinetics400/video_jpeg/train/` |

The last column in this table specifies a path to subdirectories (relative to the project root) containing images of video frames.
You can obviously use a different path structure.
In this case, you will need to adjust the paths in `data/filelists/` for every dataset accordingly.

5. Download filelists:
```
cd data/filelists
bash download.sh
```
This will download lists of training and validation paths for all datasets.

## Training
We following bash script will train a ResNet-18 model from scratch on one of the four supported datasets (see above):
```
bash ./launch/train.sh [ytvos|oxuva|track|kinetics]
```

We also provide our final models for download.

| Dataset | Mean J&F (DAVIS-2017) | Link | MD5 |
|---|:-:|:--:|---|
| OxUvA | 65.3 | [oxuva_e430_res4.pth (132M)](https://download.visinf.tu-darmstadt.de/data/2021-neurips-araslanov-vos/snapshots/oxuva_e430_res4.pth) | `af541[...]d09b3` |
| YouTube-VOS | 69.3 | [ytvos_e060_res4.pth (132M)](https://download.visinf.tu-darmstadt.de/data/2021-neurips-araslanov-vos/snapshots/ytvos_e060_res4.pth) | `c3ae3[...]55faf` |
| TrackingNet | 69.4 | [trackingnet_e088_res4.pth (88M)](https://download.visinf.tu-darmstadt.de/data/2021-neurips-araslanov-vos/snapshots/trackingnet_e088_res4.pth) | `3e7e9[...]95fa9` |
| Kinetics-400 | 68.7 | [kinetics_e026_res4.pth (88M)](https://download.visinf.tu-darmstadt.de/data/2021-neurips-araslanov-vos/snapshots/kinetics_e026_res4.pth) | `086db[...]a7d98` |


## Inference and evaluation

### Inference

To run the inference use `launch/infer_vos.sh`:
```
bash ./launch/infer_vos.sh [davis|ytvos]
```
The first argument selects the validation dataset to use (`davis` for DAVIS-2017; `ytvos` for YouTube-VOS).
The bash variables declared in the script further help to set up the paths for reading the data and the pre-trained models as well as the output directory:
* `EXP`, `RUN_ID` and `SNAPSHOT` determine the pre-trained model to load.
* `VER` specifies a suffix for the output directory (in case you would like to experiment with different configurations for label propagation).
Please, refer to `launch/infer_vos.sh` for their usage.

The inference script will create two directories with the result: `[res3|res4|key]_vos` and `[res3|res4|key]_vis`, where the prefix corresponds to the codename of the output CNN layer used in the evaluation (selected in `infer_vos.sh` using `KEY` variable).
The `vos`-directory contains the segmentation result ready for evaluation; the `vis`-directory produces the results for visualisation purposes.
You can optionally disable generating the visualisation by setting `VERBOSE=False` in `infer_vos.py`.


### Evaluation: DAVIS-2017

Please use the official [evaluation package](https://github.com/davisvideochallenge/davis2017-evaluation).
Install the repository, then simply run:
```
python evaluation_method.py --task semi-supervised --davis_path data/davis2017 --results_path <path-to-vos-directory>
```

### Evaluation: YouTube-VOS 2018
Please use the official [CodaLab evaluation server](https://competitions.codalab.org/competitions/19544#participate-submit_results).
To create the submission, rename the `vos`-directory to `Annotations` and compress it to `Annotations.zip` for uploading.

## Acknowledgements

We thank PyTorch contributors and [Allan Jabri](https://ajabri.github.io) for releasing [their implementation](https://github.com/ajabri/videowalk) of the label propagation.

## Citation
We hope you find our work useful. If you would like to acknowledge it in your project, please use the following citation:
```
@inproceedings{Araslanov:2021:DUL,
  author    = {Araslanov, Nikita and Simone Schaub-Mayer and Roth, Stefan},
  title     = {Dense Unsupervised Learning for Video Segmentation},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {34},
  year = {2021}
}
```
