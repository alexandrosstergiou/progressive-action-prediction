# The Wisdom of Crowds: Temporal Progressive Attention for Early Action Prediction


<a href="https://alexandrosstergiou.github.io/project_pages/TemPr/index.html">[Project page 🌐]</a> <a href="http://arxiv.org/abs/2204.13340">[ArXiv preprint 📃]</a> <a href="https://youtu.be/dcmd8U47BT8">[Video 🎞️]</a>

![supported versions](https://img.shields.io/badge/python-3.x-brightgreen/?style=flat&logo=python&color=green)
![Library](https://img.shields.io/badge/library-PyTorch-blue/?style=flat&logo=pytorch&color=informational)
![GitHub license](https://img.shields.io/cocoapods/l/AFNetworking)

This is the code implementation for the CVPR'23 paper <a href="http://arxiv.org/abs/2204.13340">The Wisdom of Crowds: Temporal Progressive Attention for Early Action Prediction</a>.

## Abstract
Early action prediction deals with inferring the ongoing action from partially-observed videos, typically at the outset of the video. We propose a bottleneck-based attention model that captures the evolution of the action, through progressive sampling over fine-to-coarse scales. Our proposed <b>Tem</b>poral <b>Pr</b>ogressive (TemPr) model is composed of multiple attention towers, one for each scale. The predicted action label is based on the collective agreement considering confidences of these towers. Extensive experiments over four video datasets showcase state-of-the-art performance on the task of Early Action Prediction across a range of encoder architectures. We demonstrate the effectiveness and
consistency of TemPr through detailed ablations.


<p align="center">
<img src="./figures/TemPr_h_back_hl.png" width="700" height="370" />
</p>



## Dependencies

Ensure that the following packages are installed in your machine:

+ `adaPool` (version >= 0.2)
+ `coloredlogs`  (version >= 14.0)
+ `dataset2database` (version >= 1.1)
+ `einops` (version >= 0.4.0)
+ `ffmpeg-python`  (version >=0.2.0)
+ `imgaug`  (version >= 0.4.0)
+ `opencv-python`  (version >= 4.2.0.32)
+ `ptflops` (version >= 0.6.8)
+ `torch` (version >= 1.9.0)
+ `torchinfo` (version >= 1.5.4)
+ `youtube-dl` (version >= 2020.3.24)

You can install the available PyPi packages with the command below:
```
$ pip install coloredlogs dataset2database einops ffmpeg-python imgaug opencv-python ptflops torch torchvision youtube-dl
```
and compile the `adaPool` package as:
```
$ git clone https://github.com/alexandrosstergiou/adaPool.git && cd adaPool-master/pytorch && make install
--- (optional) ---
$ make test
```


## Datasets

A custom format is used for the train/val label files of each datasets:

|`label`|`youtube_id`/`id`|`time_start`(optional)|`time_end`(optional)|`split`|
|-----|------|-----|-----|----|

This can be done through the scripts provided in `labels`


We have tested our code over the following datasets:
- **UCF-101** : [[link]](https://www.crcv.ucf.edu/data/UCF101.php)
- **Somethong-Something (sub21/v2)** : [[link]](https://developer.qualcomm.com/software/ai-datasets/something-something)
- **EPIC-KITCHENS-100** : [[link]](https://epic-kitchens.github.io/2023)
- **NTU-RGB** : [[link]](https://rose1.ntu.edu.sg/dataset/actionRecognition/)

#### Videos and image-based datasets

Based on the format that the dataset is stored on disk two options are supported by the repo:
- Videos being stored in video files (e.g. `.mp4`,`.avi`,etc.)
- Videos being stored in folders containing their frames in image files (e.g. `.jpg`)

By default it is assumed that the data are in video format however, you can overwrite this by setting the `use_frames` call argument to `True`/`true`.


#### Data directory format

We assume a fixed directory formatting that should be of the following structure:

```
<data>
|
└───<dataset>
        |
        └─── <class_i>
        │     │
        │     │─── <video_id_j>
        │     │         (for datasets w/ videos saved as frames)
        │     │         │
        │     │         │─── frame1.jpg
        │     │         └─── framen.jpg
        │     │    
        │     │─── <video_id_j+1>
        │     │         (for datasets w/ videos saved as frames)
        │     │         │
        │     │         │─── frame1.jpg
        │     │         └─── framen.jpg
       ...   ...
```

## Usage

Training for each of the datasets is done through the homonym `.yaml` configuration scripts in `configs`.

You can also use the argument parsers in `train.py` and `inference.py` for custom arguments.


#### Examples

Train on UCF-101 with observation ratio 0.3, 3 scales, with movinet backbone, with the pretrained UCF-101 backbone checkpoint stored in `weights`, and over 4 gpus:
```
python train.py --video_per 0.3 --num_samplers 3 --gpus 0 1 2 3 --precision mixed --dataset UCF-101 --frame_size 224 --batch_size 64 --data_dir data/UCF-101/ --label_dir /labels/UCF-101 --workers 16 --backbone movinet --end_epoch 70 --pretrained_dir weights/UCF-101/movinet_ada_best.pth

```

Run inference over something-something v2 with TemPr and adaptive ensemble over a single gpu with checkpoint file `my_chckpt.pth`:
```
python inference.py --config config/inference/smthng-smthng/config.yml --head TemPr_h --pool ada --gpus 0 --pretrained_dir my_chckpt.pth
```

#### Calling arguments (for both `train.py` & `inference.py`)

The following arguments are used and can be included at the parser of any training script.

|Argument name | functionality|
| :--------------: | ------- |
| `debug-mode` | Boolean for debugging messages. Useful for custom implementations/datasets. |
| `dataset` | String for the name of the dataset. used in order to obtain the respective configurations. |
| `data_dir` | String for the directory to load data from. |
| `data_dir` | String for the directory to load the train and val splits (should be `train.csv` and `val.csv`). |
| `clip-length` | Integer determining the number of frames to be used for each video. |
| `clip-size` | Tuple for the spatial size (height x width) of each frame.|
| `backbone`| String for the name of the feature extractor network.|
|`accum_grads`| Integer for the number of iterations passed to run backwards. Set to 1 to not use gradient accumulation. |
|`use_frames`| Boolean flag. When set to `True` the dataset directory should be a folder of `.jpg` images. Alternatively, video files. |
| `head`| String for the name of the attention tower network. Only `TemPr_h` can be currently used.|
| `pool` | String for the predictor aggregation method to be used. |
| `gpus` | List for the number of GPUs to be used. |
| `pretrained-3d`| String for `.pth` filepath the case that the weights are to be initialised from some previously trained model. As a non-strict weight loading implementation exists to remove certain works from the `state_dict` keys.|
|`config`| String for the `.yaml` configuration file to be used. If arguments that are part of the configuration path are passed by the user, they will be selected over the YAML ones.|


## Checkpoints




### UCF-101

|Backbone | $\rho=0.1$| $\rho=0.2$| $\rho=0.3$| $\rho=0.4$| $\rho=0.5$| $\rho=0.6$| $\rho=0.7$| $\rho=0.8$| $\rho=0.9$|
| :--------------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| `x3d` | [`chkp`](https://drive.google.com/file/d/12gYiOjLBgeEI-XVPVz4wCFNN7PWRXKbD/view?usp=sharing) | [`chkp`](https://drive.google.com/file/d/1rjSgig7Al7NQXA6j-v6Cs2YLfyhU9v-D/view?usp=sharing) | [`chkp`](https://drive.google.com/file/d/1qACMBrw-rGjTcQni_zxrawDipUCUUrmn/view?usp=sharing) | [`chkp`](https://drive.google.com/file/d/1jg0Zmv41ak7mo9Ay0YYV5sHSqZegLyrh/view?usp=sharing) | [`chkp`](https://drive.google.com/file/d/1rPOlOokVIPp5Absj8GweUCmHnmlgqRt7/view?usp=sharing) | [`chkp`](https://drive.google.com/file/d/1S0k6zsdPdKNxDggvI0W9cr6bt8vLjTc2/view?usp=sharing) | [`chkp`](https://drive.google.com/file/d/1lfLXsbX0Iw1uW_jfBnAtO8IC7ozdahdf/view?usp=sharing) | [`chkp`](https://drive.google.com/file/d/15_UFp3fouXH_dj2HPSN8Xhek9C9dX9Eo/view?usp=sharing) | [`chkp`](https://drive.google.com/file/d/1vOlt_ufN6vZj9p-227hALuVtGnfCkz3L/view?usp=sharing) |
| `movinet` | [`chkp`](https://drive.google.com/file/d/171OCDnXu6jyHmTPsYWCDaB5UDv8F7aKV/view?usp=sharing) | [`chkp`](https://drive.google.com/file/d/1jqhx-jfLgbesyRjDw4BGLOO5fgSDrzmw/view?usp=sharing) | [`chkp`](https://drive.google.com/file/d/12IEDxnZ8WS1f0eoN_CnEMYq1Lig4uZ8J/view?usp=sharing) | [`chkp`](https://drive.google.com/file/d/1XwzPxWngwI44aVQAOYjmjJVkLiH2yVXe/view?usp=sharing) | [`chkp`](https://drive.google.com/file/d/1o0mrVxdn_E62nKq3Xgj34_o3aw4Pvas2/view?usp=sharing) | [`chkp`](https://drive.google.com/file/d/1DovSWihJYMea-FICEh65tS6ln6H81g9w/view?usp=sharing) | [`chkp`](https://drive.google.com/file/d/1I7FmfauszDmlOZuscBiBsS-r2hLtFTHx/view?usp=sharing) | [`chkp`](https://drive.google.com/file/d/1D3R3XFtapQe8z0dTLf-0A5C2mr4CcTiA/view?usp=sharing) | [`chkp`](https://drive.google.com/file/d/1hn5qHo_fQ671o0_VfjL1uOS9YYHgSnXl/view?usp=sharing) |

### SSsub21

|Backbone | $\rho=0.1$| $\rho=0.2$| $\rho=0.3$| $\rho=0.5$| $\rho=0.7$| $\rho=0.9$|
| :--------------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| `movinet` | [`chkp`](https://drive.google.com/file/d/1Nv8TpM05WxehJYfNeHhrH-_eNKgP_tyE/view?usp=sharing) | [`chkp`](https://drive.google.com/file/d/1MXKAyTfAcgVqHAABdsWwLluTcBEvkgRp/view?usp=sharing) | [`chkp`](https://drive.google.com/file/d/1vYqmTEBIYrAH4omII7vA3ZtUjoqTb5F0/view?usp=sharing) | [`chkp`](https://drive.google.com/file/d/1oxo5MknKnGf4c0Z2Pjl8wVbUduHf-BpQ/view?usp=sharing) | [`chkp`](https://drive.google.com/file/d/1J-x4mm1re8ShtheEUXpcPY8MskBMRfgK/view?usp=sharing) | [`chkp`](https://drive.google.com/file/d/1I9Qtmuvj_zbJdNNTlfC5B8pITjlGbwNN/view?usp=sharing) |

## Citation

```bibtex
@inproceedings{stergiou2023wisdom,
    title = {The Wisdom of Crowds: Temporal Progressive Attention for Early Action Prediction},
    author = {Stergiou, Alexandros and Damen, Dima},
    booktitle = {IEEE/CVF Computer Vision and Pattern Recognition (CVPR)},
    year = {2023}
}
```

## License

MIT
