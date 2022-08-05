[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/flow-guided-sparse-transformer-for-video/deblurring-on-dvd-1)](https://paperswithcode.com/sota/deblurring-on-dvd-1?p=flow-guided-sparse-transformer-for-video)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/flow-guided-sparse-transformer-for-video/deblurring-on-dvd)](https://paperswithcode.com/sota/deblurring-on-dvd?p=flow-guided-sparse-transformer-for-video)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/flow-guided-sparse-transformer-for-video/deblurring-on-gopro)](https://paperswithcode.com/sota/deblurring-on-gopro?p=flow-guided-sparse-transformer-for-video)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-flow-aligned-sequence-to/video-super-resolution-on-vimeo90k)](https://paperswithcode.com/sota/video-super-resolution-on-vimeo90k?p=unsupervised-flow-aligned-sequence-to)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-flow-aligned-sequence-to/deblurring-on-gopro)](https://paperswithcode.com/sota/deblurring-on-gopro?p=unsupervised-flow-aligned-sequence-to)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-flow-aligned-sequence-to/video-enhancement-on-mfqe-v2)](https://paperswithcode.com/sota/video-enhancement-on-mfqe-v2?p=unsupervised-flow-aligned-sequence-to)

# VR-Baseline

This is a baseline for video restoration. 

More codes and pretrained models will be updated later.

#### 1. Data Preparation

Download the datasets ([GOPRO](https://seungjunnah.github.io/Datasets/gopro),[DVD](https://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/#dataset),[REDS](https://seungjunnah.github.io/Datasets/reds.html),[VIMEO](http://toflow.csail.mit.edu/),[MFQE-v2](https://github.com/ryanxingql/mfqev2.0/wiki/MFQEv2-Dataset)) and place them into VR_Baseline/data.

#### 2. Installation

```shell
cd VR_Baseline
pip install torchvision==0.9.0  torch==1.8.0  torchaudio==0.8.0
pip install -r requirements.txt
pip install openmim
mim install mmcv-full
pip install -v -e .
```

#### 3. Training

```shell
cd VR_Baseline
bash tools/dist_train.sh configs/FGST_deblur_gopro.py 8
```

#### 4. Testing
Download [pretrained model](https://drive.google.com/drive/folders/1cmT0ti0-XwuCMcAhVEQWcD6rqCEwLo2T?usp=sharing) and run the following command.
```shell
python demo/restoration_video_demo.py ${CONFIG} ${CHKPT} ${IN_PATH} ${OUT_PATH}
```

#### 5.Acknowledgement.

We refer to codes from [BasicVSR++](https://github.com/ckkelvinchan/BasicVSR_PlusPlus) and [mmediting](https://github.com/open-mmlab/mmediting). Thanks for their awesome works.
