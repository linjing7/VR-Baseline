# A Toolbox for Video Restoration

[![jiedu](https://img.shields.io/badge/中文解读-S2SVR-179bd3)](https://mp.weixin.qq.com/s/0Hqp2A8pjo1_Gn23LEpPXg)
[![jiedu](https://img.shields.io/badge/中文解读-FGST-179bd3)](https://zhuanlan.zhihu.com/p/563455469)
![visitors](https://visitor-badge.glitch.me/badge?page_id=linjing7/VR-Baseline)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/flow-guided-sparse-transformer-for-video/deblurring-on-dvd-1)](https://paperswithcode.com/sota/deblurring-on-dvd-1?p=flow-guided-sparse-transformer-for-video)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/flow-guided-sparse-transformer-for-video/deblurring-on-dvd)](https://paperswithcode.com/sota/deblurring-on-dvd?p=flow-guided-sparse-transformer-for-video)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/flow-guided-sparse-transformer-for-video/deblurring-on-gopro)](https://paperswithcode.com/sota/deblurring-on-gopro?p=flow-guided-sparse-transformer-for-video)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-flow-aligned-sequence-to/video-super-resolution-on-vimeo90k)](https://paperswithcode.com/sota/video-super-resolution-on-vimeo90k?p=unsupervised-flow-aligned-sequence-to)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-flow-aligned-sequence-to/deblurring-on-gopro)](https://paperswithcode.com/sota/deblurring-on-gopro?p=unsupervised-flow-aligned-sequence-to)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-flow-aligned-sequence-to/video-enhancement-on-mfqe-v2)](https://paperswithcode.com/sota/video-enhancement-on-mfqe-v2?p=unsupervised-flow-aligned-sequence-to)


#### Authors

 Jing Lin*, Yuanhao Cai*, Xiaowan Hu, Haoqian Wang, Youliang Yan, Xueyi Zou, Henghui Ding, Yulun Zhang, Radu Timofte, and Luc Van Gool

![ntire](/figure/ntire.png)

#### News

- **2022.12.08 :** Pretrained model, training/testing log, visual results of FGST on GoPro and DVD dataset are released.  S2SVR will be provided later.🔥
- **2022.11.30 :** Data preparation codes of GoPro and DVD are provided. :high_brightness:

- **2022.08.05 :** Pretrained model of FGST on GOPRO dataset is released. :dizzy:
- **2022.05.14 :** Our FGST and S2SVR are accepted by ICML2022. :rocket: 

|                  *Super-Resolution*                  |                         *Deblur*                         |                *Compressed Video Enhancement*                |
| :--------------------------------------------------: | :------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="./figure/lr2sr.gif"  height=135 width=240> | <img src="./figure/blur2sharp.gif" height=135 width=240> | <img src="./figure/compressed2enhanced.gif" height=135 width=240> |

#### Papers

- [Flow-Guided Sparse Transformer for Video Deblurring (ICML 2022)](https://arxiv.org/abs/2201.01893)
- [Unsupervised Flow-Aligned Sequence-to-Sequence Learning for Video Restoration (ICML 2022)](https://arxiv.org/abs/2205.10195)

|                  Method                  |                           Dataset                            |                       Pretrained Model                       |                         Training Log                         |                         Testing Log                          |                        Visual Result                         | Quantitative  Result |
| :--------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :------------------: |
| [FGST](https://arxiv.org/abs/2201.01893) |    [GoPro](https://seungjunnah.github.io/Datasets/gopro)     | [Google Drive](https://drive.google.com/file/d/1hG-sYmCAWYxRTpUFz3enxvJrP9V0PCgk/view?usp=share_link) / [Baidu Disk](https://pan.baidu.com/s/1WkbfAgGw6G2W2VY8549P8w?pwd=VR11) | [Google Drive](https://drive.google.com/file/d/1MZjrML8adrrDbwmV_MgO3pSXMWQKcwXj/view?usp=share_link) / [Baidu Disk](https://pan.baidu.com/s/19-dovgSzODQPNogokx7EIQ?pwd=VR11) | [Google Drive](https://drive.google.com/file/d/1q0Obom4r21x7hMBx0BTJ3BiDtLBgjcYM/view?usp=share_link) /  [Baidu Disk](https://pan.baidu.com/s/1L61HUuw5KISZyN59FVMxXg?pwd=VR11) | [Google Drive](https://drive.google.com/drive/folders/1RTQmisGGpNV8OTh_YAwT2Z3XWeGpcVdK?usp=share_link) /  [Baidu Disk](https://pan.baidu.com/s/1BDeNloos9T14ay6Vi1_FLw?pwd=VR11) |    33.02 / 0.947     |
| [FGST](https://arxiv.org/abs/2201.01893) | [DVD](https://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/#dataset) | [Google Drive](https://drive.google.com/file/d/1L8kk3x7d3Ef0vN4ExU_VdXsz5POZjDgr/view?usp=share_link) / [Baidu Disk](https://pan.baidu.com/s/1l8AGhqNh07CQFpF10XoyeQ?pwd=VR11) | [Google Drive](https://drive.google.com/file/d/1IggT0JCmq6J4wNTMSZGflzou2nU98jOb/view?usp=share_link) / [Baidu Disk](https://pan.baidu.com/s/1rj4NdB9l2v6QihtwK18Ghw?pwd=VR11) | [Google Drive](https://drive.google.com/file/d/1jhEjuB9Mtec6wrfDXWGeyFsxmM0j8DTL/view?usp=share_link) /  [Baidu Disk](https://pan.baidu.com/s/1zCCQ2WFcBwGIMgfWZCSxvA?pwd=VR11) | [Google Drive](https://drive.google.com/drive/folders/1hd-Fka1Ei27WSEwL5qn6romntkpxjZps?usp=share_link) /  [Baidu Disk](https://pan.baidu.com/s/1muukHrqKOFlyGsSqmmm1TQ?pwd=VR11) |    33.50 / 0.945     |

Note: access code for `Baidu Disk` is `VR11`

## 1. Create Environment:

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
  ![](../../../../../../Applications/Typora.app/Contents/Resources/TypeMark/page-dist/static/media/icon.06a6aa23.png)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

- Python packages:

```shell
pip install torchvision==0.9.0  torch==1.8.0  torchaudio==0.8.0
pip install -r requirements.txt
pip install openmim
mim install mmcv-full==1.5.0
pip install -v -e .
pip install cupy-cuda101==7.7.0
```

## 2. Prepare Dataset:

Download the datasets ([GOPRO](https://seungjunnah.github.io/Datasets/gopro),[DVD](https://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/#dataset),[REDS](https://seungjunnah.github.io/Datasets/reds.html),[VIMEO](http://toflow.csail.mit.edu/),[MFQE-v2](https://github.com/ryanxingql/mfqev2.0/wiki/MFQEv2-Dataset)) and and recollect them as the following form:

```shell
|--VR-Baseline
    |--data
    	|-- GoPro
    	    |-- test
    	    |-- train
    	|-- DVD
    	    |-- quantitative_datasets
    	      |-- GT
    	      |-- LQ
    	    |-- qualitative_datasets
    	|-- REDS
    	    |-- train_sharp_bicubic
    	    |-- train_sharp
    	|-- VIMEO
    	    |-- BIx4
    	    |-- GT
    	|-- MFQEV2
    	    |-- test
    	    |-- train
```

You can run the following command to recollect GoPro and DVD dataset:

```shell
cd VR-Baseline/data_preparation

# recollect GoPro dataset
python GoPro_Util.py --input_path INPUT_PATH --save_path SAVE_PATH

# recollect DVD dataset
python DVD_Util.py --input_path INPUT_PATH --save_path SAVE_PATH
```

You need to replace `INPUT_PATH` and `SAVE_PATH` with your own path.

## 3. Training:

```shell
cd VR_Baseline

# training FGST on GoPro dataset
bash tools/dist_train.sh configs/FGST_deblur_gopro.py 8

# training FGST on DVD dataset
bash tools/dist_train.sh configs/DVD_deblur_gopro.py 8

# training S2SVR on GoPro dataset
bash tools/dist_train.sh configs/S2SVR_deblur_gopro.py 8

# training S2SVR on REDS dataset
bash tools/dist_train.sh configs/S2SVR_sr_reds4.py 8

# training S2SVR on VIMEO dataset
bash tools/dist_train.sh configs/S2SVR_sr_vimeo.py 8

# training S2SVR on MFQEv2 dataset
bash tools/dist_train.sh configs/S2SVR_vqe_mfqev2.py 8
```

The training log, trained model will be available in `VR-Baseline/experiments/` . 


## 4. Testing:

Download [pretrained model](https://drive.google.com/drive/folders/1cmT0ti0-XwuCMcAhVEQWcD6rqCEwLo2T?usp=sharing) and run the following command.

To test on benchmark:

```shell
cd VR_Baseline

# testing FGST on GoPro dataset
bash tools/dist_train.sh configs/FGST_deblur_gopro_test.py 8

# testing FGST on DVD dataset
bash tools/dist_train.sh configs/FGST_deblur_dvd_test.py 8
```

## 5.  TODO 

These works are mostly done during the internship at [HUAWEI Noah's Ark Lab](http://dev3.noahlab.com.hk/). Due to the limitation of company regulations, the original pre-trained models can not be transferred and published here. We will retrain more models and open-source them when we have enough GPUs as soon as possible. 

- [ ] More data preparation codes
- [ ] More Pretrained Models
- [ ] Inference Results
- [ ] MFQEv2 dataloader

## 6.  Acknowledgement.

We refer to codes from [BasicVSR++](https://github.com/ckkelvinchan/BasicVSR_PlusPlus) and [mmediting](https://github.com/open-mmlab/mmediting). Thanks for their awesome works.

## 7. Citation

If this repo helps you, please consider citing our works:

```shell
# FGST
@inproceedings{fgst,
  title={Flow-Guided Sparse Transformer for Video Deblurring},
  author={Lin, Jing and Cai, Yuanhao and Hu, Xiaowan and Wang, Haoqian and Yan, Youliang and Zou, Xueyi and Ding, Henghui and Zhang, Yulun and Timofte, Radu and Van Gool, Luc},
  booktitle={ICML},
  year={2022}
}


# S2SVR
@inproceedings{seq2seq,
  title={Unsupervised Flow-Aligned Sequence-to-Sequence Learning for Video Restoration},
  author={Lin, Jing  and Hu, Xiaowan and Cai, Yuanhao and Wang, Haoqian and Yan, Youliang and Zou, Xueyi and Zhang, Yulun and Van Gool, Luc},
  booktitle={ICML},
  year={2022}
}
```
