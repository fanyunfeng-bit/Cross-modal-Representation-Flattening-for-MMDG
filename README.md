
## Code
The code was tested using `Python 3.10.4`, `torch 1.11.0+cu113` and `NVIDIA GeForce RTX 3090`.

Environments:
```
mmcv-full 1.2.7
mmaction2 0.13.0
```
### EPIC-Kitchens Dataset
### Prepare

#### Download Pretrained Weights
1. Download Audio model [link](http://www.robots.ox.ac.uk/~vgg/data/vggsound/models/H.pth.tar), rename it as `vggsound_avgpool.pth.tar` and place under the `EPIC-rgb-flow-audio/pretrained_models` directory
   
2. Download SlowFast model for RGB modality [link](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth) and place under the `EPIC-rgb-flow-audio/pretrained_models` directory
   
3. Download SlowOnly model for Flow modality [link](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth) and place under the `EPIC-rgb-flow-audio/pretrained_models` directory

#### Download EPIC-Kitchens Dataset
```
bash download_script.sh 
```
Download Audio files [EPIC-KITCHENS-audio.zip](https://polybox.ethz.ch/index.php/s/PE2zIL99OWXQfMu).

Unzip all files and the directory structure should be modified to match:

```
├── MM-SADA_Domain_Adaptation_Splits
├── rgb
|   ├── train
|   |   ├── D1
|   |   |   ├── P08_01.wav
|   |   |   ├── P08_01
|   |   |   |     ├── frame_0000000000.jpg
|   |   |   |     ├── ...
|   |   |   ├── P08_02.wav
|   |   |   ├── P08_02
|   |   |   ├── ...
|   |   ├── D2
|   |   ├── D3
|   ├── test
|   |   ├── D1
|   |   ├── D2
|   |   ├── D3


├── flow
|   ├── train
|   |   ├── D1
|   |   |   ├── P08_01 
|   |   |   |   ├── u
|   |   |   |   |   ├── frame_0000000000.jpg
|   |   |   |   |   ├── ...
|   |   |   |   ├── v
|   |   |   ├── P08_02
|   |   |   ├── ...
|   |   ├── D2
|   |   ├── D3
|   ├── test
|   |   ├── D1
|   |   ├── D2
|   |   ├── D3
```

### Video and Audio
```
cd EPIC-rgb-flow-audio
```
```
python train_video_flow_audio_EPIC_ours2.py --use_video --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/EPIC-KITCHENS/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 400 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```
```
python train_video_flow_audio_EPIC_ours2.py --use_video --use_audio -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/EPIC-KITCHENS/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 400 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```
```
python train_video_flow_audio_EPIC_ours2.py --use_video --use_audio -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/EPIC-KITCHENS/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 400 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```

### Video and Flow
```
cd EPIC-rgb-flow-audio
```
```
python train_video_flow_audio_EPIC_ours2.py --use_video --use_flow -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/EPIC-KITCHENS/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 400 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```
```
python train_video_flow_audio_EPIC_ours2.py --use_video --use_flow -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/EPIC-KITCHENS/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 400 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```
```
python train_video_flow_audio_EPIC_ours2.py --use_video --use_flow -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/EPIC-KITCHENS/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 400 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```

### Flow and Audio
```
cd EPIC-rgb-flow-audio
```
```
python train_video_flow_audio_EPIC_ours2.py --use_audio --use_flow -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/EPIC-KITCHENS/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 400 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```
```
python train_video_flow_audio_EPIC_ours2.py --use_audio --use_flow -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/EPIC-KITCHENS/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 400 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```
```
python train_video_flow_audio_EPIC_ours2.py --use_audio --use_flow -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/EPIC-KITCHENS/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 400 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```

### Video and Flow and Audio
```
cd EPIC-rgb-flow-audio
```
```
python train_video_flow_audio_EPIC_ours2.py --use_video --use_audio --use_flow -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/EPIC-KITCHENS/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 400 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```
```
python train_video_flow_audio_EPIC_ours2.py --use_video --use_audio --use_flow -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/EPIC-KITCHENS/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 400 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```
```
python train_video_flow_audio_EPIC_ours2.py --use_video --use_audio --use_flow -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/EPIC-KITCHENS/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 400 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```


### HAC Dataset
This dataset can be downloaded at [link](https://polybox.ethz.ch/index.php/s/3F8ZWanMMVjKwJK).

Download the pretrained weights similar to EPIC-Kitchens Dataset and put under the `HAC-rgb-flow-audio/pretrained_models` directory.

### Video and Audio
```
cd HAC-rgb-flow-audio
```
```
python train_video_flow_audio_HAC_ours2.py --use_video --use_audio --use_flow -s animal cartoon -t human --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/HAC/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 100 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```
```
python train_video_flow_audio_HAC_ours2.py --use_video --use_audio --use_flow -s human cartoon -t animal --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/HAC/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 100 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```
```
python train_video_flow_audio_HAC_ours2.py --use_video --use_audio --use_flow -s human animal -t cartoon --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/HAC/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 100 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```

### Video and Flow
```
cd HAC-rgb-flow-audio
```
```
python train_video_flow_audio_HAC_ours2.py --use_video --use_flow --use_flow -s animal cartoon -t human --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/HAC/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 100 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```
```
python train_video_flow_audio_HAC_ours2.py --use_video --use_flow --use_flow -s human cartoon -t animal --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/HAC/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 100 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```
```
python train_video_flow_audio_HAC_ours2.py --use_video --use_flow --use_flow -s human animal -t cartoon --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/HAC/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 100 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```

### Flow and Audio
```
cd HAC-rgb-flow-audio
```
```
python train_video_flow_audio_HAC_ours2.py --use_audio --use_flow --use_flow -s animal cartoon -t human --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/HAC/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 100 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```
```
python train_video_flow_audio_HAC_ours2.py --use_audio --use_flow --use_flow -s human cartoon -t animal --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/HAC/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 100 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```
```
python train_video_flow_audio_HAC_ours2.py --use_audio --use_flow --use_flow -s human animal -t cartoon --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/HAC/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 100 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```

### Video and Flow and Audio
```
cd HAC-rgb-flow-audio
```
```
python train_video_flow_audio_HAC_ours2.py --use_video --use_audio --use_flow --use_flow -s animal cartoon -t human --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/HAC/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 100 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```
```
python train_video_flow_audio_HAC_ours2.py --use_video --use_audio --use_flow --use_flow -s human cartoon -t animal --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/HAC/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 100 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```
```
python train_video_flow_audio_HAC_ours2.py --use_video --use_audio --use_flow --use_flow -s human animal -t cartoon --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/HAC/ --vanilla_learning --DG_algorithm naive --SMA --sma_start_step 100 --CM_mixup --mix_alpha 0.1 --contrast --distill --distill_coef 3.0 --mix_coef 2.0
```



## Acknowledgement

Many thanks to the excellent open-source projects [SimMMDG](https://github.com/donghao51/SimMMDG).
