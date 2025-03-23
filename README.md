

This repository is the official implementation for the following paper:

[InstructOCR: Instruction Boosting Scene Text Spotting](https://arxiv.org/abs/2412.15523)

## Environment

```
pytorch==1.8.1 torchvision==0.9.1
```

## Dataset 

- CurvedSynText150k [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_ABCNet_Real-Time_Scene_Text_Spotting_With_Adaptive_Bezier-Curve_Network_CVPR_2020_paper.pdf): 
  - Part1 (94,723) Download (15.8G) ([Origin](https://universityofadelaide.box.com/s/xyqgqx058jlxiymiorw8fsfmxzf1n03p), [Google](https://drive.google.com/file/d/1OSJ-zId2h3t_-I7g_wUkrK-VqQy153Kj/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1Y5pqVqfjcc4FKxW4y8R5jw) password: 4k3x) 
  - Part2 (54,327) Download (9.7G) ([Origin](https://universityofadelaide.box.com/s/e0owoic8xacralf4j5slpgu50xfjoirs), [Google](https://drive.google.com/file/d/1EzkcOlIgEp5wmEubvHb7-J5EImHExYgY/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1gRv-IjqAUu6qnXN5BXlOzQ) password: a5f5)

- Totaltext [[paper]](https://ieeexplore.ieee.org/abstract/document/8270088/) [[source]](https://github.com/cs-chan/Total-Text-Dataset). 
  - Download (0.4G) ([Google](https://drive.google.com/file/d/1jfBYrAmh6Zshb7Jc0bctRjQKpK839SFq/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/18brRQAwnqGd4A_uwPRYRng) password: 5nhw) 
  
- MLT [[paper]](https://ieeexplore.ieee.org/abstract/document/8270168).
  - Download (6.8G) ([Origin](https://universityofadelaide.box.com/s/qu2wctdcsxh73bb94krdredpmx9nzf8m), [Google](https://drive.google.com/file/d/1nE2d_sIfcAejgVIv6-UjGNcBXgxc4QfD/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1rjqmb3uuki_Ppcxq-tl7oQ) password: zqrm)

- ICDAR2013 [[paper]](https://rrc.cvc.uab.es/?ch=2) [[source]](https://rrc.cvc.uab.es/?ch=2). 
  - Download (0.2G) ([Google](https://drive.google.com/file/d/1dMffINYhIRa9UD_3pzTFllVwL6PK7KXD/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1PiSZxZlG38qjj7Xb05cXdg) password: 5ddh) 
 
- ICDAR2015 [[paper]](https://rrc.cvc.uab.es/?ch=4) [[source]](https://rrc.cvc.uab.es/?ch=4). 
  - Download (0.1G) ([Google](https://drive.google.com/file/d/1THhzo_WH1RY5DlGdBfjRA_dwu9tAmQUE/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1x3EpYLRa4EtSMNg5JqszVg) password: wjrh) 
  TextOCR [[paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Singh_TextOCR_Towards_Large-Scale_End-to-End_Reasoning_for_Arbitrary-Shaped_Scene_Text_CVPR_2021_paper.html) [[source]](https://textvqa.org/textocr/dataset/).
  HierText [[paper]](https://arxiv.org/abs/2203.15143) [[source]](https://github.com/google-research-datasets/hiertext).

Please download and extract the above datasets into the `data` folder following the file structure below.

```
data
├─icdar2013
│  │  ic13_test.csv
│  │  ic13_train.csv
│  ├─test_images
│  └─train_images
├─icdar2015
│  │  ic15_test.csv
│  │  ic15_train.csv
│  ├─test_images
│  └─train_images
├─mlt2017
│  │  train.csv
│  └─MLT_train_images
├─syntext1
│  │  train.csv
│  └─syntext_word_eng
├─syntext2
│  │  train.csv
│  └─emcs_imgs
└─......
```

## Train and finetune

The model training in the original paper uses 32 GPUs (4 nodes, 8 A100 GPUs per node).

For training:

Single-GPU:
```
python3 -m torch.distributed.launch --master_port=3141 --nproc_per_node 1 --use_env main.py \
        --data_root '/path/data/' \
        --batch_size 2 \
        --train_stage pretrain \
        --lr 4.4e-4 \
        --max_size_train 1024 \
        --train_dataset ic15_train \
        --val_dataset ic15_val \
        --dec_layers 6 \
        --max_length 25 \
        --pad_rec \
        --pre_norm \
        --rotate_prob 0.3 \
        --train \
        --depths 6 \
        --padding_bins 0 \
        --epochs 200 \
        --warmup_epochs 5 \
        --output_dir '/path/save/' \
        --prefix 'debug'

```
Multi-GPU
Pretrain1:
```
python3 -m torch.distributed.launch --nproc_per_node 8 --nnodes=4 --node_rank=$node_rank --master_addr=$master_addr --master_port=3141 --use_env main.py \
        --data_root '/path/data/' \
        --batch_size 10 \
        --train_stage pretrain \
        --max_size_train 1024 \
        --lr 4.4e-4 \
        --train_dataset totaltext_train:ic13_train:ic15_train:mlt_train:syntext1_train:syntext2_train:textocr_train:hiertext_train \
        --val_dataset ic15_val \
        --dec_layers 6 \
        --max_length 25 \
        --pad_rec \
        --pre_norm \
        --rotate_prob 0.3 \
        --train \
        --depths 6 \
        --padding_bins 0 \
        --epochs 200 \
        --warmup_epochs 5 \
        --train_point point \
        --output_dir '/path/save/' \
        --prefix 'pretrain'
```
Pretrain2: scripts/pretrain2.sh
Instruction: scripts/instruction.sh


## Performance

The end-to-end recognition performances of InstructOCR with and without instructions on public benchmarks are:

| ICDAR 2015 | Strong | Weak | Generic | Model |
| ------- | ------ | ---- | ------- | ----- |
| Base | 87.1 | 83.4 | 80.6 | [Link](https://drive.google.com/file/d/1cFcDPqFXvTowVfoH4wD767lszzlbJkJ9/view?usp=sharing) |
| Instruction | 87.5 | 84.2 | 80.6 | [Link](https://drive.google.com/file/d/12sCDMS0XGrpEkCyBvP8zNUNjbTPWyUee/view?usp=sharing) |

| ICDAR 2013 | Strong | Weak | Generic | Model |
| ------- | ------ | ---- | ------- | ----- |
| Base | 94.4 | 93.3 | 91.2 | [Link]() |
| Instruction | 94.9 | 94.1 | 91.7 | [Link]() |

| Total-Text | None | Full | Model |
| ------- | ---- | ---- | ----- |
| Base | 82.6 | 87.5 | [Link]() |
| Instruction | 83.4 | 88.3 | [Link]() |



## Evaluation

Download the ground-truth files ([GoogleDrive](https://drive.google.com/file/d/1ztyjczfn3YdBf6hpLuV2Vs2UJPlRdAjm/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1ERkKR8L58ZVlB12SpCwEVQ) password: 35tr) and lexicons ([GoogleDrive](https://drive.google.com/file/d/1JxmuDsOZ-x_WO5lck2ZQZHRcjoUtUiLo/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1so_s94_XysLjlcWasos8mA) password: 9eml), and extracted them into the `evaluation` folder.

```
evaluation
│  eval.py
├─gt
│  ├─gt_ic13
│  ├─gt_ic15
│  └─gt_totaltext
└─lexicons
    ├─ic13
    ├─ic15
    └─totaltext
```

Then the command for evaluating the inference result of Total-Text is:
```
python evaluation/eval.py \
       --result_path ./output/totaltext/results/ep349/totaltext_val.json \
       # --with_lexicon \ # uncomment this line if you want to evaluate with lexicons.
       # --lexicon_type 0 # used for ICDAR2013 and ICDAR2015. 0: Generic; 1: Weak; 2: Strong.
```