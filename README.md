

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

<!-- Please download and extract the above datasets into the `data` folder following the file structure below.

```
data
├─CTW1500
│  ├─annotations
│  │      test_ctw1500_maxlen25.json
│  │      train_ctw1500_maxlen25_v2.json
│  ├─ctwtest_text_image
│  └─ctwtrain_text_image
├─icdar2013
│  │  ic13_test.json
│  │  ic13_train.json
│  ├─test_images
│  └─train_images
├─icdar2015
│  │  ic15_test.json
│  │  ic15_train.json
│  ├─test_images
│  └─train_images
|- inversetext
|  |- test_images
|  └─ test_poly.json
├─mlt2017
│  │  train.json
│  └─MLT_train_images
├─syntext1
│  │  train.json
│  └─syntext_word_eng
├─syntext2
│  │  train.json
│  └─emcs_imgs
└─totaltext
    │  test.json
    │  train.json
    ├─test_images
    └─train_images
``` -->

## Train and finetune

The model training in the original paper uses 32 GPUs (4 nodes, 8 A100 GPUs per node).


## Performance