# 未待完续


# 1. 首先进入你的项目目录
cd /d E:\LightningDiT-main

# 2. 创建并激活 conda 环境
conda create -n lightningdit python=3.10.12
conda activate lightningdit

3.检查自身电脑配置后修改reqiurement
nvidia-smi

# 4. 安装依赖包，
pip install -r requirements.txt


5.下载模型文件（建议放在项目下的 pretrained/ 文件夹
- Download weights and data infos:

    - Download pre-trained models
        | Tokenizer | Generation Model | FID | FID cfg |
        |:---------:|:----------------|:----:|:---:|
        | [VA-VAE](https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/blob/main/vavae-imagenet256-f16d32-dinov2.pt) | [LightningDiT-XL-800ep](https://huggingface.co/hustvl/lightningdit-xl-imagenet256-800ep/blob/main/lightningdit-xl-imagenet256-800ep.pt) | 2.17 | 1.35 |
        |           | [LightningDiT-XL-64ep](https://huggingface.co/hustvl/lightningdit-xl-imagenet256-64ep/blob/main/lightningdit-xl-imagenet256-64ep.pt) | 5.14 | 2.11 |



6.下载latents_stats.pt
  - Download [latent statistics](https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/blob/main/latents_stats.pt). This file contains the channel-wise mean and standard deviation statistics.



7.更改configs/reproductions/
更改data_path和ckpt_path

8.下载FID
创建fid_stats并下载
https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz

更改 configs/reproductions/
更改fid_reference_file
