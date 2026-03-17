import requests
import os

# 创建目录
os.makedirs('fid_stats', exist_ok=True)

# 下载文件
url = 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz'
print('正在下载FID参考文件...')
response = requests.get(url)

with open('fid_stats/VIRTUAL_imagenet256_labeled.npz', 'wb') as f:
    f.write(response.content)

print('下载完成！文件保存在 fid_stats/VIRTUAL_imagenet256_labeled.npz')