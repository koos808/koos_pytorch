# Pytorch code collection

#### Default GPU Setup 확인하기

```
import torch

# 현재 Setup 되어있는 device 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))
```
* 참고 사이트 : `https://jeongwookie.github.io/2020/03/24/200324-pytorch-cuda-gpu-allocate/`

#### 원하는 GPU 할당(선택)하기
```
GPU_NUM = 0 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check
```
