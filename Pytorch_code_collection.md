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

#### Pytorch dataloader shape 확인하기
```
dataiter = iter(dataloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)
```

#### Image Augmentation 여러 방법
* 방법 1
    ```
    transform_train = transforms.Compose([transforms.Resize((32, 32)),
                                      transforms.RandomHorizontalFlip(), #0.5확률로 이미지를 뒤집음
                                      transforms.RandomRotation(10), # 10도 이하로 랜덤하게 기울인다.
                                      transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)), # 아핀 변환
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 밝기, 대비, 채도를 랜덤하게 조절한다.
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])

    transformer = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])


    training_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    validation_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transformer)

    training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=100, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=100, shuffle=False)

    ```
* 방법 2
    ```
    import Augmentor # pip install Augmentor

    ## 증강 시킬 이미지 폴더 경로
    img = Augmentor.Pipeline('./aug_image/class1')
    ## 좌우 반전
    img.flip_left_right(probability=0.5) 
    ## 상하 반전
    img.flip_top_bottom(probability=0.5)
    ## 왜곡
    #img.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=8)
    img.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    img.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
    ## 증강 이미지 수
    img.sample(4000)
    #img.sample(5000)
    img.process()
    ```