※ 파이토치 설치 Install Pytorch 

* RTX 3080 컴퓨터 설정 업데이트 날짜
  * DATE : `2021-02-24`
* 0) CUDA, Cudnn 설치
  * 내 데스크탑 사양 및 버전
  * OS : Windows10
  * VGA : `RTX 3080`
  * CUDA : `CUDA 11.0 -> cuda_11.0.3_451.82_win10`
  * cudnn : `Cudnn 8.0.4` -> `cudnn-11.0-windows-x64-v8.0.4.30` 
* 1) Anaconda 및 tensorflow 설치
  * Anaconda : `Anaconda3-2020.11-Windows-x86_64`
  * Tensorflow : `pip install tf-nightly==2.5.0.dev20210110`
  * Python : `3.8`
* 2) 가상환경 생성
  * `conda create -n koos_torch python=3.8`
* 3) pytorch 설치
  * 설치 사이트 : `https://pytorch.org/get-started/locally/`
  * Stable(1.7.1) - Windows - Conda - Python - Cuda 11.0
  * pytorch Stable(1.7.1) : `conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch`
  * ※ 참고 사항 : pytorch 설치시 시간 오래 걸림
* 4) torchtext 설치
  * `pip install torchtext` 
* 5) 각종 모듈 설치
  * `pip install numpy matplotlib scikit-learn`
* 6) 주피터 노트북 설치
  * `pip install jupyter`

* 7) GPU 사용 유무 확인 및 pytorch 버전 확인
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    ```
    * pytorch 버전 확인 : `torch.__version__`

<br>

※ 파이토치 개요
----
<br>

## 1. Intro

* 텐소플로를 비롯한 대부분의 딥러닝 프레임워크는 정적 계산 그래프(`Static Computational Graph`) 방식이기 때문에 실행 중에 `그래프`를 바꿀 수 없다.
* 정적 계산 그래프(`Static Computational Graph`) 방식은 그래프 계산 방식을 최초에 정해두기 때문에 최적화하기 쉽지만, 유연하지 않다는 단점이 있다.
* 반면, 파이토치는 동적 계산 그래프(`Dynamic Computational Graph`) 방식이므로 데이터에 유연한 모델을 만들 수 있다.

## 2. 기본 함수

* tensor 랭크 및 shape 확인 하기
    ```
    x = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    print("Size:", x.size())
    print("Shape:", x.shape)
    print("랭크(차원):", x.ndimension())
    ```
* tensor 랭크, shape 변형
  * `unsqueeze()` : `x = torch.unsqueeze(x,0)` => 2차원 -> 3차원
  * `squeeze()` : 3차원 -> 2차원
  * `view()` : `x.view(9)` => 직접 텐서의 모양 바꿈. [9] shape의 rank 1 tensor
  * 위 함수들은 원소 수 그대로 유지하면서 모양과 차원 조절함.

* `randn()` : 정규분포 난수 생성
  * `w = torch.randn(5,3, dtype=torch.float)`

* `torch.mm()` : 행렬 곱 : A(5,3)*B(3,2) + b(5,2)
  * `wx = torch.mm(w,x) + b` 

* `Autograd` : 미분 계산을 자동화하여 경사하강법을 구현.
  * 값 1.0인 스칼라 텐서 w를 정의하고, 수식을 w에 대해 미분하여 기울기를 계산
    * `w = torch.tensor(1.0, requires_grad=True)`
    * w에 대한 미분값을 w.grad에 저장함.
  * 미분 : `l.backward()`
    * l = (w*3)**2
    * l.backward() => l을 w로 미분한 값 반환

* 데이터 형식(Type) 변환
  * `torch.FloatTensor()` : `x_train = torch.FloatTensor(x_train)` => 파이토치 텐서로 변환
  * 

## 3. 응용

