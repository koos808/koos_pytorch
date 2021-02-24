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



<br>

※ 파이토치 개요
----
<br>
## 1. Intro

* 텐소플로를 비롯한 대부분의 딥러닝 프레임워크는 정적 계산 그래프(`Static Computational Graph`) 방식이기 때문에 실행 중에 `그래프`를 바꿀 수 없다.
* 정적 계산 그래프(`Static Computational Graph`) 방식은 그래프 계산 방식을 최초에 정해두기 때문에 최적화하기 쉽지만, 유연하지 않다는 단점이 있다.
* 반면, 파이토치는 동적 계산 그래프(`Dynamic Computational Graph`) 방식이므로 데이터에 유연한 모델을 만들 수 있다.
* 
