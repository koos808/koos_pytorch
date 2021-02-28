※ 참고 한 책&사이트 
---

* 1)펭귄브로의 3분 딜버닝, 파이토치맛


※ 파이토치 설치 Install Pytorch 
---

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
    ```
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
  * EX) Fashion MNIST shape : x.size() -> [64, 1, 28, 28](배치, RGB, 높이, 넓이)

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

* sub_() 함수 : 빼기
  * `y.data.sub_(1)` : y의 모든 값에서 1씩 뺌

* sign() 함수
  * `gradient.sign()`
  * 입력이 0보다 작으면 -1을, 0이면 0을, 0보다 크면 1을 출력하는 단순한 함수

* concat 함수
  * `torch.cat([z,c], 1)` : 두 벡터를 이어붙히는 연산을 실행(100+10 = 110)

## 3. 응용

* 신경망 모듈(`torch.nn.Module`)
  * `class NeuralNet(torch.nn.Module):`

  * `__init__()` 함수
    * 파이썬에서 객체가 갖는 속성값을 초기화하는 역할로, 객체가 생성될 때 자동으로 호출된다.

  * `super()` 함수
    * super() 함수를 부르면 만든 NeuralNet 클래스는 파이토치의 nn.Module 클래스의 속성들을 가지고 초기화된다.

    ```
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
    ```

* 오차함수 및 최적화 알고리즘
  * 오차함수
    * ex) `criterion = torch.nn.BCELoss()`
  * 최적화 알고리즘
    * ex) `optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate`
    * optimizer는 step() 함수를 부를 때마다 가중치를 학습률만큼 갱신한다.
    * model.parameter() 함수로 추출한 모델 내부의 가중치와 학습률을 입력한다.
  * weight_decay 사용
    * `optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)`
    * `optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)`

* item()
  * ex) `test_loss_before.item())` : item() 함수는 텐서 속의 숫자를 스칼라 값으로 반환한다.

* model.train() : 학습 모드 전환
  * 모델에 train() 함수를 호출해 학습 모드로 바꿔준다. epoch마다 새로운 gradient 값을 계산하므로 zero_grad() 함수를 호출해 경사값을 0으로 설정한다.

* 역전파 함수
  * `train_loss.backward()` : 오차 함수를 가중치로 미분하여 오차가 최소가 되는 방향을 구함
  * `optimizer.step()` : 위 에서 구한 방향으로 모델을 학습률만큼 이동시킨다.

* 모델 저장 및 불러오기(Model save & load)
  * `torch.save(model.state_dict(), './model.pt')`
  * `new_model.load_state_dict(torch.load('./model.pt'))`

* 파이토치와 torchvision 이미지 데이터 관련 대표적인 모듈
  * `torch.utils.data`  
    * 데이터셋의 표준을 정의하고 데이터셋을 불러오고 처리하는 모듈
    * 파이토치 모델을 학습시키기 위한 데이터셋의 표준을 `torch.utils.data.Datase`t에 정의함.
    * Dataset 모듈을 상속하는 파생 클래스는 학습에 필요한 데이터를 로딩해주는 `torch.utils.data.DataLoader` 인스턴스의 입력으로 사용할 수 있다.
  * `torchvision.datasets`
    * 이미지 데이터셋의 모음 ex) Fasion MNIST
  * `torchvision.transfoms`
    * 이미지 데이터셋에 쓸 수 있는 여러 가지 변환 필터 모듈
    * ex) resize, crop, brightness, 대비(contrast)
  * `torchvision.utils`
    * 이미지 데이터 저장 및 시각화 모듈

* 이미지를 텐서로 바꿔주는 코드
    ```
    transform = transforms.Compose([
    transforms.ToTensor()
    ])
    ```
    * `ToTensor()` 외에도 `Resize`, `Normalize`, `RandomCrop` 등 다양한 Transforms 기능이 많다.

* `DataLoader`
    ```
    train_loader = data.DataLoader(
        dataset     = trainset,
        batch_size  = batch_size
    )
    ```
    * `DataLoader`는 데이터셋을 Batch라는 작은 단위로 쪼개고 학습 시 반복문 안에서 데이터를 공급해주는 클래스이다.
    * 데이터 하나씩 살펴보기
      * `dataiter = iter(train_loader)`
      * `images, labels = next(dataiter)`

* GPU 사용하기
  * `USE_CUDA = torch.cuda.is_available()`
  * `DEVICE = torch.device("cuda" if USE_CUDA else "cpu")`
  * `model = Net().to(DEVICE)`
  * `to()` 함수 : 모델의 파라미터들을 지정한 장치의 메모리로 보내는 역할을 한다.
  * CPU 사용하면 필요 X, GPU 사용하려면 to("cuda")로 지정함.

* 학습하기
  * `def train(model, train_loader, optimizer):`

* `eq()` : 값이 일치하면 1, 아니면 0을 출력
  * ex) `pred.eq(target.view_as(pred))`
  * == `argmax` : 배열에서 가장 큰 값이 있는 인덱스를 출력하는 함수
  
* `view_as()` : target 텐서를 view_as() 함수 안에 들어가는 인수(ex. pred)의 모양대로 다시 정렬한다.

## 4. CNN 관련

* Conv2d
  * `self.conv1 = nn.Conv2d(1, 10, kernel_size=5)`
    * => 커널(필터) 크기 5x5를 사용해서 10개의 feature map을 생성
  * `self.conv2 = nn.Conv2d(10, 20, kernel_size=5)`
    * => 커널(필터) 크기 5x5를 사용해서 20개의 feature map을 생성
 
* keras의 Flatten 역할
  * `x.view(-1, 320)`
  * 2차원의 feature map을 바로 입력으로 넣을 수 없으므로 view() 함수를 이용하여 1차원으로 펴주는 역할을 함.(차원 축소)
  * view() 함수에 들어가는 첫 번째 입력 -1은 '남는 차원 모두'를 뜻하고, 320은 x가 가진 원소 개수를 뜻함.

* learning rate decay(학습률 감소) method
  * `scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)`
  * 50번 호출될 떄 학습률에 0.1(gamma)을 곱함. 0.1로 시작한 학습률은 50 epoch 이후 0.01로 낮아짐.
  * In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.
    * `scheduler.step()은 train 이후에 호출되어야 함. 그렇지 않으면 PyTorch가 학습률 일정의 첫 
    번째 값을 건너 뛰게 됨.

* 이미지의 기울기값을 구하도록 설정
  * img_tensor.requires_grad_(True)

## 5. RNN 관련

* 3분 딥러닝 파이토치맛 - 07.순차적인 데이터를 처리하는 RNN
  * `https://github.com/keon/3-min-pytorch/tree/master/

* `torchtext` 사용
  * `from torchtext import data, datasets`

* IMDB 데이터 RNN 예시
  * 1) import library
  * 2) IMDB 데이터셋 로딩 -> Tensor 변환
    * `TEXT`, `LABEL` 이라는 객체를 생성해 텐서로 변환하는 설정을 지정해줌.
      ```
      TEXT = data.Field(sequential=True, batch_first=True, lower=True)
      LABEL = data.Field(sequential=False, batch_first=True)
      trainset, testset = datasets.IMDB.splits(TEXT, LABEL)
      ```
    * TEXT : Sequential 파라미터를 이용해 데이터셋이 순차적인 데이터셋인지 명시해준다.(sequential=True)
    * LABEL : 레이블 값은 단순히 클래스를 나타내는 숫자이므로 순차적인 데이터가 아니다.(sequential=False)
    * batch_first 변수 : 파라미터로 신경망에 입력되는 텐서의 첫 번째 차원값이 batch_size가 되도록 정해준다.
    * lower 변수 : 텍스트 데이터 속 모든 영문 알파벳을 소문자로 처리
    * splits() 함수 : 모델에 입력되는 데이터셋을 만들어 줌
  * 3) 워드 임베딩(`word embedding`)에 필요한 단어 사전(`word vocabulary`) 생성
    * `TEXT.build_vocab(trainset, min_freq=5)` 
      * => min_freq는 학습 데이터에서 최소 5번 이상 등장한 단어만을 사전에 담겠다는 뜻.
      * => 학습 데이터에서 5번 미만으로 출현하는 단어는 Unknown을 뜻하는 `unk`라는 토큰으로 대체됨.
    * `LABEL.build_vocab(trainset)`
  * 4) train, validation, test 분할
    * `trainset, valset = trainset.split(split_ratio=0.8)`
  * 5) batch를 생성해주는 iterator 생성
        ```
        train_iter, val_iter, test_iter = data.BucketIterator.splits(
                (trainset, valset, testset), batch_size=BATCH_SIZE,
                shuffle=True, repeat=False)
        ```
  * 6) 사전 속 단어들의 개수와 레이블의 수 지정
    * `vocab_size = len(TEXT.vocab)`
    * `n_classes = 2`
  * 7) RNN 모델 생성 : `BasicGRU`
    * `self.n_layers = n_layers` : `__init__()` 함수에서 hidden vector의 layer인 n_layers를 정의
    * `self.embed = nn.Embedding(n_vocab, embed_dim)`
      * n_vocab : 사전 안 vocab 전체 단어 개수
      * embed_dim : 임베딩된 단어 텐서가 지니는 차원값. 즉, 임베딩된 토큰의 차원 값
    * RNN을 통해 생성되는 hidden vector의 차원값과 dropout을 정의
      * `self.hidden_dim = hidden_dim` # 모델 내 hidden vector의 차원 값
      * `self.dropout = nn.Dropout(dropout_p)`
    * RNN 모델 정의
      * `self.gru = nn.GRU(embed_dim, self.hidden_dim, num_layers = self.n_layers, batch_first=True)`
    * `Forward()` 함수 정의
      * RNN 계열의 신경망은 입력 데이터 외에도 첫 번째 hidden vector h_0을 정의해 x와 함께 입력해줘야 한다.
      * 직접 _init_state()라는 함수를 구현하고 호출해서 첫번 째 hidden vector를 정의
      * `x, _ = self.gru(x, h_0)` : self.gru(x, h_0)의 결과값은 (batch_size, 입력 x의 길이, hidden_dim)의 shape을 지닌 3d 텐서임
      * `h_t = x[:,-1,:]` : 이것을 통해 hidden vector들의 마지막 토큰들을 내포한 (batch_size, 1, hidden_dim) 모양의 텐서를 추출할 수 있음

    * `_init_state()` 함수
      ```
      def _init_state(self, batch_size=1):
          weight = next(self.parameters()).data # nn.GRU 모듈의 첫 번째 가중치 텐서를 추출
          return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_() 
      ```
      * parameters() 함수 : nn.Moudle의 가중치 정보들을 iterator 형태로 반환
      * 이 iterator가 생성하는 원소들은 각각 실제 신경망의 가중치 텐서(.data)를 지닌 객체
      * new() 함수 : 모델의 가중치와 같은 모양인 (n_layers, batch_size, hidden_dim) 모양을 갖춘 텐서로 변환
      * zero_() 함수 : 텐서 내 모든 값을 0으로 초기화
      * `h_0 = self._init_state(batch_size=x.size(0))` : 첫 번째 hidden vector h_0은 보통 모든 특성값이 0인 벡터로 설정
  * model save & load
    * save : `torch.save(model.state_dict(), './snapshot/txtclassification.pt')`
    * load : `model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))`

  * ! RNN이 아닌 GRU를 사용한 이유
    * 데이터 뒷부분에 다다를수록 앞부분의 정보 손실이 발생하는 RNN의 단점때문에 GRU를 사용한다.
    * 기본적인 RNN은 입력이 길어지면 학습 도중 `vanishing gradient` 이나 `explosion` 현상으로 인해 앞부분에 대한 정보를 정확히 담지 못할 수 있다.
    * **GRU는 시계열 데이터 속 벡터 사이의 정보 전달량을 조절함으로써 기울기를 적정하게 유지하고 문장 앞부분의 정보가 끝까지 도달할 수 있도록 도와준다.**
    * GRU에는 시계열 데이터 내 정보 전달량을 조절하는 **update gate**와 **reset gate**라는 개념이 존재한다.
    * **update gate** : 이전 hidden vector가 지닌 정보를 새로운 hidden vector가 얼마나 유지할지 정해준다.
    * **reset gate** : 새로운 입력이 이전 hidden vector와 어떻게 조합하는지 결정한다.

* BasicGRU 전체 코드
  ```
  class BasicGRU(nn.Module):
      def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
          super(BasicGRU, self).__init__()
          print("Building Basic GRU model...")
          self.n_layers = n_layers
          self.embed = nn.Embedding(n_vocab, embed_dim)
          self.hidden_dim = hidden_dim
          self.dropout = nn.Dropout(dropout_p)
          self.gru = nn.GRU(embed_dim, self.hidden_dim,
                            num_layers=self.n_layers,
                            batch_first=True)
          self.out = nn.Linear(self.hidden_dim, n_classes)

      def forward(self, x):
          x = self.embed(x)
          h_0 = self._init_state(batch_size=x.size(0))
          x, _ = self.gru(x, h_0)  # [i, b, h]
          h_t = x[:,-1,:]
          self.dropout(h_t)
          logit = self.out(h_t)  # [b, h] -> [b, o]
          return logit
      
      def _init_state(self, batch_size=1):
          weight = next(self.parameters()).data
          return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
  
  ```


* `Seq2Seq` 예시 : hello를 hola로 번역하는 미니 Seq2Seq 모델
  * Seq2Seq 모델은 시퀀스를 입력받아 또 다른 시퀀스를 출력함.
  * 한마디로 문장을 다른 문장으로 번역해주는 모델이다.
  * 병렬 말뭉치(`parallel corpora`)라고 하는 원문과 번역문이 쌍을 이루는 형태의 많은 텍스트 데이터가 필요함.

  * Seq2Seq 모델은 각자 다른 역할을 하는 두개의 RNN(encoder, decoder)을 이어붙인 모델이다.
    * **Encoder** : 원문 내용을 학습하는 RNN. 원문 속 모든 단어로 하나의 고정 크기 텐서를 생성(=`문맥 벡터::context vector`)함. 원문 마지막 토큰에 해당하는 hidden vector는 원문의 뜻을 모두 내포하고 있는 context vector이다.
      * Autoencoder는 정적인 데이터에서 정보를 추려 차원 수를 줄이고, 축약된 데이터는 원본 데이터의 중요한 내용들만 내포하고 있다. Seq2Seq 모델의 RNN 인코더는 동적인 시계열 데이터를 간단한 형태의 정적인 데이터로 축약한다. 즉, RNN 인코더를 거쳐 만들어진 context vector는 시계열 데이터를 압축한 데이터이다.
    * **Decoder** : encoder에게서 context vector를 이어받아 번역문 속의 토큰을 차례대로 예상한다.
    * decoder가 예상해낸 모든 토큰과 실제 번역문 사이의 오차를 줄여나가는 것이 Seq2Seq 모델이 학습하는 기본 원리이다.
  * **character embedding** : 단어 단위의 워드 임베딩이 아닌 글자 단위의 임베딩
    * 영문을 숫자로 표현하는 방식인 **아스키(ascii)** 코드를 사용해 임베딩
    * `x_ =list(map(ord, "hello"))`
    * `torch.LongTensor(x_)` : 아스키 코드 배열을 파이토치 텐서로 변환
  * **teacher forcing**
    * 티처 포싱은 디코더 학습 시 실제 번역문의 토큰을 디코더의 전 출력값 대신 입력으로 사용해 학습을 가속하는 방법이다. 학습되기 전 모델이 잘못된 예측 토큰을 입력으로 사용되는 것을 방지하는 기법.

* **Adversarial example** & **Adversarial attack** : 적대적 예제 & 적대적 공격
  * 머신러닝 모델의 착시를 유도하는 입력을 뜻함.
  * 적대적 예제를 생성해서 머신러닝 기반 시스템의 성능을 의도적으로 떨어뜨려 보안 문제를 일으키는 적대적 공격을 설명함
  * ※ **Adversarial attack**
    * 적대적 공격에서는 오차를 줄이기보단 극대화하는 쪽으로 잡음을 최적화하게 된다.
    * 모델 정보가 필요한지, 우리가 원하는 정답으로 유도할 수 있는지, 여러 모델을 동시에 헷갈리게 할 수 있는지, 학습이 필요한지 등의 여부에 따라 종류가 나뉨.
    * 종류 1) 기울기와 같은 모델 정보가 필요한지 : `화이트박스`(모델 정보를 토대로 잡음을 생성) 방법과 `블랙박스`(모델 정보없이 생성) 방법이 존재
    * 종류 2) 원하는 정답으로 유도 가능 : `표적(targeted)`, `비표적(non-targeted)`으로 분류
    * 종류 3) 잡음을 생성하기 위해 반복된 학습(최적화)이 필요 : `반복(iterative)`, `원샷(one-shot)`
    * 종류 4) 한 잡음이 특정 입력에만 적용 되는지 or 모든 이미지에 적용될 수 있는 범용적인 잡음인지
    * **가장 강력한 Attack 방법** : 모델 정보가 필요 x, 원하는 정답으로 유도 가능, 복잡한 학습이 필요 x, 여러 모델에 동시에 적용 가능

  * `FGSM(fast gradient sign method)`
    * 정상 데이터에 잡음을 더해 머신러닝 모델을 헷갈리게 하는 데이터 -> 적대적 예제
    * FGSM은 반복 학습 없이 잡음을 생성하는 `one-shot attack`으로, 입력 이미지에 대한 기울기의 정보를 추출하여 잡음을 생성함.
    * 또한, 공격 목표를 정할 수 없는 `non-targeted` 방식이자, 대상 모델의 정보가 필요한 `화이트박스` 방식이다.
    * 모델을 헷갈리게 하려면 모델의 오차값을 극대화해야 한다.





