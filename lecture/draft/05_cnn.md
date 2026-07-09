$$CV 튜토리얼$$

# Step 5. 딥러닝의 시작: CNN 기초와 PyTorch 이미지 분류

안녕하세요! 컴퓨터 비전 튜토리얼 다섯 번째 시간입니다.
지금까지 Phase 1에서 OpenCV와 함께 '규칙 기반'의 영상 처리 기초를 다졌다면, 오늘부터 시작되는 **Phase 2**에서는 드디어 **'데이터 기반'의 딥러닝(Deep Learning)** 세계로 진입합니다. 🎉

딥러닝 비전의 꽃이라고 불리는 CNN(합성곱 신경망)을 직접 설계해 보고, Apple Silicon(M1/M2 등)의 GPU 가속기인 `MPS`를 100% 활용하여 개, 고양이, 자동차 등 10가지 사물을 분류하는 인공지능 모델을 밑바닥부터 학습시켜 보겠습니다!

## 1. CNN(Convolutional Neural Network)이란?

이전 Step 4에서 우리는 이미지의 특징(Feature)을 찾기 위해 ORB 같은 알고리즘을 사용했습니다. 사람이 "이런 방식으로 모서리를 찾아라!"라고 수학적 공식을 수학적으로 정의해 둔 것이죠.

하지만 **CNN**은 다릅니다. 수많은 이미지를 보여주며 정답을 알려주면, 모델 스스로 "아, 고양이는 뾰족한 귀가 특징이구나", "자동차는 둥근 바퀴가 중요하네" 라며 **특징을 추출하는 필터를 스스로 학습**합니다.
이미지를 훑고 지나가며 특징을 뽑아내는 **Conv(합성곱) 층**과, 불필요한 정보를 줄이고 핵심만 남기는 **Pooling 층**이 샌드위치처럼 겹겹이 쌓인 구조를 생각하시면 됩니다.

## 2. PyTorch 학습의 4단계 파이프라인
PyTorch로 딥러닝 모델을 만들 때는 보통 다음 4단계를 거칩니다.

1. 데이터 준비 (Dataset & DataLoader)
2. 모델 설계 (Model Architecture)
3. 손실 함수와 최적화 기법 설정 (Loss & Optimizer)
4. 학습 및 평가 (Train & Evaluation Loop)

이 구조를 머릿속에 넣고, 파이썬 파일 `train_cnn.py`를 만들어 차근차근 코드를 작성해 봅시다.

#### 💻 실습: CIFAR-10 이미지 분류 모델 만들기

복잡한 데이터 다운로드 과정을 생략하기 위해, PyTorch에서 기본 제공하는 `CIFAR-10` 데이터셋(32x32 크기의 10가지 사물/동물 컬러 이미지 6만 장)을 사용하겠습니다.

#### 1단계: 환경 세팅 및 데이터 로더 준비
```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 💡 핵심: Mac GPU(MPS) 디바이스 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"현재 사용 중인 디바이스: {device}")

# 데이터 전처리 (Tensor로 변환 후 정규화)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 훈련(Train) 데이터셋 및 데이터로더
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 평가(Test) 데이터셋 및 데이터로더
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

classes = ('비행기', '자동차', '새', '고양이', '사슴', '개', '개구리', '말', '배', '트럭')
```


#### 2단계: Custom CNN 모델 설계
간단하지만 강력한 2개의 Conv 층을 가진 CNN 모델을 만들어봅니다.
```
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 입력 채널=3(RGB 컬러), 출력 채널=16, 필터 크기=3x3
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2) # 크기를 절반으로 줄임
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # FC(Fully Connected) 레이어: 특징들을 모아서 최종 10개 클래스로 분류
        # 32(채널) * 8(너비) * 8(높이)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10) # 최종 출력 10개 (CIFAR-10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8) # 1차원 벡터로 쭉 펴기 (Flatten)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델을 생성하고 mps 디바이스로 보냅니다.
model = SimpleCNN().to(device)
```


#### 3단계 & 4단계: 손실 함수 지정 및 훈련 루프 작성
```
# 다중 분류에 사용하는 교차 엔트로피 손실함수
criterion = nn.CrossEntropyLoss()
# 모델의 가중치를 업데이트할 옵티마이저 (Adam 사용)
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10 # 전체 데이터를 10번 반복 학습

print("🚀 학습을 시작합니다!")
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 1. 데이터를 mps 디바이스로 이동
        inputs, labels = data[0].to(device), data[1].to(device)

        # 2. 기울기 초기화
        optimizer.zero_grad()

        # 3. 순전파(Forward) -> 손실 계산(Loss) -> 역전파(Backward) -> 가중치 업데이트(Step)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    # 에폭마다 평균 Loss 출력
    print(f"[Epoch {epoch + 1}/{epochs}] Loss: {running_loss / len(trainloader):.4f}")

print("✅ 학습 완료!")

# ----------------------------------------------------
# [평가 단계] 학습되지 않은 Test 데이터로 성능 검증
correct = 0
total = 0

# 평가할 때는 기울기(Gradient)를 계산할 필요가 없으므로 torch.no_grad() 사용
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) # 가장 확률이 높은 클래스 선택
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"🎯 테스트 데이터 정확도(Accuracy): {100 * correct / total:.2f}%")
```

위 코드를 실행하면, 데이터가 다운로드된 후 터미널에 Loss 값이 점점 줄어들고 마지막에 약 60~70% 내외의 정확도가 출력되는 것을 볼 수 있습니다! (네트워크가 작고 짧게 학습해서 이 정도지만, 이것만으로도 여러분은 인공지능을 직접 탄생시킨 것입니다.)

## 마치며

축하합니다! 여러분은 오늘 처음으로 데이터셋 구축, 모델 설계, GPU를 활용한 학습 루프 작성까지 전체 **딥러닝 파이프라인**을 관통해 보았습니다.

하지만 실무에서 우리가 매번 모델 구조를 바닥부터 짜고 있을 수는 없겠죠?
다음 **Step 6**에서는 구글이나 마이크로소프트 같은 글로벌 대기업들이 어마어마한 데이터와 컴퓨팅 자원으로 미리 똑똑하게 학습시켜 놓은 모델을 가져와, 내 입맛에 맞게 살짝만 고쳐 쓰는 **전이 학습(Transfer Learning)** 이라는 실무 필수 기술을 배워보겠습니다.

다음 포스팅도 기대해 주세요!
