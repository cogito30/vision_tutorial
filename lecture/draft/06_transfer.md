$$CV 튜토리얼$$

# Step 6. 실무 AI의 핵심: 전이 학습(Transfer Learning)으로 불량 판정 모델 만들기

안녕하세요! 컴퓨터 비전 튜토리얼 여섯 번째 시간입니다.

지난 Step 5에서 우리는 딥러닝 모델(CNN)을 밑바닥부터 설계하고 개와 고양이를 분류해 보았습니다. 하지만 결과가 어땠나요? 시간이 꽤 걸렸는데도 정확도는 60~70% 언저리로 아쉬웠을 것입니다.

실제 회사 업무(예: 공장의 불량품 판별, 병원의 질병 진단)에서 매번 이렇게 처음부터 모델을 만들고 학습시킬까요? 절대 아닙니다.
실무에서는 구글, 메타 같은 빅테크 기업들이 슈퍼컴퓨터로 수백만 장의 이미지를 미리 학습시켜 놓은 초거대 모델을 가져와, **내 데이터에 맞게 살짝만 고쳐 쓰는 방식**을 사용합니다. 이것을 전이 학습(Transfer Learning)이라고 합니다.

오늘은 이 전이 학습을 이용해, 단 몇 분의 학습만으로 제조업 불량 판정 모델(정확도 95% 이상 목표)을 만들어 보겠습니다! 🚀

## 1. 전이 학습이란? (거인의 어깨 위에 올라타기)

**ImageNet**이라는 세계구급 데이터셋(1,000개의 클래스, 120만 장 이상의 이미지)으로 미리 똑똑하게 학습된(Pre-trained) 모델들은 이미 '선', '색상', '질감', '사물의 형태'를 파악하는 방법을 완벽하게 깨우치고 있습니다.

우리는 이 천재 모델(ResNet, EfficientNet 등)의 뇌를 그대로 가져온 뒤, 마지막에 정답을 맞히는 **'출력층(분류기)'만 똑 떼어내서 우리 목적(양품 vs 불량품)에 맞게 교체**해 줍니다. 그리고 우리의 데이터로 가볍게 재학습(Fine-tuning)을 시키면, 아주 적은 데이터와 시간만으로도 압도적인 성능을 낼 수 있습니다.

## 2. 실무형 커스텀 데이터셋 준비 (ImageFolder)

실무에서 내 데이터를 PyTorch 모델에 넣으려면 폴더 구조를 잘 잡는 것이 핵심입니다. PyTorch의 ImageFolder 기능을 사용하면 아래와 같이 폴더별로 이미지를 나누어 넣기만 해도 자동으로 라벨링이 끝납니다.

작업 중인 폴더에 아래와 같은 구조로 임의의 이미지들(양품/불량품 또는 강아지/고양이 등 2가지 분류)을 넣어주세요.
```
my_dataset/
├── train/          (학습용 데이터)
│   ├── normal/     (양품 이미지들...)
│   └── defect/     (불량품 이미지들...)
└── val/            (검증용 데이터)
    ├── normal/     
    └── defect/     
```


## 3. ResNet50을 활용한 전이 학습 코드 작성
새로운 파이썬 파일 transfer_learning.py를 생성하고 아래 코드를 작성해 봅시다. 이번에도 Mac의 MPS 가속을 적극 활용합니다!

#### 1단계: 데이터 로드 및 전처리
```
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os

# Mac GPU(MPS) 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# ImageNet 사전 학습 모델들이 사용했던 표준 전처리 방식 적용
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)), # ResNet 기본 입력 사이즈
        transforms.RandomHorizontalFlip(), # 데이터 증강: 좌우 반전
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'my_dataset' # 위에서 만든 폴더 이름
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes # ['defect', 'normal']

print(f"클래스 종류: {class_names}")
```


#### 2단계: 사전 학습된 모델 불러오기 및 개조(Fine-tuning)
```
# 1. ImageNet으로 사전 학습된 ResNet50 모델 불러오기
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# 2. 마지막 분류기(Fully Connected Layer) 교체하기
# ResNet50의 마지막 레이어 이름은 'fc'입니다. 기존 1,000개 출력에서 2개(정상/불량)로 변경!
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2) # 출력 클래스를 2개로 설정

model = model.to(device)

criterion = nn.CrossEntropyLoss()
# 변경된 모델 전체를 아주 작은 학습률(lr)로 미세 조정(Fine-tuning)합니다.
optimizer = optim.Adam(model.parameters(), lr=0.0001)
```


#### 3단계: 검증이 포함된 실무형 학습 루프
단순히 학습만 하는 것이 아니라, 매 에폭(Epoch)마다 검증 데이터(Val)로 정확도를 평가합니다.
```
num_epochs = 5 # 이미 똑똑한 모델이라 5번만 돌아도 충분합니다!

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    # 각 에폭마다 학습(train)과 검증(val) 단계를 거침
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # 모델을 학습 모드로 설정
        else:
            model.eval()   # 모델을 평가 모드로 설정

        running_loss = 0.0
        running_corrects = 0

        # 데이터 반복
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 순전파
            # 학습 모드에서만 기울기(gradient)를 계산함
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # 학습 모드인 경우 역전파 및 가중치 최적화
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # 통계
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase.upper()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

print('✅ 학습 완료!')
```


코드를 실행해 보시면, 단 1~2 에폭(Epoch) 만에 검증(VAL) 데이터의 정확도(Acc)가 90%를 가볍게 뚫어버리는 마법 같은 경험을 하실 수 있습니다. 이것이 바로 실무에서 전이 학습을 사랑하는 이유입니다.

## 🎯 Phase 2 수료를 축하합니다!

여러분은 이제 딥러닝 네트워크 구조를 이해하고, 대기업의 강력한 모델을 가져와 내 프로젝트에 맞게 커스터마이징 하는 '실무형 AI 개발자'의 역량을 갖추게 되었습니다.

지금까지 우리는 이미지가 주어지면 "이 사진이 무엇이냐(Classification)?"를 맞히는 기술을 배웠습니다.
하지만 자율주행 자동차는 화면 속에서 '자동차가 어디에 있는지', '사람이 몇 명 있는지' 그 **위치**를 알아야만 합니다.

다음 **Phase 3의 시작점, Step 7**에서는 현재 컴퓨터 비전 업계에서 가장 핫한 객체 탐지 알고리즘, YOLO(You Only Look Once)를 마스터해 보겠습니다. 진짜 눈(Eye)을 가진 인공지능을 만들 준비가 되셨나요? 다음 포스팅에서 뵙겠습니다!
