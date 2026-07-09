$$CV 튜토리얼$$

# Step 8. 픽셀 단위로 물체를 따내다: 이미지 분할(Image Segmentation) 완벽 가이드

안녕하세요! 컴퓨터 비전 튜토리얼 여덟 번째 시간입니다.

지난 Step 7에서는 사물의 위치를 네모난 박스(Bounding Box)로 찾는 '객체 탐지(YOLO)'를 배웠습니다. 하지만 박스만으로는 부족한 경우가 있습니다.
예를 들어, **의료 AI**가 엑스레이에서 종양의 정확한 크기와 모양을 알아내야 할 때나, **자율주행 자동차**가 울퉁불퉁한 주행 가능 영역(도로)을 정확히 파악해야 할 때는 네모 박스가 아니라 '픽셀 단위의 정교한 외곽선'이 필요합니다.

이렇게 이미지의 모든 픽셀을 분석해 어떤 객체에 속하는지 분류하는 기술을 이미지 분할(Image Segmentation)이라고 합니다. 오늘은 이 분야의 양대 산맥인 **U-Net**과 **Instance Segmentation**을 실습해 보겠습니다!

## 1. 이미지 분할의 2가지 종류
본격적인 실습 전에 알아두어야 할 핵심 개념이 있습니다.

1. **시맨틱 분할 (Semantic Segmentation)**: 이미지의 모든 픽셀을 '클래스' 단위로만 나눕니다. (예: 화면에 있는 자동차 3대를 모두 같은 색으로 칠함)
2. **인스턴스 분할 (Instance Segmentation)**: 클래스뿐만 아니라 '개별 객체'까지 구분합니다. (예: 자동차 1은 빨간색, 자동차 2는 파란색, 자동차 3은 초록색으로 다르게 칠함)

## 2. 시맨틱 분할의 교과서: U-Net 구조 이해하기
의료 영상 분할 대회를 휩쓸며 등장한 **U-Net**은 네트워크 모양이 알파벳 'U'자를 닮아서 붙여진 이름입니다.
이미지를 점점 작게 압축하며 전체적인 특징을 파악하는 **인코더(Encoder)** 파트와, 다시 원래 크기로 복원하며 픽셀의 위치를 정교하게 찍어주는 **디코더(Decoder)** 파트로 나뉩니다. 이때 잃어버린 위치 정보를 보완하기 위해 인코더의 특징을 디코더로 바로 넘겨주는 스킵 연결(Skip Connection)이 핵심입니다.

#### 💻 실습 1: PyTorch로 Mini U-Net 밑바닥부터 짜보기
새로운 파이썬 파일 `unet_model.py`를 만들고, Mac GPU(`mps`)에서 동작하는 뼈대 코드를 직접 구현해 봅시다.
```
import torch
import torch.nn as nn

class MiniUNet(nn.Module):
    def __init__(self):
        super(MiniUNet, self).__init__()
        
        # 1. Encoder (내려가는 길: 특징 추출)
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        # 2. Bottleneck (U자의 가장 아래쪽)
        self.bottleneck = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # 3. Decoder (올라가는 길: 픽셀 복원)
        # Up-Convolution (해상도 2배 확대)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1) # 128 + 128 (Skip connection)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 64 + 64 (Skip connection)
        
        # 4. Final Output (클래스 개수에 맞게 채널 수 조절, 예: 배경 vs 전경 = 2개)
        self.final_conv = nn.Conv2d(64, 2, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        e1 = torch.relu(self.enc_conv1(x))
        p1 = self.pool1(e1)
        
        e2 = torch.relu(self.enc_conv2(p1))
        p2 = self.pool2(e2)
        
        # Bottleneck
        b = torch.relu(self.bottleneck(p2))
        
        # Decoder & Skip Connections
        u1 = self.up1(b)
        # 💡 핵심: 축소할 때의 특징(e2)을 복원할 때(u1) 이어 붙여줍니다(Concat)
        merged1 = torch.cat([u1, e2], dim=1) 
        d1 = torch.relu(self.dec_conv1(merged1))
        
        u2 = self.up2(d1)
        # 💡 핵심: 축소할 때의 특징(e1)을 이어 붙여줍니다.
        merged2 = torch.cat([u2, e1], dim=1)
        d2 = torch.relu(self.dec_conv2(merged2))
        
        out = self.final_conv(d2)
        return out

# 모델 테스트 (가상의 이미지 텐서를 넣어서 크기가 제대로 복원되는지 확인)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = MiniUNet().to(device)

# (Batch=1, Channels=3, Height=256, Width=256) 형태의 가상 이미지
dummy_input = torch.randn(1, 3, 256, 256).to(device)
output = model(dummy_input)

print(f"입력 크기: {dummy_input.shape}")
print(f"출력 크기: {output.shape}") 
# 정상적이라면 입력과 똑같이 (1, 2, 256, 256) 사이즈가 나와야 합니다!
```
이 코드는 실무에서 사용하는 Segmentation 모델의 가장 근본적인 뼈대입니다. 이 모델에 데이터셋을 넣어 학습시키면 훌륭한 의료 AI나 결함 검출 AI를 만들 수 있습니다.

## 3. 웹캠으로 즐기는 실시간 인스턴스 분할 (YOLOv8-seg)
U-Net을 직접 학습시키려면 마스크(Mask) 라벨링이 된 방대한 데이터셋과 시간이 필요합니다.
우리는 지난 시간에 배운 **Ultralytics YOLO**의 힘을 빌려보겠습니다. YOLO는 객체 탐지뿐만 아니라 **인스턴스 분할(Instance Segmentation)** 모델도 기본적으로 제공합니다.

`realtime_segmentation.py` 파일을 만들고 아래 코드를 실행해 보세요!

```
import cv2
from ultralytics import YOLO

# 1. 모델 로드: 기존 'yolov8n.pt' 뒤에 '-seg'가 붙은 분할 전용 모델을 사용합니다.
model = YOLO('yolov8n-seg.pt')

# 2. 웹캠 연결
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("웹캠 실시간 픽셀 분할(Segmentation)을 시작합니다. 종료하려면 'q'를 누르세요.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Mac 웹캠 거울 모드
    frame = cv2.flip(frame, 1)

    # 3. 모델 추론 (mps 가속)
    results = model(frame, device='mps', verbose=False)

    # 4. 결과 시각화
    # plot() 함수가 바운딩 박스뿐만 아니라 픽셀 단위 마스크까지 자동으로 그려줍니다.
    annotated_frame = results[0].plot()

    # 화면에 출력
    cv2.imshow("Real-time Instance Segmentation", annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**실행 결과**:
웹캠 화면에 여러분의 모습이나 주변 사물들이 나타날 때, 단순히 네모난 박스가 쳐지는 것을 넘어서 사람의 윤곽선, 핸드폰의 모양, 컵의 둥근 형태를 따라 알록달록한 반투명 색상이 정교하게 칠해지는 것을 볼 수 있습니다. 영화의 CG 작업(크로마키 없이 배경 날리기 등)이 바로 이 기술로 이루어집니다.

## 마치며

오늘 우리는 자율주행과 의료 AI의 핵심 기술인 이미지 분할(Segmentation)의 원리를 U-Net을 통해 이해하고, YOLO를 이용해 웹캠에서 실시간 인스턴스 분할까지 구현해 보았습니다.

이로써 컴퓨터 비전의 3대장 태스크인 분류(Classification), 탐지(Detection), 분할(Segmentation)을 모두 정복하셨습니다!

다음 **Step 9**부터는 Phase 4(심화 비전 알고리즘)로 들어갑니다. 사진 한 장을 분석하는 것을 넘어, 영상 속에서 사람이 어떻게 움직이는지 추적(Tracking)하고 관절의 위치를 뽑아내는(Pose Estimation) 고급 기술을 다루어보겠습니다. 다음 포스팅에서 만나요!
