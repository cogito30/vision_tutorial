$$CV 튜토리얼$$

# Step 7. 진짜 눈을 가진 AI: YOLO로 실시간 객체 탐지(Object Detection) 마스터하기

안녕하세요! 컴퓨터 비전 튜토리얼 일곱 번째 시간입니다.
오늘부터는 실무에서 가장 수요가 많고 강력한 기술들을 다루는 **Phase 3: 핵심 컴퓨터 비전 태스크**로 진입합니다!

지난 시간까지 배운 '이미지 분류(Classification)'는 "이 사진이 개냐, 고양이냐?"를 맞히는 기술이었습니다.
하지만 자율주행 자동차나 CCTV 관제 시스템에서는 사진 속에 여러 물체가 섞여 있고, 그 물체들이 **어디에(위치), 얼마나 있는지**를 실시간으로 알아야 합니다. 이렇게 물체의 위치에 박스(Bounding Box)를 치고 클래스를 분류하는 기술을 **객체 탐지(Object Detection)** 라고 합니다.

오늘은 객체 탐지 분야의 절대 강자, **YOLO(You Only Look Once)** 를 Mac 환경에서 완벽하게 구동해 보겠습니다. 🚀

## 1. YOLO와 Ultralytics 생태계
과거에는 객체 탐지 모델을 학습시키기 위해 수백 줄의 코드를 짜고 복잡한 C++ 라이브러리를 설치해야 했습니다. 하지만 지금은 **Ultralytics**라는 회사에서 제공하는 파이썬 패키지 덕분에 단 세 줄의 코드만으로 강력한 YOLO(YOLOv8, YOLO11 등) 모델을 다룰 수 있게 되었습니다.

먼저 가상환경(`cv_env`)이 활성화된 터미널에서 Ultralytics 패키지를 설치해 줍니다.
```
pip install ultralytics
```

💡 이 패키지 하나만 설치하면 *PyTorch*와 *YOLO*에 필요한 모든 의존성이 완벽하게 해결됩니다.

## 2. 커스텀 데이터셋 준비 (Roboflow 활용)
YOLO 모델을 내 입맛에 맞게 학습시키려면 데이터가 필요합니다.
이미지마다 물체가 어디 있는지 박스를 그리고 좌표를 기록하는 작업(라벨링)을 해야 하는데, 실무에서는 **Roboflow**라는 플랫폼을 가장 많이 사용합니다.

1. [Roboflow Universe](https://universe.roboflow.com/) 에 접속합니다.
2. 원하는 데이터셋을 검색합니다. (예: `Hard Hat Workers` - 안전모 착용 감지)
3. 다운로드 버튼을 누르고 포맷을 `YOLOv8` 로 선택하여 다운받습니다.
4. 다운로드한 압축 파일을 풀면 `train`, `val`, `test` 폴더와 함께 `data.yaml` 이라는 파일이 있습니다. 이 `data.yaml` 파일이 바로 YOLO에게 "데이터가 여기 있고, 정답 클래스는 이거야!"라고 알려주는 지도 역할을 합니다.

## 3. Mac GPU(MPS)로 YOLO 모델 학습하기
데이터가 준비되었다면 학습을 진행해 봅시다.
놀랍게도 Ultralytics는 Mac의 Apple Silicon GPU(`mps`)를 완벽하게 지원합니다! 새로운 파이썬 파일 `train_yolo.py`를 만들고 실행해 보세요.

(데이터가 없으신 분들은 아래 코드의 `data` 부분을 `coco8.yaml`로 두시면, 샘플 데이터를 자동으로 다운로드하여 테스트 학습을 진행합니다.)
```
from ultralytics import YOLO

# 1. 사전 학습된 가장 가벼운 Nano 모델 불러오기
# 처음부터 학습하는 것보다 전이 학습(Transfer Learning)을 하는 것이 훨씬 빠릅니다.
model = YOLO('yolov8n.pt') 

print("🚀 YOLO 학습을 시작합니다!")

# 2. 모델 학습 (Training)
# - data: 위에서 설명한 data.yaml 파일의 절대 또는 상대 경로
# - epochs: 전체 데이터 학습 반복 횟수
# - imgsz: 입력 이미지 크기 (기본 640)
# - device: 'mps'를 지정하여 Mac GPU 가속 사용!
results = model.train(
    data='coco8.yaml', # 커스텀 데이터가 있다면 '경로/data.yaml'로 변경하세요.
    epochs=10,
    imgsz=640,
    device='mps'
)

print("✅ 학습이 완료되었습니다. 결과는 'runs/detect/train' 폴더에 저장됩니다.")
```

학습이 끝나면 프로젝트 폴더 내에 `runs/detect/train/weights/` 경로가 생성되고, 그 안에 가장 학습이 잘 된 `best.pt` 파일이 만들어집니다. 이것이 바로 여러분만의 커스텀 AI 뇌입니다!

### 4. 웹캠으로 실시간 객체 탐지하기 (Inference)
이제 대망의 하이라이트입니다!
방금 학습한 모델(또는 기본 모델)을 가져와 Mac의 웹캠에 연결하고 실시간으로 물체를 탐지해 보겠습니다. `realtime_yolo.py`를 작성합니다.
```
import cv2
from ultralytics import YOLO

# 1. 모델 로드
# 커스텀 학습을 했다면 'runs/detect/train/weights/best.pt' 경로를 입력하세요.
# 여기서는 기본 사전 학습 모델(yolov8n.pt)을 사용합니다.
model = YOLO('yolov8n.pt') 

# 2. 웹캠 연결
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("웹캠 실시간 탐지를 시작합니다. 종료하려면 'q'를 누르세요.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Mac 웹캠 거울 모드
    frame = cv2.flip(frame, 1)

    # 3. YOLO 추론 (Inference)
    # 이미지나 프레임을 모델에 넣기만 하면 끝입니다! (속도를 위해 mps 사용)
    results = model(frame, device='mps', verbose=False)

    # 4. 결과 시각화
    # results[0].plot()은 바운딩 박스와 클래스 이름이 그려진 이미지 배열을 반환합니다.
    annotated_frame = results[0].plot()

    # 화면에 출력
    cv2.imshow("YOLO Real-time Detection", annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

코드를 실행하고 웹캠 앞에 스마트폰, 컵, 마우스 등을 비춰보세요!
빠른 속도로 부드럽게 네모 박스가 그려지면서 물체의 이름을 정확하게 맞히는 것을 볼 수 있습니다. 이것이 바로 테슬라의 오토파일럿이나 스마트 팩토리에서 사용하는 기술의 뼈대입니다.

## 마치며

축하합니다! 여러분은 오늘 컴퓨터 비전 분야의 엑스칼리버라고 불리는 **YOLO**를 성공적으로 다루게 되었습니다. 불과 몇 년 전만 해도 엄청난 세팅이 필요했던 작업이, 이제는 파이썬 코드 몇 줄과 Mac GPU만으로 가능해졌다는 사실이 놀랍지 않으신가요?

다음 **Step 8**에서는 한 걸음 더 나아갑니다. 바운딩 박스(네모 상자)로 대략적인 위치만 잡는 것을 넘어, 물체의 픽셀 단위 외곽선을 정교하게 따내는 **이미지 분할(Image Segmentation)** 기술에 대해 알아보겠습니다. 의료 AI나 자율주행 차선 인식에 필수적인 기술이니 많은 기대 부탁드립니다!
