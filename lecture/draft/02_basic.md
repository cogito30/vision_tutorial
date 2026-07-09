$$CV 튜토리얼$$

# Step 2. 영상 처리의 뼈대, OpenCV와 NumPy 마스터하기

안녕하세요! 컴퓨터 비전 튜토리얼 두 번째 시간입니다.
지난 Step 1에서 Mac OS에 완벽한 비전 개발 환경을 구축하셨다면, 이제 본격적으로 코드를 작성해 볼 차례입니다.

"바로 딥러닝 모델부터 돌려보면 안 되나요?"라고 생각하실 수 있지만, 실무에서는 **데이터 전처리(Preprocessing)** 가 전체 성능의 80% 이상을 좌우합니다.
이미지를 자유자재로 자르고, 색상을 바꾸고, 사이즈를 조절하는 기본기가 탄탄해야 좋은 AI 모델을 만들 수 있습니다.

오늘은 영상 처리의 뼈대가 되는 **OpenCV**와 행렬 연산의 끝판왕 **NumPy**를 활용해 기초적이지만 가장 많이 쓰이는 실습을 진행해 보겠습니다. 📸

## 1. 이미지는 결국 '숫자 배열'이다 (NumPy의 이해)

컴퓨터는 이미지를 어떻게 볼까요? 컴퓨터에게 이미지는 가로, 세로, 그리고 색상(Channel)으로 이루어진 거대한 **숫자 배열(3D Tensor)** 일뿐입니다.

우선 파이썬 스크립트(`image_basic.py`)를 하나 만들고 이미지를 불러와 보겠습니다.
(준비물: 아무 사진이나 `sample.jpg`라는 이름으로 코드와 같은 폴더에 저장해 주세요.)
```
import cv2
import numpy as np

# 1. 이미지 읽기 (OpenCV는 기본적으로 BGR 순서로 읽어옵니다)
img = cv2.imread('sample.jpg')

if img is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인해 주세요.")
    exit()

# 2. NumPy 배열 속성 확인하기
print(f"이미지 타입: {type(img)}")
print(f"이미지 형태(Shape): {img.shape}") # (Height, Width, Channels)
print(f"이미지 데이터 타입: {img.dtype}")

# 3. 이미지 화면에 띄우기
cv2.imshow('Original Image', img)

# 4. 키보드 입력 대기 및 창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()
```


#### 💡 **Mac OS 사용자 팁**:
Mac에서 cv2.imshow()를 사용할 때 창이 먹통이 되는 경우가 종종 있습니다. 이때는 터미널이 아닌 활성화된 이미지 창을 클릭한 뒤 키보드의 아무 키(예: q나 ESC)나 누르면 창이 닫힙니다.

## 2. 웹캠 실시간 제어 및 색상 공간 변환

실무에서는 정지된 이미지뿐만 아니라 CCTV나 웹캠 같은 실시간 영상(Video Stream)을 다루는 일이 많습니다.
이번에는 Mac의 웹캠을 켜고, 실시간으로 색상 공간(Color Space)을 변환해 보겠습니다.

새로운 파일 `webcam_color.py`를 작성해 보세요.
```
import cv2

# 웹캠 연결 (0번은 기본 내장 카메라)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("웹캠 영상 출력을 시작합니다. 종료하려면 'q'를 누르세요.")

while True:
    # 한 프레임씩 읽기 (ret: 성공 여부 Boolean, frame: 이미지 배열)
    ret, frame = cap.read()
    
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break
        
    # Mac 웹캠 특성상 좌우가 반전되어 보일 수 있으므로 플립(거울 모드)
    frame = cv2.flip(frame, 1)

    # 1. 흑백(Grayscale) 변환: 연산량을 줄이기 위해 실무에서 자주 사용
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. HSV 변환: 조명 변화에 강해 특정 색상(예: 빨간불, 노란 차선)을 찾을 때 유리
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 화면에 출력
    cv2.imshow('Original BGR', frame)
    cv2.imshow('Grayscale', gray_frame)
    cv2.imshow('HSV Space', hsv_frame)

    # 'q' 키를 누르면 루프 탈출
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
```

코드를 실행하면 3개의 창이 뜨면서 여러분의 얼굴이 각기 다른 색상으로 출력될 것입니다. 특히 **HSV 색상 공간**은 나중에 특정 물체를 추적할 때 매우 유용하게 쓰이니 꼭 기억해 두세요!

## 3. NumPy를 활용한 이미지 Crop과 Padding

이미지에서 원하는 부분만 잘라내거나(Crop), 모델의 입력 크기에 맞게 여백을 채우는(Padding) 작업은 딥러닝 전처리의 핵심입니다. OpenCV 함수를 쓸 수도 있지만, 이미지는 결국 **NumPy** 배열이므로 파이썬의 **슬라이싱(Slicing)** 을 이용하면 아주 간단합니다.
```
import cv2

img = cv2.imread('sample.jpg')

# 1. Image Crop (자르기)
# Numpy Slicing: img[y_start:y_end, x_start:x_end]
# (주의: OpenCV에서 y축은 위에서 아래로 증가합니다)
cropped_img = img[100:400, 200:500] 

# 2. Image Resize (크기 조절)
# (너비, 높이) 순서로 지정합니다.
resized_img = cv2.resize(cropped_img, (224, 224)) 

# 3. Image Padding (여백 채우기)
# 위, 아래, 왼쪽, 오른쪽에 50픽셀씩 검은색(0,0,0) 테두리를 추가합니다.
padded_img = cv2.copyMakeBorder(
    resized_img, 
    top=50, bottom=50, left=50, right=50, 
    borderType=cv2.BORDER_CONSTANT, 
    value=[0, 0, 0] # BGR 컬러 (검은색)
)

cv2.imshow('Padded & Resized Image', padded_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

이 기법들은 나중에 PyTorch로 CNN 모델을 학습시킬 때, 데이터 증강(Data Augmentation) 파이프라인을 구축하는 가장 기본적인 원리가 됩니다.

## 마치며

오늘은 컴퓨터 비전 개발의 기본기인 OpenCV 웹캠 제어, 색상 변환, 그리고 NumPy를 이용한 이미지 자르기/크기 조절에 대해 알아보았습니다. 코드가 짧고 간단하지만, 자율주행이나 불량 검출 같은 거대한 시스템도 결국 이 작은 코드들에서 출발합니다.

다음 **Step 3**에서는 전통적인 컴퓨터 비전 기술의 꽃이라고 할 수 있는 **이미지 필터링(Blur)** 과 **외곽선(Edge) 검출 기법** 에 대해 알아보겠습니다.
직접 웹캠을 띄워 놓고 여러분 주변의 사물 외곽선을 실시간으로 따보는 재밌는 실습이 기다리고 있으니 기대해 주세요!
