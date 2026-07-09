$$CV 튜토리얼$$

# Step 3. 전통적 컴퓨터 비전: 필터링과 외곽선 검출

안녕하세요! 컴퓨터 비전 튜토리얼 세 번째 시간입니다.

"요즘은 딥러닝이 다 해주는 거 아닌가요?" 컴퓨터 비전을 처음 공부하시는 분들이 많이 하는 질문입니다. 절반은 맞고 절반은 틀립니다!
의료 AI에서 종양의 크기를 재거나, 공장 레일 위에서 불량품의 스크래치를 찾을 때, 혹은 영수증의 글씨(OCR)를 읽기 좋게 다듬을 때 **전통적 컴퓨터 비전(Traditional CV)** 기법은 여전히 빠르고 강력한 무기입니다. (무엇보다 가볍고 GPU도 필요 없습니다!)

오늘은 이미지에서 노이즈를 제거하는 **필터링(Blurring)** 과 물체의 형태를 파악하는 **외곽선 검출(Edge Detection)**, 그리고 이를 응용한 **차선 인식 기초**를 실습해 보겠습니다.

## 1. 이미지 노이즈 제거: 가우시안 블러 (Gaussian Blur)
현실 세계의 카메라로 찍은 사진에는 항상 자잘한 '노이즈(Noise)'가 껴있습니다. 이 노이즈를 그대로 둔 채로 선이나 물체를 찾으려고 하면, 컴퓨터는 노이즈마저 선으로 착각하게 됩니다.
그래서 우리는 이미지를 살짝 흐리게(Blur) 만들어서 노이즈를 뭉개버리는 작업을 먼저 해야 합니다. 실무에서 가장 많이 쓰이는 것이 바로 **가우시안 블러**입니다.

## 2. 물체의 윤곽선 찾기: 캐니 에지 (Canny Edge Detection)

노이즈를 제거했다면, 이제 픽셀의 색상이 급격하게 변하는 지점(경계선)을 찾을 차례입니다.
가장 대표적이고 성능이 좋은 알고리즘이 바로 **Canny Edge Detector**입니다.

백문이 불여일견! Step 2에서 배웠던 웹캠 코드에 블러와 외곽선 검출을 적용해 실시간으로 확인해 봅시다.
새로운 파이썬 파일 realtime_edge.py를 만들고 아래 코드를 실행해 보세요.
```
import cv2

cap = cv2.VideoCapture(0)

print("실시간 외곽선 검출을 시작합니다. 종료하려면 'q'를 누르세요.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Mac 거울 모드
    frame = cv2.flip(frame, 1)

    # 1. 흑백 변환 (연산량 최소화)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. 가우시안 블러 (노이즈 제거)
    # 커널 크기는 (5, 5)처럼 항상 홀수여야 합니다. 숫자가 클수록 더 흐려집니다.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Canny Edge 검출
    # cv2.Canny(이미지, 최소 임계값, 최대 임계값)
    # 픽셀 변화량이 150 이상이면 무조건 외곽선으로 간주, 50~150 사이는 연결된 선일 경우만 인정
    edges = cv2.Canny(blurred, 50, 150)

    # 화면 출력
    cv2.imshow('Original Video', frame)
    cv2.imshow('Canny Edges', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

- **실행 결과**: 여러분의 얼굴이나 주변 사물들의 스케치 도면처럼 선만 추출되어 보일 것입니다. 손이나 물건을 움직여보면서 경계선이 어떻게 잡히는지 관찰해 보세요!

## 3. 허프 변환(Hough Transform)으로 직선 찾기: 자율주행 차선 인식 기초

외곽선을 찾았다면, 이 외곽선들 중에서 '직선'만 골라낼 수는 없을까요?
이때 사용하는 수학적 기법이 **허프 변환(Hough Transform)** 입니다. 자율주행 자동차가 도로의 차선을 인식하는 가장 기초적인 원리가 바로 Canny Edge + Hough Transform의 조합입니다.

인터넷에서 적당한 도로 사진(`road_sample.jpg`)을 하나 다운로드하여 같은 폴더에 저장한 뒤, `lane_detection.py`를 작성해 보세요.
```
import cv2
import numpy as np

# 1. 이미지 로드 및 기본 전처리
img = cv2.imread('road_sample.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

# 2. ROI (Region of Interest, 관심 영역) 설정
# 도로의 차선은 주로 화면의 아래쪽에 위치하므로, 하늘 등 불필요한 영역의 외곽선은 무시합니다.
height, width = img.shape[:2]
mask = np.zeros_like(edges) # 검은색 빈 캔버스 생성

# 차선이 있을 만한 사다리꼴 좌표 설정 (이미지 크기에 따라 수정 필요)
polygon = np.array([[
    (50, height),                   # 좌측 하단
    (width//2 - 50, height//2 + 50), # 좌측 상단 (중앙 부근)
    (width//2 + 50, height//2 + 50), # 우측 상단 (중앙 부근)
    (width - 50, height)             # 우측 하단
]], np.int32)

# 다각형 영역만 흰색(255)으로 채우기
cv2.fillPoly(mask, polygon, 255)

# 원본 외곽선 이미지와 마스크를 AND 연산하여 관심 영역의 외곽선만 남김
masked_edges = cv2.bitwise_and(edges, mask)

# 3. 허프 변환을 이용한 직선 검출 (확률적 허프 변환)
# minLineLength: 선으로 인정할 최소 길이
# maxLineGap: 끊어져 있어도 하나의 선으로 이어줄 최대 간격
lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, 
                        threshold=50, minLineLength=40, maxLineGap=20)

# 4. 원본 이미지 위에 찾은 빨간색(0, 0, 255) 직선 그리기
line_img = np.zeros_like(img)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0] # 직선의 시작점(x1, y1)과 끝점(x2, y2)
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

# 원본 이미지와 선이 그려진 이미지를 합치기 (투명도 조절)
result = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

# 결과 출력
cv2.imshow('Canny Edges', edges)
cv2.imshow('Masked Edges', masked_edges)
cv2.imshow('Lane Detection', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

이 코드를 실행하면 복잡한 도로 사진 속에서 양옆의 차선 부분만 붉은색 직선으로 깔끔하게 그려지는 것을 확인할 수 있습니다.

## 마치며

오늘은 가우시안 블러로 노이즈를 다듬고, 캐니 에지로 형태를 따고, 허프 변환으로 차선을 찾아보는 전통적 비전 기술의 정수를 맛보았습니다. 딥러닝이 아무리 발전해도, 이미지를 모델에 넣기 전 'ROI(관심 영역)'를 자르고 가공하는 이런 기법들은 여러분을 실무에서 빛나게 해 줄 기초 체력이 됩니다.

다음 Step 4에서는 두 장의 사진을 파노라마처럼 이어 붙일 때 사용하는 핵심 기술인 **특징점 추출과 매칭(Feature Matching)** 에 대해 알아보겠습니다. 드디어 Phase 1의 마지막 단계이니 조금만 더 힘내주세요!
