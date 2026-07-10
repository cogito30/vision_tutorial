$$ \[CV 심화 튜토리얼\] $$

#Step 2. 라벨링 노가다의 종말: SAM 2 (Segment Anything Model 2) 마스터하기

안녕하세요! 컴퓨터 비전 심화 튜토리얼 두 번째 시간입니다.
지난 시간에 배운 DINOv2가 이미지의 '특징'을 스스로 깨우친 뇌(Brain)였다면, 오늘 배울 **SAM 2**는 세상 모든 물체의 '형태'를 픽셀 단위로 완벽하게 오려내는 마법의 가위(Scissors)라고 할 수 있습니다.

과거에는 U-Net이나 Mask R-CNN 같은 분할(Segmentation) 모델을 만들기 위해 사람이 일일이 수만 장의 사진에 다각형 점을 찍어가며 외곽선을 그리는 엄청난 노동(라벨링)이 필요했습니다. 하지만 메타(Meta)가 SAM(Segment Anything Model)을 발표한 이후, 업계의 라벨링 생태계는 완전히 바뀌었습니다.

오늘은 기존 SAM보다 훨씬 빠르고 동영상까지 지원하는 최신 **SAM 2**를 Mac 환경에서 다루고, 실무용 **오토 라벨링(Auto-labeling) 파이프라인**을 직접 구축해 보겠습니다! 🚀

## 1. 프롬프트 기반 분할 (Promptable Segmentation) 이란?
ChatGPT에 "사과에 대해 설명해 줘"라고 '텍스트 프롬프트'를 주면 답이 나오듯, SAM 2는 '시각적 프롬프트'를 주면 그에 맞는 마스크(Mask)를 반환합니다.

- **점(Point) 프롬프트**: 이미지의 특정 픽셀 위치를 클릭하면, 그 점이 포함된 객체 전체를 분할합니다.
- **박스(Box) 프롬프트**: 물체가 대략적으로 있는 네모난 영역을 주면, 그 안에서 정확한 픽셀 외곽선을 따냅니다.

## 2. SAM 2 초간단 실습: 점 하나로 물체 따기
복잡한 Meta의 원본 저장소를 클론할 필요 없이, 우리가 Phase 3에서 유용하게 썼던 `ultralytics` 패키지가 최근 SAM 2까지 완벽하게 통합 지원하기 시작했습니다.

테스트할 사진(`my_image.jpg`)을 준비하고 `sam2_point.py`를 작성해 봅시다. Mac의 `MPS` 가속을 그대로 사용합니다!

```python
import cv2
from ultralytics import SAM

# 1. Mac GPU(MPS)에서 작동하는 가장 가벼운 SAM 2 모델 로드 (자동 다운로드 됨)
print("🧠 SAM 2 모델을 불러오는 중...")
model = SAM('sam2-n.pt') 

# 2. 이미지에 '점(Point)' 프롬프트 부여하기
# points: [x 좌표, y 좌표] (이미지에서 오려내고 싶은 물체의 대략적인 중앙 픽셀 위치)
# labels: 1은 전경(Foreground, 찾고 싶은 물체), 0은 배경(Background, 제외할 물체)
results = model.predict(
    "my_image.jpg", 
    points=[250, 400], 
    labels=[1], 
    device='mps'
)

# 3. 결과 시각화
annotated_img = results[0].plot()

cv2.imshow("SAM 2 Point Segmentation", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**실행 결과**:
여러분이 지정한 `(250, 400)` 픽셀 좌표에 있는 객체(예: 사람, 강아지, 자동차 등)가 무엇이든 간에, SAM 2가 스스로 윤곽선을 파악하여 깔끔한 색상 마스크를 씌워주는 것을 볼 수 있습니다. 처음 보는 물체라도 기가 막히게 분할해 냅니다(Zero-shot).

## 3. \[실무 핵심\] YOLO + SAM 2 오토 라벨링 파이프라인 만들기
시니어 개발자는 SAM 2를 단순한 누끼 따기 툴로 쓰지 않습니다.
실무에서는 "**객체 탐지(YOLO)로 박스를 먼저 찾고 ➡️ 그 박스 좌표를 SAM 2의 프롬프트로 넘겨 ➡️ 완벽한 픽셀 마스크를 자동으로 얻어내는"** 파이프라인을 구축합니다. 이를 통해 라벨링 비용을 0원으로 수렴시킬 수 있습니다.

`auto_labeling.py`를 작성하여 이 강력한 워크플로우를 직접 구현해 봅시다.

```python
from ultralytics import YOLO, SAM
import cv2

print("🚀 오토 라벨링 파이프라인을 시작합니다...")

# 1. 두 개의 모델 로드 (둘 다 mps 지원)
yolo_model = YOLO('yolov8n.pt') # 빠르고 가벼운 위치 탐지용
sam_model = SAM('sam2-n.pt')    # 정교한 픽셀 분할용

img_path = "my_image.jpg"

# 2. Step 1: YOLO로 이미지 내의 모든 객체 바운딩 박스(Bounding Box) 찾기
print("1️⃣ YOLO로 객체 위치를 탐색 중...")
yolo_results = yolo_model.predict(img_path, device='mps', verbose=False)

# 바운딩 박스 좌표 추출 (N개의 객체에 대한 [x1, y1, x2, y2] 리스트)
bboxes = yolo_results[0].boxes.xyxy.tolist()

if not bboxes:
    print("❌ 찾은 객체가 없습니다.")
else:
    print(f"✅ 총 {len(bboxes)}개의 객체를 찾았습니다! SAM 2로 넘깁니다.")
    
    # 3. Step 2: 찾은 박스들을 SAM 2의 '박스 프롬프트(bboxes)'로 전달
    # 사람이 일일이 점을 찍을 필요 없이 YOLO가 찾은 영역을 그대로 던져줍니다.
    sam_results = sam_model.predict(
        img_path, 
        bboxes=bboxes, 
        device='mps', 
        verbose=False
    )
    
    # 4. 결과 출력
    # YOLO가 찾은 네모 박스 영역을 기반으로, SAM 2가 정교하게 픽셀을 따낸 결과!
    annotated_img = sam_results[0].plot()
    
    cv2.imshow("YOLO + SAM 2 Auto Labeling", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

이 짧은 코드가 바로 수많은 AI 데이터 가공 회사들이 내부적으로 사용하는 **MLOps 오토 라벨링 툴**의 핵심 엔진입니다. 이 코드를 웹 프레임워크(Streamlit, FastAPI)에 얹기만 하면 여러분만의 멋진 AI 라벨링 SaaS 서비스가 완성됩니다!

## 마치며

오늘 우리는 파운데이션 모델의 양대 산맥 중 하나인 **SAM 2**를 다루며, 픽셀 단위 분할의 한계를 돌파했습니다. 이제 데이터 라벨링의 고통에서 벗어나, 아이디어와 파이프라인 구축에 집중할 수 있는 환경이 마련되었습니다.

Phase 1(거대 파운데이션 모델)을 성공적으로 마쳤습니다!
다음 **Step 3**에서는 Phase 2 영역으로 넘어갑니다. YOLO의 고질적인 병목이었던 박스 지우기(NMS) 과정을 아예 없애버린 **차세대 객체 탐지기**, **RT-DETR (Real-Time DEtection TRansformer)** 의 구조와 실습을 진행하겠습니다.

점점 더 흥미진진해지는 심화 비전 세계, 다음 포스팅에서 뵙겠습니다!
