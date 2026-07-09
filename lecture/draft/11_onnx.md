$$CV 튜토리얼$$

# Step 11. 실무 배포의 첫걸음: 모델 경량화와 ONNX 변환
안녕하세요! 컴퓨터 비전 튜토리얼 열한 번째 시간입니다.
드디어 이 기나긴 튜토리얼의 마지막 대단원인 **Phase 5: 실무 최적화 및 배포(Deployment & MLOps)** 에 도착하셨습니다! 🎉

지금까지 우리는 PyTorch를 이용해 강력한 딥러닝 모델들을 학습시켰습니다. 하지만 실무에서 이 모델을 실제 프로덕트(스마트폰 앱, 로봇 청소기, 웹 서비스)에 넣으려고 하면 큰 장벽에 부딪힙니다.
"PyTorch 환경을 전부 설치하기엔 용량이 너무 커요!", "Python은 너무 무겁고 느려서 C++이나 Java에서 돌려야 해요!"

그래서 우리는 무거운 파이썬/파이토치 환경을 걷어내고, **모델의 뼈대와 가중치만 쏙 뽑아내어 가볍고 빠른 범용 포맷**으로 변환해야 합니다. 그 업계 표준이 바로 **ONNX(Open Neural Network Exchange)** 입니다.

오늘은 우리가 만든 PyTorch 모델을 ONNX로 변환하고, 얼마나 속도가 빨라지는지 직접 측정해 보겠습니다! 🚀

## 1. 딥러닝계의 PDF, ONNX 란?
문서를 작성할 때 한글(.hwp)이나 워드(.docx)로 작성한 뒤, 누구에게나 깨지지 않고 보여주기 위해 최종적으로 PDF로 변환하곤 하죠?
딥러닝 모델도 마찬가지입니다. PyTorch, TensorFlow 등 어떤 프레임워크로 학습했든 간에, 이를 **ONNX(.onnx)** 라는 표준 포맷으로 변환하면 C++, C#, Java, JavaScript 등 거의 모든 언어와 환경에서 모델을 실행(Inference)할 수 있습니다.

먼저, ONNX 변환과 실행을 위한 패키지를 터미널에서 설치해 줍니다.
```
pip install onnx onnxruntime
```

## 2. PyTorch 모델을 ONNX로 변환하기 (Export)

기존에 학습해 둔 모델이 있다면 그것을 사용해도 좋고, 여기서는 실습을 위해 PyTorch에서 기본 제공하는 가벼운 이미지 분류 모델(`ResNet18`)을 불러와서 변환해 보겠습니다.

새로운 파이썬 파일 `export_onnx.py`를 만들고 아래 코드를 실행해 보세요.
```
import torch
import torchvision.models as models

print("1. PyTorch 모델을 불러옵니다...")
# 사전 학습된 ResNet18 모델 로드 및 평가 모드(eval) 설정
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# 2. 모델이 입력받을 '더미(Dummy) 데이터' 생성
# ONNX 변환기는 이 더미 데이터를 모델에 통과시키면서 연산의 흐름(그래프)을 추적합니다.
# 형태: (Batch Size, Channels, Height, Width)
dummy_input = torch.randn(1, 3, 224, 224)

print("2. ONNX 포맷으로 변환(Export) 중입니다...")
# 3. ONNX로 내보내기
torch.onnx.export(
    model,                      # 변환할 모델
    dummy_input,                # 모델의 입력값 추적을 위한 더미 데이터
    "resnet18.onnx",            # 저장될 파일 이름
    export_params=True,         # 학습된 가중치(Weight)도 함께 저장
    opset_version=11,           # ONNX 버전 (보통 11 또는 13을 많이 사용)
    do_constant_folding=True,   # 상수 폴딩 최적화 적용 (모델을 더 빠르고 가볍게 만듦)
    input_names=['input'],      # 입력 노드의 이름 지정
    output_names=['output']     # 출력 노드의 이름 지정
)

print("✅ 성공! 'resnet18.onnx' 파일이 생성되었습니다.")
```

실행 후 폴더를 확인해 보면 `resnet18.onnx` 파일이 예쁘게 생성되어 있을 것입니다. 이제 이 파일 하나만 있으면 파이토치가 없어도 인공지능을 돌릴 수 있습니다!

## 3. ONNX Runtime으로 추론 속도 비교하기
ONNX 파일은 **ONNX Runtime(ORT)** 이라는 가벼운 엔진 위에서 돌아갑니다. ORT는 불필요한 학습 기능이 다 빠져있고 철저하게 '실행(Inference)'에만 최적화되어 있어서 원본 PyTorch보다 훨씬 빠릅니다.

과연 얼마나 속도 차이가 날까요? `compare_speed.py`를 작성하여 비교해 봅시다. (공정한 비교를 위해 둘 다 CPU 환경에서 측정합니다.)
```
import torch
import torchvision.models as models
import onnxruntime as ort
import numpy as np
import time

# ----------------------------------------
# 1. 준비 작업
# ----------------------------------------
# PyTorch 모델 및 입력 데이터 (Tensor)
model_pt = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
dummy_input_pt = torch.randn(1, 3, 224, 224)

# ONNX 런타임 세션 및 입력 데이터 (NumPy Array)
# ONNX는 PyTorch Tensor 대신 순수 NumPy 배열을 받습니다!
ort_session = ort.InferenceSession("resnet18.onnx")
dummy_input_np = dummy_input_pt.numpy()

# ----------------------------------------
# 2. PyTorch 속도 측정 (100번 반복 평균)
# ----------------------------------------
print("🔥 PyTorch 추론 속도 측정 중...")
# 웜업 (초기 지연시간 무시를 위해 몇 번 미리 실행)
for _ in range(10): model_pt(dummy_input_pt) 

start_time = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = model_pt(dummy_input_pt)
pt_time = (time.time() - start_time) / 100

# ----------------------------------------
# 3. ONNX Runtime 속도 측정 (100번 반복 평균)
# ----------------------------------------
print("🚀 ONNX Runtime 추론 속도 측정 중...")
# 웜업
for _ in range(10): ort_session.run(None, {'input': dummy_input_np})

start_time = time.time()
for _ in range(100):
    # ONNX Runtime은 딕셔너리 형태로 입력값을 전달합니다.
    _ = ort_session.run(None, {'input': dummy_input_np})
onnx_time = (time.time() - start_time) / 100

# ----------------------------------------
# 4. 결과 출력
# ----------------------------------------
print("\n" + "="*40)
print(f"⏱️ PyTorch 1장 추론 시간: {pt_time * 1000:.2f} ms")
print(f"⏱️ ONNX 1장 추론 시간: {onnx_time * 1000:.2f} ms")
print(f"🎉 성능 향상: ONNX가 약 {pt_time / onnx_time:.2f}배 더 빠릅니다!")
print("="*40)
```

**실행 결과 예시**:
```
========================================
⏱️ PyTorch 1장 추론 시간: 25.40 ms
⏱️ ONNX 1장 추론 시간: 11.20 ms
🎉 성능 향상: ONNX가 약 2.27배 더 빠릅니다!
========================================
```

실제 수치는 컴퓨터 사양에 따라 다르지만, 일반적으로 ONNX Runtime이 CPU 상에서 PyTorch보다 2~3배 이상 빠른 퍼포먼스를 보여줍니다.

## 마치며

오늘은 무겁고 복잡한 연구용 코드를 가볍고 범용적인 배포용 포맷(ONNX)으로 변환하는 실무 핵심 기술을 배웠습니다. 이제 여러분은 이 ONNX 파일을 넘겨주며 프론트엔드/백엔드 개발자들과 완벽하게 협업할 수 있습니다.

하지만 우리가 사용하는 Mac에는 CPU보다 훨씬 효율적인 AI 전용 반도체(Neural Engine)가 숨어있다는 사실을 아시나요?
다음 대망의 마지막 Step 12 에서는 **Apple 생태계의 끝판왕인 CoreML 변환**으로 Mac의 NPU 성능을 100% 끌어내고, 이를 **FastAPI 웹 서버**에 올려 전 세계 어디서든 내 모델을 사용할 수 있게 서비스화하는 과정을 진행해 보겠습니다.

컴퓨터 비전 여정의 화려한 피날레, 다음 포스팅을 기대해 주세요!
