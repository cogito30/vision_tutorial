$$CV 튜토리얼$$

# Step 12. AI 모델의 완성: CoreML 최적화 및 FastAPI 배포
안녕하세요! 드디어 컴퓨터 비전 튜토리얼의 마지막 대단원, 열두 번째 시간입니다!
길고 길었던 이 로드맵의 종착지인 Phase 5: 실무 최적화 및 배포의 하이라이트에 오신 것을 진심으로 환영합니다. 🎉

지난 Step 11에서는 모델을 빠르고 가볍게 만드는 표준 포맷 'ONNX'에 대해 배웠습니다.
오늘은 여기서 한 걸음 더 나아가, **1) Apple 기기(Mac, iPhone)의 NPU 성능을 100% 끌어내는 CoreML 포맷으로 변환**해 보고, **2) 웹 프론트엔드나 모바일 앱과 연동할 수 있도록 FastAPI를 이용해 나만의 AI 서버를 구축**해 보겠습니다.

자, 이제 내 컴퓨터 안에서만 돌던 AI를 세상 밖으로 꺼내볼까요? 🚀

## 0. 필수 패키지 설치
이번 실습을 위해 Apple의 머신러닝 변환 툴인 coremltools와, 빠르고 모던한 파이썬 웹 프레임워크인 fastapi, 그리고 서버 구동을 위한 uvicorn을 설치합니다.
```
pip install coremltools fastapi uvicorn python-multipart
```

## 1. Apple 생태계의 끝판왕, CoreML 변환
Mac(M1, M2 등)이나 iPhone에는 CPU, GPU 외에도 인공지능 연산만을 전문으로 처리하는 **Neural Engine (NPU)** 이라는 강력한 하드웨어가 탑재되어 있습니다.
우리가 만든 PyTorch 모델을 `.mlpackage` (CoreML 포맷)으로 변환하면, 운영체제가 알아서 이 Neural Engine을 활용해 배터리 소모는 최소화하면서 엄청난 속도로 추론을 수행합니다. iOS 앱 개발자들과 협업할 때 반드시 필요한 포맷이기도 합니다.

새로운 파이썬 파일 `export_coreml.py`를 작성해 보세요.
```
import torch
import torchvision.models as models
import coremltools as ct

print("1. PyTorch 모델을 불러옵니다...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()

# CoreML 변환을 위해서는 먼저 모델을 TorchScript(JIT) 형태로 추적(Trace)해야 합니다.
example_input = torch.rand(1, 3, 224, 224) 
traced_model = torch.jit.trace(model, example_input)

print("2. CoreML 포맷으로 변환 중입니다...")
# 이미지가 입력으로 들어온다는 것을 CoreML에 명시적으로 알려줍니다.
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)]
)

# 모델 저장 (.mlpackage 형식)
mlmodel.save("ResNet18.mlpackage")
print("✅ 성공! 'ResNet18.mlpackage'가 생성되었습니다.")
```

이 코드를 실행하고 생성된 `ResNet18.mlpackage` 파일을 Mac의 Finder에서 더블 클릭해 보세요! Xcode가 열리면서 모델의 입력/출력 정보와 함께, 심지어 코드 한 줄 없이도 드래그 앤 드롭으로 이미지를 넣어 테스트해 볼 수 있는 화면이 나타날 것입니다.

## 2. FastAPI로 나만의 AI 서빙(Serving) 서버 구축하기
앱 개발자나 웹 프론트엔드 개발자가 우리 AI 모델을 사용하려면 어떻게 해야 할까요?
가장 표준적인 방법은 우리가 "이미지를 보내주면, AI가 분석해서 결과를 텍스트(JSON)로 응답해 주는 API 서버"를 만들어 제공하는 것입니다.

파이썬에서 이를 가장 쉽고 빠르게 만들어주는 프레임워크가 바로 FastAPI입니다. 이전 단계에서 만들었던 `resnet18.onnx` 파일을 이용해 초간단 AI 서버를 띄워보겠습니다.

`main.py` 파일을 생성하고 아래 코드를 작성합니다.
```
import io
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File
from PIL import Image

# FastAPI 앱 생성
app = FastAPI(title="My Computer Vision API")

# ONNX 모델 로드 (서버가 켜질 때 한 번만 로드하여 메모리에 올려둠)
ort_session = ort.InferenceSession("resnet18.onnx")

# 이미지 전처리 함수
def preprocess_image(image_bytes):
    # 1. 바이트 데이터를 이미지로 열기
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # 2. ResNet 입력 크기에 맞게 리사이즈
    img = img.resize((224, 224))
    # 3. NumPy 배열 변환 및 정규화
    img_np = np.array(img).astype(np.float32) / 255.0
    # 4. HWC(높이,너비,채널) -> CHW(채널,높이,너비) 순서 변경
    img_np = np.transpose(img_np, (2, 0, 1))
    # 5. 배치 차원 추가 (1, 3, 224, 224)
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

# POST 방식의 예측(Predict) API 엔드포인트 생성
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # 1. 클라이언트가 업로드한 이미지 읽기
    image_bytes = await file.read()
    
    # 2. 전처리
    input_tensor = preprocess_image(image_bytes)
    
    # 3. ONNX 모델 추론
    outputs = ort_session.run(None, {'input': input_tensor})
    predictions = outputs[0]
    
    # 4. 가장 확률이 높은 클래스 번호 찾기
    predicted_class = int(np.argmax(predictions))
    
    # 5. JSON 형태로 결과 반환
    return {
        "filename": file.filename, 
        "predicted_class_id": predicted_class,
        "message": "AI 분석이 완료되었습니다!"
    }
```

**서버 실행 및 테스트**
터미널을 열고 `main.py` 파일이 있는 곳에서 아래 명령어를 입력하여 서버를 켭니다.
```
uvicorn main:app --reload
```

`Uvicorn running on http://127.0.0.1:8000` 이라는 메시지가 뜬다면 서버가 성공적으로 켜진 것입니다!

이제 인터넷 브라우저를 열고 `http://127.0.0.1:8000/docs` 에 접속해 보세요.
FastAPI가 자동으로 만들어준 멋진 API 테스트 UI(Swagger)가 나타납니다. 여기서 `/predict` API를 클릭하고 `Try it out` 버튼을 눌러 아무 사진이나 업로드해 보세요. 여러분의 AI가 사진을 분석하고 즉각적으로 결과를 반환해 줄 것입니다!

## 🎓 대단원의 막을 내리며

여러분, 정말 수고 많으셨습니다! 👏

Mac OS 환경 세팅으로 시작하여 OpenCV 영상 처리(Phase 1), PyTorch 딥러닝 기초(Phase 2), YOLO와 U-Net을 활용한 탐지/분할(Phase 3), 생성형 AI와 추적 알고리즘(Phase 4)을 거쳐, 오늘 CoreML과 API 서버 배포(Phase 5)까지 도달했습니다.

이 12단계의 로드맵은 단순한 이론이 아니라, **현업 컴퓨터 비전 엔지니어들이 매일 숨 쉬듯 사용하는 기술의 정수**를 담았습니다. 이제 여러분은 빈 폴더에서 시작해 딥러닝 모델을 설계하고, 웹 서비스로 배포까지 할 수 있는 **완성형 AI 개발자**의 역량을 갖추게 되셨습니다.

앞으로 자신만의 멋진 커스텀 데이터셋으로 세상을 놀라게 할 AI 프로덕트를 만들어 나가시길 응원합니다.
지금까지 긴 튜토리얼을 함께 해주셔서 감사합니다. 여러분의 무한한 성장을 기대합니다! 🚀
