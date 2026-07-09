$$CV 튜토리얼$$

# Step 10. AI가 그림을 그리고 말을 하다: 생성형 AI와 VLM(시각-언어 모델)
안녕하세요! 컴퓨터 비전 튜토리얼 열 번째 시간입니다.
지금까지 우리는 이미지가 '무엇인지' 맞히고, '어디에 있는지' 찾고, '어떻게 움직이는지' 추적하는 기술을 배웠습니다. 이는 모두 이미 존재하는 이미지를 '분석'하는 기술들이었죠.

하지만 최근 1~2년 사이, 컴퓨터 비전 패러다임이 완전히 뒤집혔습니다. AI가 새로운 이미지를 창조해 내는 생성형 AI(Generative AI)와, 텍스트와 이미지를 동시에 이해하여 사진을 보고 사람처럼 대화하는 VLM(Vision-Language Model)의 시대가 열렸습니다.

오늘은 최신 트렌드에 발맞춰, Mac 환경에서 직접 이미지를 생성해 보고, 그 이미지를 묘사하는 AI를 파이썬 코드로 구현해 보겠습니다! 🚀

## 0. 필수 라이브러리 설치
가장 널리 쓰이는 허깅페이스(Hugging Face)의 라이브러리들을 설치합니다.
터미널에서 가상환경(cv_env)을 켜고 아래 명령어를 입력해 주세요.
```
pip install diffusers transformers accelerate
```

## 1. 텍스트로 이미지를 만들다: Stable Diffusion
생성형 비전 AI의 대명사인 Stable Diffusion(스테이블 디퓨전)은 노이즈(지지직거리는 화면)에서 시작해, 우리가 입력한 텍스트(Prompt) 조건에 맞게 점진적으로 노이즈를 깎아내며 선명한 그림을 만들어내는 모델입니다.

Hugging Face의 diffusers 패키지를 사용하면 아주 쉽게 Mac의 mps 가속을 활용해 이미지를 생성할 수 있습니다.
새로운 파일 text_to_image.py를 만들고 코드를 실행해 보세요!
```
import torch
from diffusers import StableDiffusionPipeline

# 1. Mac GPU(MPS) 디바이스 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# 2. Stable Diffusion 모델 로드 (v1.5 버전 사용)
# 첫 실행 시 수 GB의 모델을 다운로드하므로 시간이 조금 걸립니다!
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16 # Mac 메모리 절약을 위해 반정밀도 사용
)
pipe = pipe.to(device)

# Mac 메모리 최적화를 위한 설정 (메모리가 부족해 꺼지는 현상 방지)
pipe.enable_attention_slicing()

# 3. 프롬프트(Prompt) 작성: AI에게 그려달라고 할 텍스트
prompt = "A futuristic city under a glass dome on Mars, cyberpunk style, highly detailed, 4k"
print(f"🎨 다음 프롬프트로 이미지를 생성합니다: '{prompt}'")

# 4. 이미지 생성
# num_inference_steps: 노이즈를 깎는 횟수 (보통 30~50)
image = pipe(prompt, num_inference_steps=30).images[0]

# 5. 결과 저장 및 확인
image.save("mars_city.png")
image.show()
print("✅ 이미지 생성이 완료되어 'mars_city.png'로 저장되었습니다!")
```

**실행 결과**: 
코드 실행 후 잠시 기다리면, 여러분의 Mac GPU가 맹렬히 회전하며 화성 위의 사이버펑크 유리 돔 도시를 기가 막히게 그려냅니다!

## 2. 이미지를 읽고 설명하다: VLM (Vision-Language Model)
그렇다면 반대로, 방금 AI가 그린 사진(또는 내 폰에 있는 아무 사진)을 AI에게 보여주면 "이게 어떤 상황인지 글로 설명해 봐!"라고 할 수 있을까요?
이것이 바로 이미지와 텍스트를 하나로 묶어 처리하는 **멀티모달(Multi-modal) VLM** 기술입니다. ChatGPT의 시각 기능이 바로 이 기술을 기반으로 합니다.

여기서는 가벼우면서도 성능이 뛰어나 실무에서 이미지 자동 캡셔닝(Captioning)에 많이 쓰이는 **BLIP (Bootstrapping Language-Image Pre-training)** 모델을 사용해 보겠습니다.

`image_captioning.py` 파일을 만들어 주세요.
```
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. BLIP 모델 및 프로세서(이미지+텍스트 전처리용) 로드
print("🤖 BLIP 모델을 불러오는 중...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# 2. 분석할 이미지 불러오기 (방금 생성한 이미지 활용)
img_path = "mars_city.png"
raw_image = Image.open(img_path).convert('RGB')
print(f"📸 '{img_path}' 이미지를 분석합니다...")

# 3. 이미지를 모델이 이해할 수 있는 텐서(Tensor) 형태로 변환 후 mps로 이동
inputs = processor(raw_image, return_tensors="pt").to(device)

# 4. 이미지 캡션 생성 (추론)
# max_new_tokens: 생성할 텍스트의 최대 길이
out = model.generate(**inputs, max_new_tokens=50)

# 5. 생성된 토큰(숫자)을 사람이 읽을 수 있는 텍스트로 디코딩
caption = processor.decode(out[0], skip_special_tokens=True)

print("\n" + "="*50)
print(f"✨ AI의 이미지 설명: {caption}")
print("="*50)
```

**실행 결과 예시**:
```
✨ AI의 이미지 설명: a city on mars inside a glass dome with a red sky
```

방금 전 우리가 프롬프트로 입력했던 내용을 AI가 반대로 찰떡같이 묘사하는 것을 볼 수 있습니다. 쇼핑몰 이미지 자동 태깅, 시각 장애인을 위한 화면 해설 시스템 등이 이 기술로 만들어집니다.

## 마치며

오늘은 생성형 AI(Stable Diffusion)로 세상에 없던 이미지를 그려보고, VLM(BLIP)을 통해 이미지를 텍스트로 이해하는 최신 컴퓨터 비전의 정수를 경험해 보았습니다.

이것으로 가장 길고 깊이 있었던 **Phase 4: 심화 비전 알고리즘** 단계가 막을 내립니다! 정말 고생 많으셨습니다.

하지만 여기서 끝이 아닙니다. 실무에서는 이렇게 훌륭한 모델을 내 컴퓨터에서만 돌리는 게 아니라, **웹 서비스나 스마트폰 앱으로 배포(Deployment)** 해야만 진짜 가치가 생깁니다.
다음 **Phase 5**이자 튜토리얼의 마지막 대단원에서는 여러분이 만든 모델을 가볍게 압축하고(ONNX), Mac/iOS 전용 칩셋(CoreML)에 올린 뒤, API 서버(FastAPI)로 배포하는 진짜 '개발자'의 영역을 다루겠습니다.

마지막까지 화이팅입니다!
