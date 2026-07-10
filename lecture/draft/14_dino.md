# [CV 심화 튜토리얼] Step 1. 시각 지능의 진화: 파운데이션 모델(VFM)과 DINOv2 마스터하기

안녕하세요! 컴퓨터 비전 심화 튜토리얼의 첫 번째 시간입니다.
지금까지 우리는 특정 목적(예: 개/고양이 분류, 안전모 탐지)을 위해 라벨링된 데이터를 모으고, CNN이나 YOLO 모델을 '처음부터' 학습시키는 방법을 배웠습니다.

하지만 2026년 현재 글로벌 빅테크 기업들의 접근 방식은 완전히 다릅니다. 사람처럼 세상의 모든 이미지를 보고 이미 '시각적 원리'를 통달한 거대한 뇌를 먼저 만든 뒤, 이를 가져다 쓰는 방식을 택합니다. 이를 **거대 시각 파운데이션 모델 (Vision Foundation Model, VFM)** 이라고 부릅니다.

오늘은 그중에서도 메타(Meta)가 공개하여 업계를 뒤흔든 **DINOv2**를 활용해, 정답(Label) 데이터 없이도 이미지의 특징을 완벽하게 뽑아내는 실무 기술을 알아보겠습니다! 🚀

## 1. DINOv2는 무엇이 특별할까? (Self-Supervised Learning)

기존 모델들은 "이것은 고양이야", "이것은 강아지야"라고 정답표(Label)를 쥐여주며 학습시켰습니다(지도 학습). 이 방식은 정답을 만드는 '라벨링' 비용이 어마어마하게 듭니다.

하지만 **DINOv2**는 **자기지도학습(Self-Supervised Learning)** 이라는 마법 같은 방식을 사용합니다.
이미지 수억 장을 던져주고 모델 스스로 "아, 이 사진의 일부를 가려도 원본은 이런 모양이겠구나", "이 두 사진은 각도만 다를 뿐 같은 물체구나"를 깨닫게 만듭니다. 그 결과, DINOv2는 이미지의 깊이, 윤곽선, 질감 등 **기하학적 특징(Feature)을 인간 수준으로 정교하게 이해**하게 되었습니다.

## 2. 실무 활용 포인트: Image Retrieval (이미지 검색/유사도)

DINOv2를 실무에서 가장 강력하게 쓸 수 있는 곳이 바로 '이미지 검색(유사도 기반 매칭)'과 'Zero-shot 분류'입니다.
수만 개의 상품 이미지가 있을 때, DINOv2를 통과시켜 나온 '특징 벡터(Feature Vector)'들의 거리를 비교하면, 비슷한 디자인의 상품을 0.1초 만에 찾아낼 수 있습니다. 학습(Training)은 단 1 에폭도 필요하지 않습니다!

## 3. DINOv2로 이미지 유사도 비교하기 (Hands-on)

백문이 불여일견이죠! 파이썬 파일 `dinov2_similarity.py`를 만들고, 비교하고 싶은 두 장의 사진(`image1.jpg`, `image2.jpg`)을 준비해 코드를 실행해 봅시다. Mac GPU(mps)를 활용합니다.
```python
import torch
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F

# 1. Mac GPU(MPS) 디바이스 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 사용 중인 디바이스: {device}")

# 2. PyTorch Hub에서 DINOv2 모델 로드 (가장 가벼운 ViT-Small 버전)
print("🧠 DINOv2 모델을 다운로드/로드 중입니다...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model = model.to(device)
model.eval() # 추론 모드

# 3. DINOv2 전용 이미지 전처리 파이프라인
# ViT 구조의 특성상 이미지를 14x14 픽셀의 패치(Patch) 단위로 쪼개기 때문에,
# 이미지 크기를 14의 배수로 맞추는 것이 좋습니다. (보통 224x224 사용)
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_feature(image_path):
    # 이미지 열기 및 전처리
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device) # 배치 차원 추가
    
    # 모델에 통과시켜 특징 벡터 추출 (기울기 계산 제외)
    with torch.no_grad():
        # DINOv2는 이미지 전체를 대표하는 [CLS] 토큰의 벡터를 반환합니다.
        feature = model(img_tensor) 
    
    return feature

# 4. 두 이미지의 특징 벡터 추출
feat1 = extract_feature('image1.jpg')
feat2 = extract_feature('image2.jpg')

# 5. 두 벡터 간의 코사인 유사도(Cosine Similarity) 계산
# 1에 가까울수록 완벽히 똑같고, 0에 가까울수록 전혀 다른 이미지입니다.
similarity = F.cosine_similarity(feat1, feat2)

print("\n" + "="*40)
print(f"📊 두 이미지의 DINOv2 유사도 점수: {similarity.item():.4f}")
print("="*40)
```

**실행 결과 및 팁**:
- 비슷한 종류의 옷이나 같은 종류의 동물을 넣으면 `0.8` 이상의 높은 유사도가 나옵니다.
- 전혀 다른 사물을 넣으면 `0.3` 이하의 낮은 점수가 나옵니다.
- 실무에서는 이 특징 벡터들을 **Faiss (Facebook AI Similarity Search)** 와 같은 벡터 DB에 저장해 두고 구글 이미지 검색처럼 빠르고 정확한 사내 검색 엔진을 만듭니다.

## 마치며

오늘은 파운데이션 모델의 개념과, 단 몇 줄의 코드만으로 세계 최고 수준의 이미지 특징 추출기를 내 로컬 Mac 환경에서 구동하는 방법을 알아보았습니다. 기존 CNN 모델로 수십 시간을 끙끙대며 학습시켰던 시절을 생각하면 엄청난 발전이죠?

다음 **Step 2**에서는 또 다른 파운데이션 모델의 끝판왕이자, 라벨링 노가다를 영원히 끝내버릴 구원자!
이미지 내의 모든 물체의 픽셀 외곽선을 클릭 한 번으로 따주는 **SAM 2 (Segment Anything Model 2)** 에 대해 실습해 보겠습니다. 다음 포스팅에서 만나요!
