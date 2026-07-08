$$CV 튜토리얼$$

# Step 1. 완벽한 Mac OS 비전 개발 환경 세팅 (Apple Silicon)

안녕하세요! 컴퓨터 비전 전문 개발자로 가는 첫걸음, 대망의 Step 1입니다.
오늘은 Apple Silicon(M1/M2/M3/M4 등)이 탑재된 Mac OS에서 컴퓨터 비전 개발을 위한 최적의 Python 환경을 구축해 보겠습니다.

과거에는 딥러닝과 비전 개발을 위해 값비싼 NVIDIA GPU(CUDA)가 필수였지만, 이제는 Mac의 GPU를 활용하는 **MPS(Metal Performance Shaders)** 덕분에 Mac 환경에서도 충분히 쾌적한 딥러닝 비전 모델 학습과 추론이 가능해졌습니다.

그럼, 가장 깔끔하고 충돌 없는 개발 환경을 함께 세팅해 봅시다! 🚀

## 1. 패키지 관리자 설치: 왜 Miniforge인가?

Python 환경을 격리하고 패키지 충돌을 막기 위해 가상환경(Virtual Environment)을 사용하는 것은 필수입니다. Mac(ARM64 아키텍처)에서는 Anaconda 대신 Miniforge를 사용하는 것을 강력히 추천합니다.
**Miniforge**는 기본 채널이 `conda-forge`로 설정되어 있어, Apple Silicon에 최적화된 패키지를 가장 빠르고 안정적으로 다운로드할 수 있습니다.

**설치 방법 (Homebrew 사용 시)**
터미널(Terminal)을 열고 아래 명령어를 입력하세요.
```
brew install miniforge
```
설치가 완료되면 쉘(Shell)을 초기화하여 `conda` 명령어를 활성화합니다.
```
conda init zsh
```
터미널을 껐다가 다시 켜면 환경 세팅이 적용됩니다.

## 2. 컴퓨터 비전용 가상환경 생성
이제 컴퓨터 비전 프로젝트 전용 가상환경을 만듭니다. 이름은 `cv_env`로 하고, Python 버전은 호환성이 가장 좋은 `3.10`으로 설정하겠습니다.
```
# 가상환경 생성
conda create -n cv_env python=3.10 -y

# 가상환경 활성화
conda activate cv_env
```

터미널 프롬프트 앞에 (`cv_env`)가 생겼다면 성공적으로 가상환경에 들어온 것입니다.

## 3. 핵심 비전 라이브러리 설치 (OpenCV & NumPy)

영상 처리의 뼈대가 되는 `OpenCV`와 행렬 연산의 핵심인 `NumPy`를 설치합니다.
여기서 팁! `opencv-python` 대신 `opencv-contrib-python`을 설치하면, 특허가 풀린 유용한 추가 비전 알고리즘(SIFT 등)까지 한 번에 사용할 수 있습니다.
```
pip install opencv-contrib-python numpy matplotlib
```
(참고: 시각화를 위해 matplotlib도 함께 설치합니다.)

## 4. PyTorch 설치와 Mac GPU(MPS) 세팅

가장 중요한 딥러닝 프레임워크인 PyTorch를 설치할 차례입니다.
Apple Silicon Mac에서는 이제 복잡한 설정 없이 기본 PyTorch만 설치해도 자동으로 MPS 가속을 지원합니다.
```
pip install torch torchvision torchaudio
```

## 5. 환경 구축 성공 여부 테스트 (가장 중요!)
설치한 패키지들이 정상적으로 동작하는지, 특히 Mac의 GPU를 PyTorch가 제대로 인식하는지 확인해야 합니다.
터미널에서 `test_env.py` 파일을 하나 생성하고 아래 코드를 복사해서 실행해 보세요.
```
# test_env.py
import sys
import cv2
import numpy as np
import torch

print("="*40)
print(f"🐍 Python 버전: {sys.version.split(' ')[0]}")
print(f"👁️ OpenCV 버전: {cv2.__version__}")
print(f"🔢 NumPy 버전: {np.__version__}")
print(f"🔥 PyTorch 버전: {torch.__version__}")
print("="*40)

# Mac GPU(MPS) 활성화 확인 로직
if torch.backends.mps.is_available():
    print("✅ 성공: Mac GPU(MPS)가 활성화되었습니다! 🚀")
    
    # 간단한 텐서 연산을 MPS 디바이스에서 실행해보기
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(f"👉 MPS 텐서 테스트: {x}")
else:
    print("❌ 실패: MPS를 찾을 수 없습니다. (CPU 모드로 동작합니다)")
print("="*40)
```

**실행 방법:**
```
python test_env.py
```

**성공적인 출력 화면 예시**:
```
========================================
🐍 Python 버전: 3.10.x
👁️ OpenCV 버전: 4.x.x
🔢 NumPy 버전: 1.x.x
🔥 PyTorch 버전: 2.x.x
========================================
✅ 성공: Mac GPU(MPS)가 활성화되었습니다! 🚀
👉 MPS 텐서 테스트: tensor([1.], device='mps:0')
========================================
```


## 마치며

축하합니다! 🎉 이제 Mac에서 이미지를 조작하고 딥러닝 모델을 쌩쌩 돌릴 수 있는 모든 준비가 끝났습니다.
다음 Step 2에서는 오늘 설치한 OpenCV와 NumPy를 활용하여 실제로 웹캠 영상을 띄우고, 이미지를 변환하는 영상 처리 기초 실습을 진행해 보겠습니다.

다음 포스팅에서 만나요!
