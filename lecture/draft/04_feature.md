$$CV 튜토리얼$$

# Step 4. 두 이미지를 하나로! 특징점 추출과 파노라마 매칭

안녕하세요! 컴퓨터 비전 튜토리얼 네 번째 시간입니다. 드디어 길고 길었던 **Phase 1(영상 처리 기초)의 마지막 단계**에 도착하셨습니다! 🎉

오늘은 이미지에서 변하지 않는 고유한 '특징(Feature)'을 찾아내고, 다른 이미지에서 같은 특징을 찾아 서로 짝지어주는(Matching) 기술을 배웁니다.
이 기술은 스마트폰의 **파노라마 사진 촬영, 영수증 스캔 시 찌그러진 문서 펴기, 증강현실(AR) 마커 인식** 등 실무에서 어마어마하게 활용되는 꿀 기술입니다.

## 1. 특징점(Keypoint)이란 무엇일까?

어떤 사진을 회전시키거나, 크기를 줄이거나, 조금 어둡게 만들어도 변하지 않고 "아, 이건 그 부분이구나!" 하고 알아볼 수 있는 점들을 **특징점(Keypoint)** 이라고 합니다. 주로 모서리(Corner)나 질감이 뚜렷한 부분이 특징점이 됩니다.

과거에는 SIFT, SURF라는 알고리즘이 유명했지만 연산량이 많고 특허 문제가 있었습니다. 그래서 우리는 OpenCV에서 무료로 제공하면서도 엄청나게 빠른 **ORB (Oriented FAST and Rotated BRIEF)** 알고리즘을 사용할 것입니다.

#### 💻 실습 1: ORB를 이용한 특징점 매칭

특정 책의 표지 이미지(`book_template.jpg`)와, 그 책이 책상 위에 올려져 있는 사진(`book_scene.jpg`) 두 장을 준비한 뒤, 같은 폴더에 넣고 아래 코드를 실행해 보세요.
```
import cv2

# 1. 이미지 흑백으로 불러오기 (특징점 추출은 보통 흑백에서 수행합니다)
img1 = cv2.imread('book_template.jpg', cv2.IMREAD_GRAYSCALE) # 찾고 싶은 원본 타겟
img2 = cv2.imread('book_scene.jpg', cv2.IMREAD_GRAYSCALE)    # 타겟이 포함된 실제 배경

if img1 is None or img2 is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

# 2. ORB 생성자 초기화
orb = cv2.ORB_create()

# 3. 특징점(Keypoints)과 디스크립터(Descriptors) 계산
# 디스크립터는 각 특징점의 '지문' 같은 역할을 합니다.
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 4. 특징점 매칭 (Brute-Force Matcher 사용)
# 해밍 거리(Hamming distance)를 기준으로 두 디스크립터 간의 유사도를 비교합니다.
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 5. 매칭 결과 정렬 (거리가 짧을수록 유사도가 높음)
matches = sorted(matches, key=lambda x: x.distance)

# 6. 상위 50개의 매칭 결과만 그리기
result_img = cv2.drawMatches(
    img1, kp1, 
    img2, kp2, 
    matches[:50], None, 
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 화면에 출력
cv2.imshow('ORB Feature Matching', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

- **실행 결과**: 원본 책 표지의 점들과 배경 사진 속 책의 점들이 알록달록한 선으로 연결되는 마법 같은 결과를 보실 수 있습니다!

## 2. 호모그래피(Homography)와 파노라마 스티칭
특징점들을 매칭했다면, 이 정보들을 가지고 무엇을 할 수 있을까요?
가장 대표적인 것이 바로 **이미지 스티칭(Image Stitching)**, 즉 파노라마 만들기입니다.

왼쪽을 찍은 사진과 오른쪽을 찍은 사진이 있을 때, 두 사진에서 겹치는 부분의 특징점들을 찾습니다. 그리고 한 사진을 다른 사진의 시점으로 변환하기 위해 공간을 찌그러뜨리고 펴는 변환 행렬을 구하는데, 이를 **호모그래피(Homography)** 라고 합니다.

#### 💻 실습 2: 두 장의 사진을 이어 붙여 파노라마 만들기
스마트폰으로 제자리에서 살짝 돌며 찍은 겹치는 사진 두 장(`left.jpg`, `right.jpg`)을 준비해 보세요.
```
import cv2
import numpy as np

# 1. 이미지 로드
img_left = cv2.imread('left.jpg')
img_right = cv2.imread('right.jpg')
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# 2. 특징점 추출 및 매칭 (실습 1과 동일하지만, SIFT를 사용해 더 정밀하게 해봅니다)
sift = cv2.SIFT_create() # 최근 OpenCV 버전에서는 SIFT 특허가 만료되어 무료 사용 가능!
kp1, des1 = sift.detectAndCompute(gray_right, None)
kp2, des2 = sift.detectAndCompute(gray_left, None)

# FLANN 매처 사용 (데이터가 많을 때 BF보다 빠름)
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# 좋은 매칭점만 선별 (Lowe's ratio test)
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 3. 호모그래피 계산 및 이미지 덮어씌우기
if len(good_matches) > 10:
    # 매칭된 점들의 픽셀 좌표 추출
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC 알고리즘을 이용해 오차를 걸러내고 호모그래피 행렬(M) 계산
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # right 이미지를 호모그래피 행렬에 따라 원근 변환(Warping)
    # 너비는 두 이미지 너비의 합, 높이는 left 이미지 높이로 캔버스 크기 지정
    h, w = img_left.shape[:2]
    panorama = cv2.warpPerspective(img_right, M, (w * 2, h))
    
    # 변환된 넓은 캔버스 좌측에 left 이미지를 덮어씌움
    panorama[0:h, 0:w] = img_left

    # 검은 여백을 대략적으로 잘라내기 (실무에서는 더 정교하게 처리합니다)
    panorama_cropped = panorama[:, :w + int(w * 0.4)]

    cv2.imshow('Panorama Stitching', panorama_cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("매칭점이 충분하지 않아 파노라마를 만들 수 없습니다.")
```

💡 **설명**: 이 코드는 오른쪽 사진(`img_right`)을 왜곡시켜서 왼쪽 사진(`img_left`)의 시점에 맞게 늘려 캔버스에 붙이고, 그 위에 왼쪽 사진을 덮어씌우는 방식으로 파노라마를 완성합니다.

## 🎯 Phase 1 수료를 축하합니다!

이것으로 이미지를 다루는 가장 근본적인 기술, **Phase 1(개발 환경 구축 및 이미지 처리 기초)** 을 모두 마쳤습니다!
지금까지 배운 OpenCV와 NumPy 스킬은 앞으로 딥러닝을 다룰 때 가장 강력한 무기가 될 것입니다.

다음 **Step 5**부터는 드디어 대망의 **Phase 2: 딥러닝 기반 비전 기초**로 진입합니다.
PyTorch를 이용해 처음부터 인공지능 신경망(CNN)을 설계하고, 개와 고양이를 분류하는 모델을 직접 학습시켜 보겠습니다. 다음 포스팅에서 진짜 "AI 개발"을 시작해 봅시다! 🚀
