$$CV 튜토리얼$$

# Step 9. 움직임을 지배하다: 객체 추적(Tracking)과 포즈 추정(Pose Estimation)

안녕하세요! 컴퓨터 비전 튜토리얼 아홉 번째 시간입니다.
오늘부터는 남들과 차별화되는 경쟁력을 갖출 수 있는 **Phase 4: 심화 비전 알고리즘** 단계로 진입합니다. 🎉

지금까지 우리는 '한 장의 사진' 안에서 물체를 찾고 분할하는 방법을 배웠습니다. 하지만 현실 세계는 동영상(Video)처럼 연속된 프레임으로 이루어져 있습니다.
만약 CCTV에서 "저기 지나가는 빨간 옷 입은 사람이 어제 그 사람인가?"를 알기 위해서는, 매 프레임마다 물체를 찾는 것을 넘어 **이전 화면의 물체와 지금 화면의 물체가 동일한 것인지 이어주는(ID 부여) 기술**이 필요합니다. 이를 객체 추적(Object Tracking)이라고 합니다.

또한, 사람이 어떤 행동을 하고 있는지(예: 스쿼트 자세, 쓰러짐 감지 등)를 알기 위해 관절의 위치를 뽑아내는 **포즈 추정(Pose Estimation)** 기술도 함께 알아보겠습니다!

## 1. 객체 추적 (Object Tracking): 너의 이름은 (ID)
객체 추적은 쉽게 말해 **"객체 탐지(Detection) + 매칭(Matching)"** 입니다.
프레임 1에서 찾은 사람과 프레임 2에서 찾은 사람이 같은 사람인지 판단하기 위해 이동 방향, 속도, 생김새 등을 비교합니다. 전통적으로는 **DeepSORT**, 최근에는 **ByteTrack**이나 **BoT-SORT** 같은 알고리즘이 실무에서 널리 쓰입니다.

과거에는 YOLO로 객체를 찾고, 그 결과를 DeepSORT 코드에 넣어주는 복잡한 연동 작업이 필요했습니다. 하지만 최신 `Ultralytics YOLO` 패키지는 이 강력한 추적 알고리즘들을 아예 내장(Built-in)하고 있습니다!

#### 💻 실습 1: 실시간 다중 객체 추적기 만들기
새로운 파일 `realtime_tracking.py`를 만들고 코드를 실행해 보세요.
이전 객체 탐지 코드와 거의 똑같지만, 딱 하나의 함수(`predict` -> `track`)만 바뀝니다.

```
import cv2
from ultralytics import YOLO

# 1. 모델 로드 (가장 가벼운 nano 모델 사용)
model = YOLO('yolov8n.pt')

# 2. 웹캠 연결
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("웹캠 실시간 객체 추적을 시작합니다. 종료하려면 'q'를 누르세요.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Mac 웹캠 거울 모드
    frame = cv2.flip(frame, 1)

    # 3. 모델 추론 (Tracking)
    # 💡 핵심: 기존에는 model(frame)을 썼지만, 추적을 위해서는 model.track()을 사용합니다.
    # persist=True : 이전 프레임의 객체 ID를 기억하라는 뜻입니다.
    # tracker='bytetrack.yaml' : ByteTrack 알고리즘을 사용합니다. (기본값은 botsort)
    results = model.track(frame, persist=True, device='mps', tracker='bytetrack.yaml', verbose=False)

    # 4. 결과 시각화
    annotated_frame = results[0].plot()

    # 화면 출력
    cv2.imshow("YOLO Object Tracking (ByteTrack)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**실행 결과**: 
웹캠을 켜고 화면 안에서 움직여 보세요! 머리 위에 단순히 'person'이라는 이름뿐만 아니라 `person 1`, `person 2` 처럼 고유한 ID(숫자)가 부여되어 따라다니는 것을 볼 수 있습니다. 잠깐 화면 밖으로 나갔다 들어와도 (설정에 따라) 같은 사람으로 인식하기도 합니다.
이 기술이 바로 매장 내 방문객 카운팅, 고속도로 차량 속도 측정 등에 쓰이는 핵심 원리입니다.

## 2. 포즈 추정 (Pose Estimation): 디지털 스켈레톤 추출
포즈 추정은 이미지 속 사람을 찾아내고, 그 사람의 주요 관절(눈, 코, 어깨, 팔꿈치, 무릎 등 보통 17개 포인트) 위치(x, y 좌표)를 정확하게 짚어내는 기술입니다. 스마트 피트니스 앱, 무인 매장 행동 분석, 버추얼 유튜버(VTuber)의 모션 캡처 등이 모두 이 기술을 기반으로 합니다.

#### 💻 실습 2: 실시간 포즈 추정 및 관절 좌표 가져오기
이 역시 YOLO의 포즈 전용 모델(`-pose.pt`)을 사용하면 아주 쉽게 구현할 수 있습니다.
이번 실습에서는 단순히 뼈대를 그리는 것을 넘어, **실제 관절의 x, y 좌표값을 코드상에서 어떻게 가져오는지(추출하는지)** 알아보겠습니다. 이게 있어야 나중에 "팔꿈치 각도" 같은 것을 계산할 수 있으니까요!

`realtime_pose.py`를 작성해 보세요.
```
import cv2
import math
from ultralytics import YOLO

# 1. 모델 로드: 포즈 전용 모델 (yolov8n-pose.pt)
model = YOLO('yolov8n-pose.pt')

cap = cv2.VideoCapture(0)
print("웹캠 실시간 포즈 추정을 시작합니다. 종료하려면 'q'를 누르세요.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)

    # 2. 포즈 추정 모델 실행
    results = model(frame, device='mps', verbose=False)

    # 3. 자동 시각화 (뼈대 그리기)
    annotated_frame = results[0].plot()

    # 4. 💡 실무 팁: 특정 관절 좌표 데이터 추출하기
    # results[0].keypoints 에 17개의 관절(x, y, 신뢰도) 정보가 들어있습니다.
    if results[0].keypoints is not None:
        # 첫 번째 감지된 사람의 관절 정보 가져오기
        keypoints = results[0].keypoints.xy[0] 
        
        if len(keypoints) > 10:
            # 관절 인덱스 9번(왼쪽 손목)과 10번(오른쪽 손목)의 좌표
            left_wrist = keypoints[9]
            right_wrist = keypoints[10]

            # 두 손목의 좌표가 정상적으로 추출되었다면(0이 아니라면)
            if left_wrist[0].item() != 0 and right_wrist[0].item() != 0:
                lx, ly = int(left_wrist[0].item()), int(left_wrist[1].item())
                rx, ry = int(right_wrist[0].item()), int(right_wrist[1].item())
                
                # 손목 위치에 강조하는 큰 원 그리기
                cv2.circle(annotated_frame, (lx, ly), 10, (0, 255, 0), -1) # 왼쪽 초록색
                cv2.circle(annotated_frame, (rx, ry), 10, (0, 0, 255), -1) # 오른쪽 빨간색
                
                # 두 손목 사이의 거리 계산 예시 (이런 식으로 활용합니다!)
                distance = math.sqrt((lx - rx)**2 + (ly - ry)**2)
                cv2.putText(annotated_frame, f"Wrists Dist: {int(distance)}", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # 화면 출력
    cv2.imshow("YOLO Pose Estimation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**실행 결과**: 
화면에 여러분의 몸을 비춰보세요! 눈, 코, 입부터 어깨, 팔꿈치, 손목이 선으로 연결된 스켈레톤이 그려집니다. 그리고 우리가 추가한 코드 덕분에 양쪽 손목에는 특별히 크고 색깔 있는 점이 찍히며, 두 손목 사이의 거리가 화면 좌측 상단에 실시간으로 표시될 것입니다.

여러분은 방금 피트니스 앱을 개발하기 위한 핵심 엔진을 완성하셨습니다!

## 마치며

오늘은 단순한 이미지 분석을 넘어, 시간의 흐름을 추적하는 객체 추적(ByteTrack)과 인체의 움직임을 수치화하는 **포즈 추정(Pose Estimation)** 기술을 실습해 보았습니다.
이 단계까지 오셨다면, 이제 여러분은 머릿속에 있는 웬만한 비전 AI 아이디어(예: 매장 도둑 감지, 스마트 헬스장 시스템, 횡단보도 보행자 분석 등)를 대부분 프로토타입으로 만들어낼 수 있는 능력을 갖추셨습니다.

다음 **Step 10**에서는 최근 AI 업계를 뒤흔들고 있는 **생성형 AI(Generative AI)**와 비전의 만남을 다룹니다. 텍스트를 입력해 이미지를 만들어내고(Stable Diffusion), 이미지를 보고 AI가 상황을 텍스트로 설명해 주는(Vision-Language Model) 최신 트렌드를 직접 구현해 보겠습니다. 다음 포스팅도 많은 기대 부탁드립니다!
