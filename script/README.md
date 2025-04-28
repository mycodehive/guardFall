## ✅ OpenCV + MediaPipe를 사용해서 실시간으로 관절 움직임(좌표)을 추출

### 1단계: MediaPipe란 무엇인가?
  - 📌 MediaPipe란?
  - 구글에서 만든 멀티모달 머신러닝 프레임워크
  - 특히 실시간 영상에서 사람의 관절(Pose), 손(Hands), 얼굴(Face) 등을 감지
  - 내부적으로는 딥러닝 모델이 적용되어 있어서 정확도도 높음

  - ✨ 왜 좋은가?
    - OpenCV처럼 Python에서 쉽게 쓸 수 있고, CPU에서도 빠르게 작동함! (GPU 없어도 실시간 가능!)
    - 포즈 감지, 얼굴 랜드마크, 손 추적 등 다양한 모듈 제공

### 3단계: 전체 구조 미리 보기
  - 웹캠 (cv2.VideoCapture)
  - MediaPipe로 관절(포즈)을 감지
  - 각 관절 위치(x, y, z)를 실시간으로 추출해서 출력

### 4단계: MediaPipe 공부하기
   ![image](https://github.com/user-attachments/assets/4963e292-b53a-4bea-8fec-020056d08178)
  - Pose landmark detection : [Guide](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker?hl=ko)
  - python Version : [Guide](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python?hl=ko)


### 5단계: 결과확인
  - 화면에 사람의 관절 연결선이 표시됨 (MediaPipe에서 자동으로 그림)
  - 터미널에 **실시간 관절 위치 좌표(x, y, z)**가 출력됨
  - 좌표는 0~1 사이 값 (이미지 크기 기준 상대 좌표)
  - x는 왼쪽 → 오른쪽, y는 위쪽 → 아래쪽, z는 카메라와의 거리
