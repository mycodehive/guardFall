  - YOLO, Mediapipe 학습 방법의 차이 => 두가지 모델로 앙상블 로직 구현도 가능함
  - 최초 자세에 따른 모델의 판정 이슈 => 시계열 로직 필요, 오탐지케이스 발굴로 데이터 보정 필요
  - 추가적인 비전 사용자 학습 필요 사항 => LSTM, depth anything 사용자 학습 필요


| createmodel | createmodel_lstm | createmodel_ensemble |
|------|------|------|
| ![image](https://github.com/user-attachments/assets/5810e024-23fe-4cc8-b669-127fbfc12aa4) | ![image](https://github.com/user-attachments/assets/4defea71-bf7d-4c8a-9bf6-810788cb6fa4) | ![image](https://github.com/user-attachments/assets/4198f7d0-da29-4df2-8acb-db002c33b008) |
