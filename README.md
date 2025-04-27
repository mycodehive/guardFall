🛡️ 배려대상자(노인, 아이 등) 낙상 감지 및 알림 시스템 소개

![image](https://github.com/user-attachments/assets/a21be991-f272-46e6-bbd5-7aa691472083)

### 📋 프로젝트 개요
- 배려대상자의 **낙상 사고**는 골절, 후유증, 심각할 경우 사망으로 이어질 수 있어 **빠른 감지와 대응**이 중요합니다.
- 본 시스템은 **웹캠**을 이용해, 보호가 필요한 분들의 **움직임**을 실시간으로 감지하고,**낙상 여부를 판단**하고, 필요 시 **텔레그램 알림**을 보내는 것을 목표로 합니다.

### 📚 사용 기술
- **Python** : 3.11.0 (이하 라이브러리들은 해당 3.11.0 버전에 맞춰서 설치함.)
- **OpenCV** (영상 처리) : 4.11.0.86
- **MediaPipe** (포즈 인식) : 0.10.21
- **TensorFlow** (딥러닝 모델) : 2.17.0
- **Streamlit** (웹 UI) : 1.27.0
- **python-telegram-bot** (알림 전송) : 22.0

### 🏗️ 시스템 구조 (개요)
```plaintext
카메라 (웹캠/휴대폰) → 관절 추출 (MediaPipe) → 상태 분석 (딥러닝 모델) → 낙상 판단 → 알림 (Telegram) 발송
```
![image](https://github.com/user-attachments/assets/546b47df-e08d-4a8f-9768-a3a6ec9fbc1d)   

---

# 📖 메뉴 구성 및 설명

## 1. 홈 (Home)

### 📝 설명
- 시스템의 **개요**와 **프로젝트 목표**를 안내합니다.
- 시스템 사용 방법 및 주의사항 등을 간략히 소개합니다.
---

## 2. 영상 분석 (Video Analysis)

### 📝 설명
- 업로드된 영상 파일(MP4)로 사람의 포즈(관절)를 인식하고, 실시간으로 좌표값 추출 및 낙상 여부 판단을 수행합니다.
- 추출한 좌표값으로 낙상 라벨링을 하여 분석된 결과를 **CSV 파일**로 저장할 수 있습니다.
- 라벨링은 학습한 결과와 비교하여 꾸준히 업데이트 됩니다.
---

## 3. 모델 생성 (Model Training)

### 📝 설명
- 수집된 관절 좌표 데이터(CSV)를 기반으로 딥러닝 모델을 훈련하여 낙상 여부를 예측할 수 있는 모델을 생성합니다.
- 학습이 끝나면 `.keras` 모델과 `.pkl` 모델 파일이 저장됩니다.
```plaintext
[📦 .keras 파일]
 - "훈련 끝난 신경망 모델"을 담은 딥러닝 모델 파일
 - 특히 TensorFlow나 Keras 프레임워크에서 만든 학습된 모델을 저장
 - 파일 내부 구조 : 모델의 구조 (레이어들), 모델의 가중치(weight) 값
         
[📦 .pkl 파일]
 - 파이썬 안에서 만든 어떤 물건을 그대로 파일로 저장하는 파이썬 객체를 저장하는 파일
 - 👉 관절 좌표 데이터를 학습할 때, 숫자 범위를 정규화(normalization)하는 스케일러를 저장하는 데 주로 사용
 - 파일 내부 구조 : 데이터셋, 모델, 스케일러(정규화기) 
```
---

## 4. 실시간 감시 (Real-time Monitoring)

### 📝 설명
- 실제 서비스용 모드.
- 웹캠으로 실시간 모니터링하여 낙상 위험이 감지되면,텔레그램 알림을 전송합니다.
---

# 📦 개발 파일 구조

```plaintext
/guardFall
├── streamlit_app.py         # 메인 앱 (메뉴 연결)
├── script/
│    ├── fallpredict.py      # 낙상여부 기준 사용자 파일
│    └── util.py             # 유용한 사용자 함수
├── page/
│    ├── home.py             # 랜딩페이지
│    ├── analyzermov.py      # 영상 분석 (Video Analysis)
│    ├── createmodel.py      # 모델 생성 (Model Training)
│    ├── motoring.py         # 실시간 감시 (Real-time Monitoring)
│    └── link.py             # 각종 링크안내 페이지
├── user/
│    ├── model/
│    │    ├── fall_model.h5  # 학습된 모델
│    │    └── scaler.pkl     # 파이썬 객체 저장장
│    ├── csv/                # 영상 분석 csv 파일 디렉토리리
│    └── mov/                # 업로드된 영상 파일 디렉토리리
├── db/
│    └── guardfall.db        # 좌표값 저장하는 db
├── telegram_config.py       # 텔레그램 봇 설정
└── requirements.txt         # 설치 패키지 목록(uv add -r .\\requirements.txt (pip install tensorflow==2.17.0 만 따로 인스톨))
```
