# 노인 낙상 감지 및 알림 시스템 개발 계획서

![image](https://github.com/user-attachments/assets/a21be991-f272-46e6-bbd5-7aa691472083)

1. 프로젝트 개요
>프로젝트명: 지능형 노인 낙상 감지 및 인터랙티브 알림 시스템

>목적: 1인 가구 노인의 안전을 위해, 낙상 사고를 실시간으로 감지하고 보호자에게 즉시 알림을 전송하는 시스템 개발. 카메라 기반 낙상 감지 기능에 더해, ChatGPT API를 활용한 자연어 설명, 보호자와의 대화형 알림, 대응 가이드 제공 등을 통해 보다 인간 중심적이고 스마트한 돌봄 시스템을 구현

>주요 기능:
>
>카메라를 통한 실시간 영상 분석,
>
>낙상 상황 인식 (딥러닝 기반 또는 영상처리 기반),
>
>낙상 발생 시 알림 (앱/문자/웹 대시보드/이메일 등),
>
>수집된 데이터로 낙상 예상시간 및 위험 코어타임 사전 알림,

2. 시스템 구성
>하드웨어 : 일반 웹캠 or 라즈베리파이 카메라 (Raspberry Pi + Pi Camera 고려 가능) or 여분 휴대폰
>
>소프트웨어 :
>
>Python
>
>OpenCV (영상 처리)
>
>MediaPipe / OpenPose (사람 자세 추정)
>
>TensorFlow/Keras (딥러닝 모델 선택 시)
>
>Telegram API (알림 전송)
>
>Streamlit  (웹 대시보드)

3. 시스템 플로우
>![image](https://github.com/user-attachments/assets/546b47df-e08d-4a8f-9768-a3a6ec9fbc1d)

4. 기능정의
>|기능|설명|
>|:-----:|:-----:|
>|실시간 영상 스트리밍|카메라를 통해 실시간 영상 수집|
>|자세추정|사람의 뼈대나 중심을 추정하여 자세 분석|
>|낙상 판별 알고리즘|누워 있는 자세를 일정 시간 이상 유지 시 낙상으로 판단|
>|알림 전송|보호자에게 알림 메시지 전송 (SMS/앱/웹)|
>|기록 저장|낙상 이벤트 기록 DB 저장 및 조회 기능|

5. 개발일정
> 1주차 : 요구사항 정리, OpenCV 실습, 카메라 연결 / 사람 감지 및 자세 추정 (MediaPipe/YOLO/딥러닝 모델 비교)
> 
> 2주차 : DB 설계 / 낙상 판별 알고리즘 설계 및 테스트
> 
> 3주차 : 알림 시스템 연동 (Telegram) / 웹 대시보드 또는 앱 알림 기능
> 
> 4주차 : 통합 테스트 및 시연 영상 제작 / 추가 수정 보완
> 
> 5주차 : 최종테스트 및 발표

>Gamma : [https://gamma.app/docs/-od3bunrb5l3o40z](https://gamma.app/docs/-od3bunrb5l3o40z)
>
>Google Docs : [https://docs.google.com/document/d/1BMVVvwdUbuOgPqAxZ6644Fkr5GLpo8zpEisSsnekLC4/edit?usp=sharing](https://docs.google.com/document/d/1BMVVvwdUbuOgPqAxZ6644Fkr5GLpo8zpEisSsnekLC4/edit?usp=sharing)

6. 데이터셋
>https://nhiss.nhis.or.kr/   국민건강보험공단이에요,  데이터 찾으셔서 낙상 관련 진단 코드로 고령자 낙상 데이터 추출 가능합니다. 
>https://opendata.hira.or.kr/ - 심평원 데이터 있는 곳이에요, 낙상 진료 코드로 검색, 병원 이용 현황, 연령대별 통계 확인 가능합니다.

---------------------------------------

[study]
  - medipipe
     - [https://velog.io/@givemetangerine/MediaPipe-%EC%BD%94%EB%93%9C%EB%A5%BC-%EB%9C%AF%EC%96%B4%EB%B3%B4%EC%9E%90](https://velog.io/@givemetangerine/MediaPipe-%EC%BD%94%EB%93%9C%EB%A5%BC-%EB%9C%AF%EC%96%B4%EB%B3%B4%EC%9E%90)
     - [https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/python](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/python)

