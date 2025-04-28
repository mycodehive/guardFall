## 배려대상자 낙상 감지 및 인터랙티브 알림 시스템 ([Notion](https://mycodehive.notion.site/1db04465e0ff80149aa8df807b38de12))

### Step 1
  - medipipe 라이브러리 때문에 python은 3.11.0으로 맞춤
  - window : [python-3.11.0-amd64.exe](https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe)
  - Mac :macOS 11 (Big Sur) 이상 → [python-3.11.0-macos11.pkg](https://www.python.org/ftp/python/3.11.0/python-3.11.0-macos11.pkg)
  - "C:\Python311" 로 설치

### Step 2
  - uv package 라이브러리 설치
    - window : <pre>powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"</pre>
    - mac : <pre>curl -LsSf https://astral.sh/uv/install.sh | sh</pre>
  - 확인 : uv --version

### Step 3
  - visual code 설정
    - 작업 폴더를 선택해서 workspace 만들기

      ![image](https://github.com/user-attachments/assets/e54fcbfd-390d-459d-ad36-60bc62c936d0)

  - C:\guardFall>uv venv --python 3.11
       <pre>
       Using CPython 3.11.0 interpreter at: C:\Python311\python.exe
       Creating virtual environment at: .venv
       Activate with: .venv\Scripts\activate</pre>

### Step 4
  - 아래 3가지 파일의 파이썬 버전을 "3.11"로 맞춤
       <pre>
        C:\guardFall\.venv\pyvenv.cfg
        C:\\guardFall\.python-version
        C:\guardFall\pyproject.toml</pre>

### Step 5
  - uv add -r requirements.txt 로 설치
  - 단, TensorFlows는 가상환경 내에서 uv로 설치리 의존성 파일까지 설치되면서 윈도우에서 맥용 라이브러가 설치됨. 오류발생
    - window : (guardFall) C:\guardFall\pip install TensorFlow=2.17.0 으로 설치
    - mac : requirements.txt 에 TensorFlow=2.17.0 을 추가하여 uv add -r requirements.txt 처리
  - C:\guardFall\pyproject.toml 의 dependencies 항목 확인하기

### Step 6
  - 실행하기
    - streamlit 실행 : uv run -- streamlit run streamlit_app.py
    - 일반 py 실행 : uv run main.py
