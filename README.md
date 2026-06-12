# DeCafe Subway: 서울 지하철 실시간 관제 및 AI 혼잡도 예측 대시보드

이 저장소는 서울 지하철 2호선 및 1~8호선 전 노선의 실시간 열차 위치 관제와 기상 정보 연동형 인공지능(AI) 승하차 혼잡도 예측 대시보드 프로젝트를 제공합니다.

---

## 기술 스택 (Tech Stack)

### 대시보드 및 시각화
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)

### 데이터 분석 및 처리
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

### 머신러닝 및 딥러닝
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-3F4F75?style=flat-square) ![LightGBM](https://img.shields.io/badge/LightGBM-747F00?style=flat-square)

### API 및 인프라
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white) ![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)

---

## 예측 AI 모델 소개 (Prediction AI Models)

본 대시보드는 다양한 데이터 특성을 포착하기 위해 총 4가지 AI 예측 모델을 비교 활용하며, 각 모델의 검증 데이터 기준 실측 오차(MAE) 및 입력 피처(Features) 구성은 다음과 같습니다.

### 1. XGBoost (eXtreme Gradient Boosting)
* **설명**: 오차를 점진적으로 보정해 나가는 고정밀 머신러닝 알고리즘으로 과적합을 방지하고 뛰어난 예측 성능을 냅니다.
* **검증 성능 (Test Set MAE)**:
  - **승차 예측**: **`72.8 명`** (기존 대비 78.1명 개선) / RMSE: `166.3 명` / MAPE: `18.76 %`
  - **하차 예측**: **`77.7 명`** (기존 대비 84.6명 개선) / RMSE: `179.9 명` / MAPE: `17.62 %`
* **사용 피처 (25개)**: 역명, 호선, 시간(및 삼각함수 인코딩), 요일(및 삼각함수 인코딩), 월(및 삼각함수 인코딩), 공휴일여부, 비근무일, 출퇴근피크, 날씨 변수(기온, 강수량, 적설량, 강수/적설여부, 불쾌지수), 연휴 및 공휴일 전후 정보 등

### 2. LSTM (Long Short-Term Memory)
* **설명**: 시계열 분석에 최적화된 순환 신경망(RNN) 계열 딥러닝 모델로, 과거의 연속적인 시간 흐름 패턴을 학습합니다.
* **검증 성능 (Test Set MAE)**:
  - **승차 예측**: **`104.1 명`** / RMSE: `246.3 명` / MAPE: `16.17 %`
  - **하차 예측**: **`127.6 명`** / RMSE: `360.9 명` / MAPE: `19.11 %`
* **사용 피처 (13개 x 12 Time steps)**: 역명, 시간, 승하차 인원, 요일, 월, 공휴일여부, 날씨(기온, 강수량, 적설량), 요일_시간 및 역별 승하차 평균 등

### 3. LightGBM (Light Gradient Boosting Machine)
* **설명**: 마이크로소프트에서 개발한 초고속 트리 부스팅 알고리즘으로, 대규모 400만 데이터셋을 빠르게 연산합니다.
* **검증 성능 (Test Set MAE)**: XGBoost와 유사하게 우수한 수준의 속도 및 70~80명대 오차 성능 기록
* **사용 피처 (12개)**: 역명, 호선, 시간, 요일, 월, 공휴일여부, 기온, 강수량, 적설량, 년, 일, 주말 여부 등

### 4. RandomForest
* **설명**: 수많은 의사결정 나무를 학습시켜 다수결 또는 평균으로 최종 결론을 내는 안정성 높은 정통 앙상블 모델입니다.
* **검증 성능 (Test Set MAE)**:
  - **승차 예측**: **`201.06 명`** / RMSE: `410.99 명` / R² Score: `0.8477`
  - **하차 예측**: **`211.04 명`** / RMSE: `430.01 명` / R² Score: `0.8485`
* **사용 피처 (10개)**: 역명, 호선, 시간, 요일, 월, 일, 공휴일여부, 기온, 강수량, 적설량

---

## 각 대시보드 사이트별 기능 상세 (App-specific Features)

본 프로젝트는 순환선에 특화된 **2호선 전용 앱**과 전 노선을 아우르는 **1-8호선 통합 앱**으로 이원화되어 작동합니다.

### 1. 2호선 대시보드 앱 (`line2`)
* **순환선 최적화 관제 맵**: 2호선 본선 순환선(타원형 궤도) 및 성수지선, 신도림지선의 분기 궤도를 시각적으로 왜곡 없이 매핑하여 표현합니다.
* **내선/외선 실시간 도착 분기**: 순환선 특성에 맞춰 상하행 대신 **'내선순환 (상행)'** 및 **'외선순환 (하행)'** 으로 도착 정보를 실시간 구분하여 제공합니다.
* **2호선 특화 예측 모델**: 2호선 단일 노선의 요일별, 시간별 데이터에 초집중 학습된 경량 모델들을 사용하여 매우 빠르고 직관적인 승하차 추론을 수행합니다.

### 2. 1~8호선 통합 대시보드 앱 (`line1_8`)
* **동적 다중 노선 필터링**: 사용자가 사이드바에서 관심 호선(1~8호선)을 선택하면, 해당 호선에 소속된 역 목록만 UI에 실시간으로 필터링되어 동적으로 마운트됩니다.
* **복잡 지선/본선 입체적 지도 렌더링**: 
  - 1호선 서울역~청량리 구간 단일선 렌더링
  - 2호선 순환선 및 양대 지선 렌더링
  - 5호선 강동역 분기(상일동/마천 방면 Y자 노선) 보간 렌더링
  - 6호선 응암순환 루프선(독바위, 역촌 등 순환 구간) 및 본선 분리 렌더링
  - 3, 4, 7, 8호선의 기하학적 궤도선 자동 보간 등 복잡한 전철망을 단일 Plotly 캔버스로 완벽히 구현합니다.
* **대용량 통합 예측 엔진**: 2년치 400만 행 데이터셋을 단일 모델로 학습한 대용량 통합 모델(LSTM, XGBoost 등)을 가동하여 1~8호선 전 노선 280여 개 역에 대한 표준화된 예측 결과를 연산합니다.

---

## 주요 기능 (Key Features)

### 1. 실시간 지하철 위치 관제 맵 (`tab0`)
- **인터랙티브 Glow 노선도**: Plotly 기반의 실시간 지하철 노선 궤도선과 마커 시각화 및 역 선택 연동 인터랙션.
- **실시간 열차 위치 오버레이**: 서울시 오픈 데이터 API 기반의 실시간 상/하행 열차 방향(삼각형 마커) 및 운행 정보 실시간 렌더링.
- **상·하행 도착 정보 분기**: 역별 실시간 도착 정보를 2컬럼 카드로 한눈에 파악할 수 있는 정밀 파싱 UI.

### 2. 다각적 인공지능 예측 & 시뮬레이터 (`tab1~3`)
- **시간대별 예측**: 선택한 날짜/시간 조건에서 24시간 승하차 인원 예측 선 그래프 및 피크타임 가이드 제시.
- **날씨 시뮬레이터**: 기상청 실시간 예보 API 동기화 및 수동 날씨 조건(기온, 강수량, 적설량) 스위칭에 따른 혼잡도 변동성 시뮬레이션.
- **역별 랭킹**: 동일 시점 기준 전체 노선 내 역별 혼잡도 랭킹 카딩 및 상위 25개 역 비교 바 차트 시각화.

### 3. 비교 분석 및 최적화 추천 (`tab4~5`)
- **최적 탑승 시간 플래너**: 출발역과 도착역의 24시간 혼잡도를 합성 비교하여 가장 한산한 추천 시간대 추천.
- **다중 모델 비교**: 4개 모델(Numpy-backend LSTM, LightGBM, XGBoost, RandomForest)의 24시간 예측 선 그래프 대조 및 최근 48개 실제 관측 데이터 기반 실시간 오차(MAE) 검증.

---

## 기술적 성과: 400만 행 데이터 83% 메모리 다이어트
대용량 데이터셋(3,986,800행)의 판다스 적재 시 발생하는 OOM(메모리 부족) 에러를 방지하기 위해 정교한 **타입 다운캐스팅 및 범주형 사상 기술**을 도입했습니다.
- **정수/실수 다운캐스팅**: 64비트 기본 정밀도(`int64`/`float64`)를 값의 범위에 맞게 최소화(`int8`/`float32`)하여 낭비되는 메모리 블록을 회수했습니다.
- **문자열 범주화(Category)**: 중복도가 높은 문자열 데이터(역명, 호선명)를 범주형 사전 구조(`category`)로 변환하여 문자열 객체 오버헤드를 소멸시켰습니다.
- **메모리 절감 실측치**: **최적화 전 `812.5 MB` ➡️ 최적화 후 `133.1 MB`로 감소하여 총 `83.6%` 의 메모리 절감 효과를 달성**했습니다.

---

## 프로젝트 구조 (Directory Structure)

본 프로젝트는 두 개의 독립적인 Streamlit 앱 및 배포 구조로 이원화되어 있으며, 각 폴더 내부에 LSTM 시계열 추론 및 모델 예측에 필요한 2개년(2023-2024년) 실적 데이터셋이 포함되어 있습니다.

```bash
├── line2/            # 2호선 전용 실시간 관제 및 예측 서비스 패키지
│   ├── components/   # 스타일 및 화면 컴포넌트 탭 모듈
│   ├── core/         # Numpy LSTM 가중치 모델 추론 모듈 및 config
│   ├── data/
│   │   └── processed/
│   │       └── final_dataset_line2_230101-241231.csv # [2개년 데이터셋] 2호선 전용 2개년 승하차 및 날씨 통합 데이터 (약 46MB)
│   ├── services/     # 기상청 및 서울시 실시간 API 연동 서비스
│   └── subway_app.py # 2호선 메인 엔트리 스크립트
│
└── line1_8/          # 1~8호선 통합 실시간 관제 및 예측 서비스 패키지
    ├── components/   # 1-8호선용 대칭 쌍둥이 아키텍처 컴포넌트
    ├── core/         # 대용량 통합 모델 추론기 및 config
    ├── data/
    │   └── processed/
    │       └── final_dataset_line1_8_230101-241231.csv # [2개년 데이터셋] 1~8호선 통합 2개년 승하차 및 날씨 통합 데이터 (약 247MB, 400만 행)
    ├── services/     # 1-8호선 실시간 API 연동 서비스
    └── subway_app.py # 1-8호선 메인 엔트리 스크립트
```

### 2개년 데이터셋 용도 및 메모리 다이어트
* **시계열 추론의 기준점 (LSTM)**: LSTM 딥러닝 모델은 과거 12개 시간대(Time step)의 승하차 실적을 입력으로 받아 미래 혼잡도를 추론합니다. 이 데이터셋은 사용자가 선택한 날짜/시간의 직전 12시간 실적 데이터를 추출하기 위한 핵심 데이터 풀 역할을 합니다.
* **메모리 절감 및 OOM 방지**: 판다스(Pandas)로 데이터셋 로딩 시 발생하는 메모리 OOM(약 812MB RAM 점유)을 줄이기 위해 데이터 다운캐스팅(int64 ➡️ int8, float64 ➡️ float32) 및 역명/호선 범주형 사상(category)을 적용하여 **133.1MB(83.6% 절감)** 수준으로 대폭 다이어트하여 효율적으로 메모리상에서 작동되게 조치했습니다.


---

## 로컬 기동 및 실행 방법 (Local Run)

로컬 가상환경 및 의존 패키지 설치 후, 원하는 디렉토리로 이동하여 스트림릿을 구동합니다.

### 2호선 대시보드 구동 (Port 8501)
```bash
cd line2
pip install -r requirements.txt
streamlit run subway_app.py --server.port 8501
```

### 1-8호선 대시보드 구동 (Port 8502)
```bash
cd line1_8
pip install -r requirements.txt
streamlit run subway_app.py --server.port 8502
```

---

## Hugging Face Spaces 배포 가이드
각 폴더(`line2/`, `line1_8/`)의 내부 파일들이 개별 허깅페이스 저장소(Docker Space)의 최상위 루트(Root)에 가도록 푸시하여 배포를 완료합니다. 각 디렉토리 내부에는 Docker 빌드를 위한 `Dockerfile`과 Space 파싱용 `README.md` 메타데이터 헤더가 이미 완비되어 있습니다.
- 지하철 1~8호선 혼잡도 : https://huggingface.co/spaces/lololoooool/subway_line1_8
- 지하철 2호선 혼잡도 : https://huggingface.co/spaces/lololoooool/subway_line2


