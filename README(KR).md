# AI Play ML-Train

머신 러닝 관련 기능 중 모델 학습 기능을 위한 API 서버

## :one: Stack

- Python 3.8.12
- FastAPI 0.73.0
- Pandas 1.4.1
- scikit-learn 1.0.2
- JWT
- Swagger

<br/>

## 2️⃣ 배포 플랫폼 및 서버 주소

- 플랫폼 : Heroku
- 주소 : https://aiplay-mltrain.herokuapp.com/

<br/>

## :three: API 명세

- DOCS : https://aiplay-mltrain.herokuapp.com/docs

<details>
  <summary>펼쳐보기</summary>

| Method | URL                      | Description                                                                 |
| ------ | ------------------------ | --------------------------------------------------------------------------- |
| GET    | /model/steps             | 모델 파이프라인 구성 요소(인코더, 스케일러, 모델) 목록 (단계별 제목만) 출력 |
| GET    | /model/steps_detail      | 모델 파이프라인 구성 요소(인코더, 스케일러, 모델) 목록 출력                 |
| POST   | /model/transform         | 인코더, 스케일러에 의한 데이터프레임 변환 결과 출력                         |
| POST   | /model/fit_transform     | 모델 학습 후 데이터프레임 변환 결과 출력                                    |
| POST   | /model/fit               | 모델 훈련 수행                                                              |
| POST   | /model/predict           | 모델의 타겟 예측값 생성                                                     |
| POST   | /model/score             | 모델의 예측 성능 측정                                                       |
| POST   | /model/fit_predict       | 모델 훈련 후 타겟 예측값 생성                                               |
| POST   | /model/predict_score     | 타겟 예측값 생성 후 예측 성능 측정                                          |
| POST   | /model/fit_predict_score | 모델 훈련 -> 예측값 생성 -> 모델 성능 측정                                  |
| POST   | /model/make_encoder      | 인코더 객체 생성 및 저장                                                    |
| POST   | /model/make_scaler       | 스케일러 객체 생성 및 저장                                                  |
| POST   | /model/make_model        | 모델 객체 생성 및 저장                                                      |
| POST   | /model/make_pipeline     | 모델 파이프라인 객체 생성 및 저장                                           |
| POST   | /model/make_optimizer    | 모델 옵티마이저 객체 생성 및 저장                                           |

</details>

<br/>

## :four: 트러블 슈팅 기록

- https://github.com/AI-Play/ML-Train/discussions

<br/>

## :five: 개발 환경 준비 사항

<details>
  <summary>펼쳐보기</summary>

```
# 새 가상환경 만들기
# 1. 사용해야 할 python version이 있는 디렉토리로 이동
# 2. 새 가상환경 생성을 위한 명령어 실행
python -m venv /path/to/new/virtual/environment

# 3. 가상환경 활성화하기
source /path/to/new/virtual/environment/bin/activate

# 4. 필요한 패키지 설치
pip install -r requirements.txt
```

##### 실행

```
uvicorn main:app --reload
```

</details>
