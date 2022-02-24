from typing import Optional
from fastapi import Request, Query
import json
import pickle
import boto3
import io

import pandas as pd
# import modin.pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

scores = {
    ## Classification
    "accuracy"         : accuracy_score,
    "f1"               : f1_score,
    "roc_auc"          : roc_auc_score,  # requires predict_proba support
    "precision"        : precision_score,
    "recall"           : recall_score,
    # "balanced_accuracy": balanced_accuracy_score,
    # "top_k_accuracy"   : top_k_accuracy_score,
    # "average_precision": average_precision_score,
    # "neg_brier_score"  : brier_score_loss,
    # "neg_log_loss"     : log_loss, # requires predict_proba support
    # "jaccard"          : jaccard_score,

    ## Regression
    "r2"                         :r2_score,
    "neg_mean_absolute_error"    :mean_absolute_error,
    "neg_mean_squared_error"     :mean_squared_error,
    # "explained_variance"         :explained_variance_score,
    # "max_error"                  :max_error,
    # "neg_mean_squared_log_error" :mean_squared_log_error,
    # "neg_median_absolute_error"  :median_absolute_error,
    # "neg_mean_poisson_deviance"  :mean_poisson_deviance,
    # "neg_mean_gamma_deviance"    :mean_gamma_deviance,
    # "neg_mean_absolute_percentage_error":mean_absolute_percentage_error,
}

def boolean(x):
    if   x.lower() == "true" : return True
    elif x.lower() == "false": return False


def s3_model_save(bucket, key, body):
    s3 = boto3.client('s3') 
    s3.put_object(
        Bucket = bucket,
        Key    = key,
        Body   = pickle.dumps(body)
    )


def s3_model_load(bucket, key):
    s3 = boto3.client('s3') 
    file = io.BytesIO(
        s3.get_object(
            Bucket = bucket,
            Key    = key
        )["Body"].read()
    )
    return pickle.load(file)


async def model_transform(
    item   : Request,
    name   : str,
    bucket : str,
    key    : str, 
) -> str:

    item = json.loads(await item.json())
    X_train = pd.DataFrame(item[X_train])

    # # 테스트용
    # name   = "test_pipe.pickle"
    # bucket = "aiplay-test-bucket"
    # key    = "test"


    ## s3 에서 객체 불러오기
    key = key+"/"+name
    pipe = s3_model_load(bucket, key)
    return pd.DataFrame(pipe.transform(X_train)).to_json(orient="records")


async def model_fit_transform(
    item   : Request,
    name   : str,
    bucket : str,
    key    : str, 
) -> str:

    item = json.loads(await item.json())
    X_train = pd.DataFrame(item[X_train])
    if "y_train" in item:
        y_train = pd.DataFrame(item[y_train])
    else:
        y_train = None

    # # 테스트용
    # name   = "test_pipe.pickle"
    # bucket = "aiplay-test-bucket"
    # key    = "test"


    ## s3 에서 객체 불러오기
    key = key+"/"+name
    pipe = s3_model_load(bucket, key)

    if y_train: df = pd.DataFrame(pipe.fit_transform(X_train, y_train))
    else      : df = pd.DataFrame(pipe.fit_transform(X_train))

    s3_model_save(bucket, key, pipe)
    return df.to_json(orient="records")


async def model_fit(
    item   : Request,
    name   : str,
    bucket : str,
    key    : str, 
) -> str:

    # 입력 X_train, y_train
    item = json.loads(await item.json())
    X_train = pd.DataFrame(item["X_train"])
    y_train = pd.DataFrame(item["y_train"])

    # # 임시 전처리(테스트용)
    # df = pd.read_json(await item.json())
    # df = df.dropna(thresh = 500, axis=1)
    # df = df.dropna().reset_index(drop=True)
    # df = df.drop(["PassengerId","Name", "Ticket", "Date", "0", "1", "Cabin"], axis=1)
    # X_train = df.drop(["Survived"],axis=1)
    # y_train = df["Survived"]


    # # 테스트용
    # name   = "test_pipe.pickle"
    # bucket = "aiplay-test-bucket"
    # key    = "test"


    ## s3 에서 객체 불러오기
    key = key+"/"+name
    pipe = s3_model_load(bucket, key)

    ## 로컬에서 객체 불러오기(테스트용)
    # with open("test_pipe.pickle", "rb") as f:
    #     pipe = pickle.load(f)

    # 모델 학습
    pipe.fit(X_train, y_train)

    # 학습된 객체 s3에 저장
    s3_model_save(bucket, key, pipe)

    ## 예측 되는지 확인(테스트용)
    # return pd.DataFrame(pipe.predict(X_train), columns=["Predict"]).to_json(orient="records")

    # 훈련 완료 메시지 리턴
    return "training completed"


async def model_predict(
    item   : Request,
    name   : str,
    bucket : str,
    key    : str,
    *,
    proba  : Optional[str] = Query("false", max_length=50)
) -> str:

    # 데이터 로드
    X_test = pd.read_json(await item.json())

    proba = boolean(proba)
    if proba is None: return '"proba" should be bool, "true" or "false"'

    # 테스트용
    # df = df.drop(["PassengerId","Name", "Ticket", "Cabin"], axis=1)
    # X_test = df.dropna().reset_index(drop=True)
    # print(X_test)

    # # 테스트용
    # name   = "test_pipe.pickle"
    # bucket = "aiplay-test-bucket"
    # key    = "test"

    # s3에서 모델 객체 불러오기
    key = key+"/"+name
    pipe = s3_model_load(bucket, key)
    
    # 예측 proba: True 예상 확률, False 예상 label
    if proba: return pd.DataFrame(pipe.predict_proba(X_test), columns=["y_pred_prob"]).to_json(orient="records")
    else    : return pd.DataFrame(pipe.predict(X_test), columns=["y_pred"]).to_json(orient="records")


async def model_fit_predict(
    item   : Request,
    name   : str,
    bucket : str,
    key    : str,
    *,
    save   : Optional[str] = Query("true",  max_length=50),
    proba  : Optional[str] = Query("false", max_length=50),
) -> str:

    # # 테스트용 입력
    # name   = "test_pipe.pickle"
    # bucket = "aiplay-test-bucket"
    # key    = "test"

    save = boolean(save)
    if save is None: return '"save" should be bool, "true" or "false"'

    proba = boolean(proba)
    if proba is None: return '"proba" should be bool, "true" or "false"'

    item = json.loads(await item.json())
    X_train = pd.DataFrame(item["X_train"])
    y_train = pd.DataFrame(item["y_train"])
    X_valid = pd.DataFrame(item["X_valid"])
    # y_valid = pd.DataFrame(item["y_valid"])

    # s3 에서 모델 객체 불러오기
    key = key+"/"+name
    pipe = s3_model_load(bucket, key)

    # 모델 학습
    pipe.fit(X_train, y_train)

    # save가 true면 학습된 모델 객체 s3에 저장
    if save: s3_model_save(bucket, key, pipe)

    # 예측 proba: True 예상 확률, False 예상 label
    if proba: return pd.DataFrame(pipe.predict_proba(X_valid), columns=["y_pred_prob"]).to_json(orient="records")
    else    : return pd.DataFrame(pipe.predict(X_valid), columns=["y_pred"]).to_json(orient="records")


async def model_score(
    item   : Request,
    score  : str,
) -> str:
    
    # 데이터 로드
    ys = json.loads(await item.json())
    y_true = pd.DataFrame(ys["y_true"])
    y_pred = pd.DataFrame(ys["y_pred"]) # roc_auc 는 y_pred가 predict proba가 들어가야함

    # score 파라미터 로드
    try: params = ys[score]
    except: params = {}

    # 점수 값 리턴
    print(scores[score](y_true, y_pred, **params))
    # return scores[score](y_true, y_pred, **params)


async def model_predict_score(
    item   : Request,
    score  : str,
    name   : str,
    bucket : str,
    key    : str,
    *,
    proba  : Optional[str] = Query("false", max_length=50)
) -> str:

    # # 테스트용 입력
    # name   = "test_pipe.pickle"
    # bucket = "aiplay-test-bucket"
    # key    = "test"

    proba = boolean(proba)
    if proba is None: return '"proba" should be bool, "true" or "false"'

    # 데이터 로드
    item = json.loads(await item.json())
    # X_train = pd.DataFrame(item["X_train"])
    # y_train = pd.DataFrame(item["y_train"])
    X_valid = pd.DataFrame(item["X_valid"])
    y_valid = pd.DataFrame(item["y_valid"])

    # score 파라미터 로드
    try: params = item[score]
    except: params = {}

    # 모델 로드
    key = key+"/"+name
    pipe = s3_model_load(bucket, key)

    # 예측 proba: True 예상 확률, False 예상 label
    if proba: y_pred = pd.DataFrame(pipe.predict_proba(X_valid), columns=["y_pred_prob"]).to_json(orient="records") # roc_auc
    else    : y_pred = pd.DataFrame(pipe.predict(X_valid), columns=["y_pred"]).to_json(orient="records")
    
    # 점수 값 리턴
    print(scores[score](y_valid, y_pred, **params))
    # return scores[score](y_valid, y_pred, **params)


async def model_fit_predict_score(
    item   : Request,
    score  : str,
    name   : str,
    bucket : str,
    key    : str,
    *,
    save   : Optional[str] = Query("true", max_length=50),
    proba  : Optional[str] = Query("false", max_length=50),
) -> str:

    # # 테스트용 입력
    # name   = "test_pipe.pickle"
    # bucket = "aiplay-test-bucket"
    # key    = "test"

    save = boolean(save)
    if save is None: return '"save" should be bool, "true" or "false"'

    proba = boolean(proba)
    if proba is None: return '"proba" should be bool, "true" or "false"'

    # 데이터 로드
    item = json.loads(await item.json())
    X_train = pd.DataFrame(item["X_train"])
    y_train = pd.DataFrame(item["y_train"])
    X_valid = pd.DataFrame(item["X_valid"])
    y_valid = pd.DataFrame(item["y_valid"])

    # score 파라미터 로드
    try: params = item[score]
    except: params = {}

    # 모델 로드
    key = key+"/"+name
    pipe = s3_model_load(bucket, key)

    # 모델 학습
    pipe.fit(X_train, y_train)

    # 모델 세이브 if save is True 
    if save: s3_model_save(bucket, key, pipe)

    # 예측 proba: True 예상 확률, False 예상 label
    if proba: y_pred = pd.DataFrame(pipe.predict_proba(X_valid), columns=["y_pred_prob"]).to_json(orient="records") # for roc_auc
    else    : y_pred = pd.DataFrame(pipe.predict(X_valid), columns=["y_pred"]).to_json(orient="records")

    # 점수 값 리턴
    print(scores[score](y_train, y_pred, **params)) # train score
    print(scores[score](y_valid, y_pred, **params)) # valid score
    # return str(scores[score](y_train, y_pred, **params)), str(scores[score](y_valid, y_pred, **params))