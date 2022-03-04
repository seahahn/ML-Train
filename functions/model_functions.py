from typing import Optional
from fastapi import Request, Query
import json, pickle, boto3, io

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
from .make_model import MODELS, BUCKET

SCORES = {
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


def s3_model_save(key, body):
    s3 = boto3.client('s3') 
    s3.put_object(
        Bucket = BUCKET,
        Key    = key,
        Body   = pickle.dumps(body)
    )


def s3_model_load(key):
    s3 = boto3.client('s3') 
    file = io.BytesIO(
        s3.get_object(
            Bucket = BUCKET,
            Key    = key
        )["Body"].read()
    )
    return pickle.load(file)


async def model_steps(
    name   : str,
    key    : str,
) -> str:
    """
    ```python
    pipe.named_steps # 를 리턴하는 함수
    ```
    Args:
    ```
    name   (str, required): 생성한 모델를 저장할 파일명
    key    (str, required): 키를 생성해서 리턴해야 하는 지 논의 필요!
    ```
    Returns:
    ```
    str: list of named steps exclude ML model
    ```
    """
    key = key+"/"+name
    pipe = s3_model_load(key)
    return [i for i in pipe.named_steps.keys() if i not in MODELS]


async def model_transform(
    item   : Request,
    name   : str,
    key    : str,
    *,
    target : Optional[str] = Query(None, max_length=50),
) -> str:
    """
    ```
    S3의 bucket에서 key/name 을 load함.
    불러온 객체의 메소드 transform을 사용.
    ```
    Args:
    ```
    item   (Request, required): JSON, transform할 DataFrame
    name   (str,     required): 생성한 모델를 저장할 파일명
    key    (str,     required): 키를 생성해서 리턴해야 하는 지 논의 필요!
    *
    target (str,     optional): Default: None, pipe에 있는 모델만 가능
    ```
    Returns:
    ```
    str: JSON. transform 결과
    ```
    """
    target = None if target == "" else target
    
    X = pd.read_json(await item.json())

    # # 테스트용
    # name   = "test_pipe.pickle"
    # key    = "test"


    ## s3 에서 객체 불러오기
    key = key+"/"+name
    pipe = s3_model_load(key)
    steps = pipe.named_steps
    if target is not None:
        if target not in set(steps):
            return f'"target" should be in pipe: {steps}'

    try:
        for key, v in list(steps.items()):
            if key in MODELS:
                break
            if type(X) == pd.DataFrame:
                cols = X.columns
            X = v.transform(X)
            if key == target:
                if type(X) == pd.DataFrame:
                    return X.to_json(orient="records")
                else:
                    return pd.DataFrame(X, columns=cols).to_json(orient="records")
        if type(X) == pd.DataFrame:
            return X.to_json(orient="records")
        else:
            return pd.DataFrame(X, columns=cols).to_json(orient="records")
    except:
        return "훈련되지 않은 모델입니다."


async def model_fit_transform(
    item   : Request,
    name   : str,
    key    : str, 
) -> str:
    """
    ```
    S3의 bucket에서 key/name 을 load함.
    불러온 객체의 메소드 fit_transform을 사용.
    fit한 객체를 다시 똑같은 S3 bucket의 key/name에 save함

    타겟 인코더 등의 fit을 하기위해 y값도 받아야함
    y_train을 사용하지 않을 경우 JSON에서 "y_train"을 없애거나, "y_train":None 을 입력
    JSON 형식
    item = {
        "X_train": DataFrame,
        "y_train": DataFrame or None
    }
    ```
    Args:
    ```
    item   (Request, required): JSON, transform할 DataFrame.
    name   (str,     required): 생성한 모델를 저장할 파일명
    key    (str,     required): 키를 생성해서 리턴해야 하는 지 논의 필요!
    ```
    Returns:
    ```
    str: JSON. transform 결과
    ```
    """
    item = await item.json()
    X_train = pd.read_json(item["X_train"])
    if "y_train" in item:
        y_train = pd.read_json(item["y_train"])
    else:
        y_train = None

    # # 테스트용
    # name   = "test_pipe.pickle"
    # key    = "test"


    ## s3 에서 객체 불러오기
    key = key+"/"+name
    pipe = s3_model_load(key)

    cols = X_train.columns
    if y_train: df = pd.DataFrame(pipe.fit_transform(X_train, y_train), columns=cols)
    else      : df = pd.DataFrame(pipe.fit_transform(X_train), columns=cols)

    s3_model_save(key, pipe)
    return df.to_json(orient="records")


async def model_fit(
    item   : Request,
    name   : str,
    key    : str, 
) -> str:
    """
    ```
    S3의 bucket에서 key/name 을 load함.
    불러온 객체의 메소드 fit_transform을 사용.
    fit한 객체를 다시 똑같은 S3 bucket의 key/name에 save함

    타겟 인코더 등의 fit을 하기위해 y값도 받아야함
    y_train을 사용하지 않을 경우 JSON에서 "y_train"을 없애거나, "y_train":None 을 입력
    JSON 형식
    item = {
        "X_train": DataFrame,
        "y_train": DataFrame or None
    }
    ```
    Args:
    ```
    item   (Request, required): JSON, transform할 DataFrame.
    name   (str,     required): 생성한 모델를 저장할 파일명
    key    (str,     required): 키를 생성해서 리턴해야 하는 지 논의 필요!
    ```
    Returns:
    ```
    str: 성공 메시지
    ```
    """
    # 입력 X_train, y_train
    item = await item.json()
    X_train = pd.read_json(item["X_train"])
    y_train = pd.read_json(item["y_train"])

    if X_train.shape[0] != y_train.shape[0]:
        return "X_train과 y_train의 row 길이가 같아야합니다. [shape = (row,column)]"

    # # 임시 전처리(테스트용)
    # df = pd.read_json(await item.json())
    # df = df.dropna(thresh = 500, axis=1)
    # df = df.dropna().reset_index(drop=True)
    # df = df.drop(["PassengerId","Name", "Ticket", "Date", "0", "1", "Cabin"], axis=1)
    # X_train = df.drop(["Survived"],axis=1)
    # y_train = df["Survived"]


    # # 테스트용
    # name   = "test_pipe.pickle"
    # key    = "test"


    ## s3 에서 객체 불러오기
    key = key+"/"+name
    pipe = s3_model_load(key)

    ## 로컬에서 객체 불러오기(테스트용)
    # with open("test_pipe.pickle", "rb") as f:
    #     pipe = pickle.load(f)

    # 모델 학습
    pipe.fit(X_train, y_train)

    # 학습된 객체 s3에 저장
    s3_model_save(key, pipe)

    ## 예측 되는지 확인(테스트용)
    # return pd.DataFrame(pipe.predict(X_train), columns=["Predict"]).to_json(orient="records")

    # 훈련 완료 메시지 리턴
    return "training completed"


async def model_predict(
    item   : Request,
    name   : str,
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
    # key    = "test"

    # s3에서 모델 객체 불러오기
    key = key+"/"+name
    pipe = s3_model_load(key)
    
    # 예측 proba: True 예상 확률, False 예상 label
    if proba: return pd.DataFrame(pipe.predict_proba(X_test), columns=["y_pred_prob"]).to_json(orient="records")
    else    : return pd.DataFrame(pipe.predict(X_test), columns=["y_pred"]).to_json(orient="records")


async def model_fit_predict(
    item   : Request,
    name   : str,
    key    : str,
    *,
    save   : Optional[str] = Query("true",  max_length=50),
    proba  : Optional[str] = Query("false", max_length=50),
) -> str:

    # # 테스트용 입력
    # name   = "test_pipe.pickle"
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
    pipe = s3_model_load(key)

    # 모델 학습
    pipe.fit(X_train, y_train)

    # save가 true면 학습된 모델 객체 s3에 저장
    if save: s3_model_save(key, pipe)

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
    print(SCORES[score](y_true, y_pred, **params))
    return SCORES[score](y_true, y_pred, **params)


async def model_predict_score(
    item   : Request,
    score  : str,
    name   : str,
    key    : str,
    *,
    proba  : Optional[str] = Query("false", max_length=50)
) -> str:

    # # 테스트용 입력
    # name   = "test_pipe.pickle"
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
    pipe = s3_model_load(key)

    # 예측 proba: True 예상 확률, False 예상 label
    if proba: y_pred = pd.DataFrame(pipe.predict_proba(X_valid), columns=["y_pred_prob"]).to_json(orient="records") # roc_auc
    else    : y_pred = pd.DataFrame(pipe.predict(X_valid), columns=["y_pred"]).to_json(orient="records")
    
    # 점수 값 리턴
    print(SCORES[score](y_valid, y_pred, **params))
    return SCORES[score](y_valid, y_pred, **params)


async def model_fit_predict_score(
    item   : Request,
    score  : str,
    name   : str,
    key    : str,
    *,
    save   : Optional[str] = Query("true", max_length=50),
    proba  : Optional[str] = Query("false", max_length=50),
) -> str:

    # # 테스트용 입력
    # name   = "test_pipe.pickle"
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
    pipe = s3_model_load(key)

    # 모델 학습
    pipe.fit(X_train, y_train)

    # 모델 세이브 if save is True 
    if save: s3_model_save(key, pipe)

    # 예측 proba: True 예상 확률, False 예상 label
    if proba: y_pred = pd.DataFrame(pipe.predict_proba(X_valid), columns=["y_pred_prob"]).to_json(orient="records") # for roc_auc
    else    : y_pred = pd.DataFrame(pipe.predict(X_valid), columns=["y_pred"]).to_json(orient="records")

    # 점수 값 리턴
    print(SCORES[score](y_train, y_pred, **params)) # train score
    print(SCORES[score](y_valid, y_pred, **params)) # valid score
    return str(SCORES[score](y_train, y_pred, **params)), str(SCORES[score](y_valid, y_pred, **params))