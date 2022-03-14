from typing import Optional
from fastapi import Request, Query
import json

import pandas as pd
# import modin.pandas as pd



from .internal_func import (
    s3_model_save, 
    s3_model_load, 
    boolean,
    check_error,
    MODELS,
    METRICS,
)


## 나중에 sql로 바꿀 예정
@check_error
async def model_steps(
    name   : str,
    key    : str,
) -> tuple:
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
    return True, [i for i in pipe.named_steps.keys()]


@check_error
async def model_steps_detail(
    name   : str,
    key    : str,
) -> tuple:
    """
    ```python
    pipe 정보 그대로 리턴하는 함수
    ```
    Args:
    ```
    name   (str, required): 모델 파일명
    key    (str, required): 키 (S3 버킷 이하 경로)
    ```
    Returns:
    ```
    str: list of named steps exclude ML model
    ```
    """
    key = key+"/"+name
    pipe = s3_model_load(key)
    return True, f'{pipe}'


@check_error
async def model_transform(
    item   : Request,
    name   : str,
    key    : str,
    *,
    target : Optional[str] = Query(None, max_length=50),
) -> tuple:
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
            return False, f'"target" should be in pipe: {steps}'

    try:
        for key, v in list(steps.items()):
            if key in MODELS:
                break
            if type(X) == pd.DataFrame:
                cols = X.columns
            X = v.transform(X)
            if key == target:
                if type(X) == pd.DataFrame:
                    return True, X.to_json(orient="records")
                else:
                    return True, pd.DataFrame(X, columns=cols).to_json(orient="records")
        if type(X) == pd.DataFrame:
            return True, X.to_json(orient="records")
        else:
            return True, pd.DataFrame(X, columns=cols).to_json(orient="records")
    except:
        return False, "훈련되지 않은 모델입니다."


@check_error
async def model_fit_transform(
    item   : Request,
    name   : str,
    key    : str, 
) -> tuple:
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
    return True, df.to_json(orient="records")


@check_error
async def model_fit(
    item   : Request,
    name   : str,
    key    : str, 
) -> tuple:
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
        return False, "X_train과 y_train의 row 길이가 같아야합니다. [shape = (row,column)]"

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
    print(key)
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
    return True, "training completed"


@check_error
async def model_predict(
    item   : Request,
    name   : str,
    key    : str,
    # *,
    # proba  : Optional[str] = Query("false", max_length=50)
) -> tuple:
    """
    ```python
    pipe.predict(X)
    pipe.predict_proba(X) # if ML Model is Classification
    ```
    Args:
    ```
    item (Request, required): JSON, single dataframe
    name (str,     required): 생성한 모델를 저장할 파일명
    key  (str,     required): 키를 생성해서 리턴해야 하는 지 논의 필요!
    ```
    Returns:
    ```
    str: JSON, {"y_pred":..., "y_pred_proba":...}
    ```
    """

    # 데이터 로드
    X_test = pd.read_json(await item.json())

    # proba = boolean(proba)
    # if proba is None: return False, '"proba" should be bool, "true" or "false"'

    # s3에서 모델 객체 불러오기
    key = key+"/"+name
    pipe = s3_model_load(key)

    # 예측 proba: True 예상 확률, False 예상 label
    ## {"y_pred":..., "y_pred_proba":...}
    try:
        return True, {
            "y_pred"      : pd.DataFrame(pipe.predict(X_test)).to_json(orient="records"),
            "y_pred_proba": pd.DataFrame(pipe.predict_proba(X_test)).to_json(orient="records")
        }
    except:
        return True, {
            "y_pred": pd.DataFrame(pipe.predict(X_test)).to_json(orient="records")
        }


@check_error
async def model_fit_predict(
    item   : Request,
    name   : str,
    key    : str,
    *,
    save   : Optional[str] = Query("true",  max_length=50),
) -> tuple:

    # # 테스트용 입력
    # name   = "test_pipe.pickle"
    # key    = "test"

    save = boolean(save)
    if save is None: return False, '"save" should be bool, "true" or "false"'

    item = await item.json()
    X_train = pd.read_json(item["X_train"])
    y_train = pd.read_json(item["y_train"])
    X_valid = pd.read_json(item["X_valid"])
    # y_valid = pd.read_json(item["y_valid"])

    # s3 에서 모델 객체 불러오기
    key = key+"/"+name
    pipe = s3_model_load(key)

    # 모델 학습
    pipe.fit(X_train, y_train)

    # save가 true면 학습된 모델 객체 s3에 저장
    if save: s3_model_save(key, pipe)

    # 예측 proba: True 예상 확률, False 예상 label
    ## {"y_pred":..., "y_pred_proba":...}
    try:
        return True, {
            "y_pred"      : pd.DataFrame(pipe.predict(X_valid)).to_json(orient="records"),
            "y_pred_proba": pd.DataFrame(pipe.predict_proba(X_valid)).to_json(orient="records")
        }
    except:
        return True, {
            "y_pred": pd.DataFrame(pipe.predict(X_valid)).to_json(orient="records")
        }


@check_error
async def model_score(
    item  : Request,
    metric: str,
) -> tuple:
    """
    ```python
    item = {
        "y_ture":..., 
        "y_pred":..., 
        "y_pred_proba":(optional)...
    }
    metric에 아래 metric중 하나를 입력
    ["accuracy", "f1", "roc_auc", "precision", "recall", "r2", "neg_mean_absolute_error", "neg_mean_squared_error"]

    return MATRICS[matric](y_true, y_pred)
    ```
    Args:
    ```
    item  (Request, required): JSON, {"y_ture":..., "y_pred":..., "y_pred_proba":(optional)...}
    metric(str,     required): ["accuracy", "f1", "roc_auc", "precision", "recall", "r2", "neg_mean_absolute_error", "neg_mean_squared_error"] 중 하나
    ```
    Returns:
    ```
    str: 점수
    ```
    """
    # 데이터 로드
    # {"y_ture":..., "y_pred":..., *, "y_pred_proba":...}
    ys = await item.json()
    y_true = pd.read_json(ys["y_true"])
    try:
        y_pred = pd.read_json(ys["y_pred_proba"]).iloc[:,1] if metric in ["roc_auc"] else pd.read_json(ys["y_pred"]) 
    except:
        return False, f'"{metric}"은 회귀 모델에서 사용할 수 없습니다.'
    # roc_auc 는 y_pred가 predict proba가 들어가야함
    
    # score 파라미터 로드
    # try: params = ys[score]
    # except: params = {}

    # 점수 값 리턴
    print(METRICS[metric](y_true, y_pred))
    return True, f"{metric}:{METRICS[metric](y_true, y_pred)}"


@check_error
async def model_predict_score(
    item   : Request,
    metric : str,
    name   : str,
    key    : str,
) -> tuple:

    # 데이터 로드
    item = await item.json()
    # X_train = pd.DataFrame(item["X_train"])
    # y_train = pd.DataFrame(item["y_train"])
    X_valid = pd.read_json(item["X_valid"])
    y_valid = pd.read_json(item["y_valid"])

    # 모델 로드
    key = key+"/"+name
    pipe = s3_model_load(key)

    # 예측 proba: True 예상 확률, False 예상 label
    try:
        y_pred = {
            "y_pred"      : pd.DataFrame(pipe.predict(X_valid)).to_json(orient="records"),
            "y_pred_proba": pd.DataFrame(pipe.predict_proba(X_valid)).to_json(orient="records")
        }
    except:
        y_pred = {
            "y_pred": pd.DataFrame(pipe.predict(X_valid)).to_json(orient="records")
        }
    try:
        y_pred = pd.read_json(y_pred["y_pred_proba"]).iloc[:,1] if metric in ["roc_auc"] else pd.read_json(y_pred["y_pred"]) 
    except:
        return False, f'"{metric}"은 회귀 모델에서 사용할 수 없습니다.'
    
    # 점수 값 리턴
    print(METRICS[metric](y_valid, y_pred))
    return True, METRICS[metric](y_valid, y_pred)


@check_error
async def model_fit_predict_score(
    item   : Request,
    metric : str,
    name   : str,
    key    : str,
    *,
    save   : Optional[str] = Query("true", max_length=50),
) -> tuple:

    # # 테스트용 입력
    # name   = "test_pipe.pickle"
    # key    = "test"

    save = boolean(save)
    if save is None: return False, '"save" should be bool, "true" or "false"'

    # 데이터 로드
    item = await item.json()
    X_train = pd.read_json(item["X_train"])
    y_train = pd.read_json(item["y_train"])
    X_valid = pd.read_json(item["X_valid"])
    y_valid = pd.read_json(item["y_valid"])

    # 모델 로드
    key = key+"/"+name
    pipe = s3_model_load(key)

    # 모델 학습
    pipe.fit(X_train, y_train)

    # 모델 세이브 if save is True 
    if save: s3_model_save(key, pipe)

    # 예측 proba: True 예상 확률, False 예상 label
    try:
        y_pred_train = {
            "y_pred"      : pd.DataFrame(pipe.predict(X_train)).to_json(orient="records"),
            "y_pred_proba": pd.DataFrame(pipe.predict_proba(X_train)).to_json(orient="records")
        }
    except:
        y_pred_train = {
            "y_pred": pd.DataFrame(pipe.predict(X_train)).to_json(orient="records")
        }
    try:
        y_pred = {
            "y_pred"      : pd.DataFrame(pipe.predict(X_valid)).to_json(orient="records"),
            "y_pred_proba": pd.DataFrame(pipe.predict_proba(X_valid)).to_json(orient="records")
        }
    except:
        y_pred = {
            "y_pred": pd.DataFrame(pipe.predict(X_valid)).to_json(orient="records")
        }
    try:
        y_pred_train = pd.read_json(y_pred_train["y_pred_proba"]).iloc[:,1] if metric in ["roc_auc"] else pd.read_json(y_pred_train["y_pred"]) 
    except:
        return False, f'"{metric}"은 회귀 모델에서 사용할 수 없습니다.'
    try:
        y_pred = pd.read_json(y_pred["y_pred_proba"]).iloc[:,1] if metric in ["roc_auc"] else pd.read_json(y_pred["y_pred"]) 
    except:
        return False, f'"{metric}"은 회귀 모델에서 사용할 수 없습니다.'

    # 점수 값 리턴
    print(METRICS[metric](y_train, y_pred_train)) # train score
    print(METRICS[metric](y_valid, y_pred)) # valid score
    return True, {"Train_Score":str(METRICS[metric](y_train, y_pred_train)), "Valid Score":str(METRICS[metric](y_valid, y_pred))}


async def optimizer_fit(
    item         : Request,
    name         : str,
    key          : str,
    *,
    other_metric : Optional[str] = Query(None, max_length=50),
) -> tuple:


    # 데이터 로드
    item = await item.json()
    X_train = pd.read_json(item["X_train"])
    y_train = pd.read_json(item["y_train"])
    X_valid = pd.read_json(item["X_valid"])
    y_valid = pd.read_json(item["y_valid"])

    # 모델 로드
    key = key+"/"+name
    op_model = s3_model_load(key)

    op_model.fit(X_train, y_train)
    
