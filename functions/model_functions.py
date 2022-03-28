from typing import Optional
from fastapi import Request, Query

import pandas as pd
# import modin.pandas as pd

from .internal_func import (
    s3_model_save, 
    s3_model_load, 
    boolean,
    check_error,
    MODELS,
    METRICS,
    OPTIMIZERS,
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
    try   : model = s3_model_load(key)
    except: return False, "모델을 불러오는데 실패하였습니다."

    model_type = type(model)
    print(model_type)
    print(OPTIMIZERS)
    if model_type in OPTIMIZERS.values():
        if "best_estimator_" in model.__dict__:
            return True, [i for i in model.best_estimator_.named_steps.keys()]
        else: 
            return False, "훈련되지 않은 옵티마이저 입니다."
    else:
        return True, [i for i in model.named_steps.keys()]

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
    try   : pipe = s3_model_load(key)
    except: return False, "모델을 불러오는데 실패하였습니다."
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
    try   : model = s3_model_load(key)
    except: return False, "모델을 불러오는데 실패하였습니다."

    model_type = type(model)
    if model_type in OPTIMIZERS.values():
        if "best_estimator_" in model.__dict__:
            steps = model.best_estimator_.named_steps
        else: 
            return False, "훈련되지 않은 옵티마이저 입니다."
    else:
        steps = model.named_steps

    if target is not None:
        if target not in set(steps):
            return False, f'"target" should be in pipe: {steps}'

    for key, v in list(steps.items()):
        if key in MODELS:
            break
        if type(X) == pd.DataFrame:
            cols = X.columns
        try   : X = v.transform(X)
        except: return False, "훈련되지 않은 모델입니다."
        if key == target:
            if type(X) == pd.DataFrame:
                return True, X.to_json(orient="records")
            else:
                return True, pd.DataFrame(X, columns=cols).to_json(orient="records")
    if type(X) == pd.DataFrame:
        return True, X.to_json(orient="records")
    else:
        return True, pd.DataFrame(X, columns=cols).to_json(orient="records")


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
    try   : pipe = s3_model_load(key)
    except: return False, "모델을 불러오는데 실패하였습니다."

    cols = X_train.columns
    if y_train: df = pd.DataFrame(pipe.fit_transform(X_train, y_train), columns=cols)
    else      : df = pd.DataFrame(pipe.fit_transform(X_train), columns=cols)

    try   : s3_model_save(key, pipe)
    except: return False, "훈련에는 성공하였으나 모델 저장에 실패하였습니다.(알 수 없는 에러)"

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
    try   : model = s3_model_load(key)
    except: return False, "모델을 불러오는데 실패하였습니다."

    # 모델 학습
    model.fit(X_train, y_train)

    # 학습된 객체 s3에 저장
    try   : s3_model_save(key, model)
    except: return False, "훈련에는 성공하였으나 모델 저장에 실패하였습니다.(알 수 없는 에러)"

    ## 예측 되는지 확인(테스트용)
    # return pd.DataFrame(pipe.predict(X_train), columns=["Predict"]).to_json(orient="records")

    # 훈련 완료 메시지 리턴
    type_model = type(model)
    if type_model in OPTIMIZERS.values():
        # optimizer
        return True, f"optimizing completed, best parameters = {model.best_params_}, best score = {model.best_score_}"
    else:
        # single model or pipeline model
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
    item (Request, required): JSON, {"X_train": dataframe, "X_valid": dataframe, "X_test":dataframe}
    name (str,     required): 생성한 모델를 저장할 파일명
    key  (str,     required): 키를 생성해서 리턴해야 하는 지 논의 필요!
    ```
    Returns:
    ```
    str: JSON, {"y_pred":..., "y_pred_proba":...}
    ```
    """

    # 데이터 로드
    item = await item.json()

    # proba = boolean(proba)
    # if proba is None: return False, '"proba" should be bool, "true" or "false"'

    # s3에서 모델 객체 불러오기
    key = key+"/"+name
    try   : pipe = s3_model_load(key)
    except: return False, "모델을 불러오는데 실패하였습니다."

    # 예측 proba: True 예상 확률, False 예상 label
    ## {"y_pred":..., "y_pred_proba":...}
    output = {}
    for name, X_json in item.items():
        if X_json is None:
            continue
        X = pd.read_json(X_json)
        try:
            output[name] = {
                "y_pred"      : pd.DataFrame(pipe.predict(X)).to_json(orient="records"),
                "y_pred_proba": pd.DataFrame(pipe.predict_proba(X)).to_json(orient="records")
            }
        except ValueError:
            return False, "훈련되지 않은 함수입니다. fit 함수를 실행 후 다시 predict 함수를 해주세요."  
        except:
            output[name] = {
                "y_pred": pd.DataFrame(pipe.predict(X)).to_json(orient="records")
            }
    
    return True, output


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

    # s3 에서 모델 객체 불러오기
    key = key+"/"+name
    try   : pipe = s3_model_load(key)
    except: return False, "모델을 불러오는데 실패하였습니다."

    # 모델 학습
    pipe.fit(X_train, y_train)

    # save가 true면 학습된 모델 객체 s3에 저장
    if save: 
        try   : s3_model_save(key, pipe)
        except: return False, "훈련에는 성공하였으나 모델 저장에 실패하였습니다.(알 수 없는 에러)"

    # 예측 proba: True 예상 확률, False 예상 label
    ## {"y_pred":..., "y_pred_proba":...}
    output = {}
    for name, X_json in item.items():
        if X_json is None:
            continue
        X = pd.read_json(X_json)
        try:
            output[name] = {
                "y_pred"      : pd.DataFrame(pipe.predict(X)).to_json(orient="records"),
                "y_pred_proba": pd.DataFrame(pipe.predict_proba(X)).to_json(orient="records")
            }
        except:
            output[name] = {
                "y_pred": pd.DataFrame(pipe.predict(X)).to_json(orient="records")
            }
    
    return True, output


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
    item = await item.json()

    output = {}
    for name, i_y in item.items():
        if i_y is None:
            continue

        y_true = pd.read_json(i_y["y_true"])

        try:
            y_pred = pd.read_json(i_y["y_pred_proba"]).iloc[:,1] if metric in ["roc_auc"] else pd.read_json(i_y["y_pred"]) 
            output[name] = f"{METRICS[metric](y_true, y_pred)}"
        
        except:
            # roc_auc 는 y_pred가 predict proba가 들어가야함
            return False, f'"{metric}"은 회귀 모델에서 사용할 수 없습니다.'
        
    # 점수 값 리턴
    return True, output


@check_error
async def model_predict_score(
    item   : Request,
    metric : str,
    name   : str,
    key    : str,
) -> tuple:

    # 데이터 로드
    item = await item.json()

    # 모델 로드
    key = key+"/"+name
    try   : pipe = s3_model_load(key)
    except: return False, "모델을 불러오는데 실패하였습니다."

    # predict
    y_preds = {}
    for name, X_json in item.items():
        if X_json is None:
            continue
        X = pd.read_json(X_json)
        try:
            y_preds[name] = {
                "y_pred"      : pd.DataFrame(pipe.predict(X)).to_json(orient="records"),
                "y_pred_proba": pd.DataFrame(pipe.predict_proba(X)).to_json(orient="records")
            }
        except:
            y_preds[name] = {
                "y_pred": pd.DataFrame(pipe.predict(X)).to_json(orient="records")
            }

    
    # score
    output = {}
    for name, i_y in y_preds.items():
        if i_y is None:
            continue
        y_true = pd.read_json(i_y["y_true"])
        try:
            y_pred = pd.read_json(i_y["y_pred_proba"]).iloc[:,1] if metric in ["roc_auc"] else pd.read_json(i_y["y_pred"]) 
            output[name] = f"{METRICS[metric](y_true, y_pred)}"
        except:
            return False, f'"{metric}"은 회귀 모델에서 사용할 수 없습니다.'
            # roc_auc 는 y_pred가 predict proba가 들어가야함
    
    # 점수 값 리턴
    return True, output


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

    # 모델 로드
    key = key+"/"+name
    try   : pipe = s3_model_load(key)
    except: return False, "모델을 불러오는데 실패하였습니다."

    # 모델 학습
    pipe.fit(X_train, y_train)

    # 모델 세이브 if save is True 
    if save: 
        try   : s3_model_save(key, pipe)
        except: return False, "훈련에는 성공하였으나 모델 저장에 실패하였습니다.(알 수 없는 에러)"

    # predict
    y_preds = {}
    for name, X_json in item.items():
        if X_json is None:
            continue
        X = pd.read_json(X_json)
        try:
            y_preds[name] = {
                "y_pred"      : pd.DataFrame(pipe.predict(X)).to_json(orient="records"),
                "y_pred_proba": pd.DataFrame(pipe.predict_proba(X)).to_json(orient="records")
            }
        except:
            y_preds[name] = {
                "y_pred": pd.DataFrame(pipe.predict(X)).to_json(orient="records")
            }

    # score
    output = {}
    for name, i_y in y_preds.items():
        if i_y is None:
            continue
        y_true = pd.read_json(i_y["y_true"])
        try:
            y_pred = pd.read_json(i_y["y_pred_proba"]).iloc[:,1] if metric in ["roc_auc"] else pd.read_json(i_y["y_pred"]) 
            output[name] = f"{METRICS[metric](y_true, y_pred)}"
        except:
            return False, f'"{metric}"은 회귀 모델에서 사용할 수 없습니다.'
            # roc_auc 는 y_pred가 predict proba가 들어가야함

    # 점수 값 리턴
    return True, output