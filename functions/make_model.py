from typing import Optional
from fastapi import Request, Query
import json
import pickle
import boto3
import io

import pandas as pd
# import modin.pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from category_encoders import OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.linear_model import (
    LinearRegression, 
    LogisticRegression,
    # RidgeClassifier,
    # Ridge,
    # Lasso,
)
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.ensemble import (
#     RandomForestClassifier, 
#     RandomForestRegressor,
# )



ENCODERS = {
    "onehot_encoder" : OneHotEncoder,
    "ordinal_encoder": OrdinalEncoder,
    "target_encoder" : TargetEncoder
}

SCALERS = {
    "standard_scaler": StandardScaler,
    "minmax_scaler"  : MinMaxScaler
}

MODELS = {
    "linear_regression"  : LinearRegression,
    "logistic_regression": LogisticRegression,
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


async def make_encoder(
    item   : Request,
    name   : str, # 저장 될 이름.
    bucket : str,
    key    : str, # 키를 생성해서 리턴해야 하는지 생각중입니다!
    encoder: str,
) -> str:
    """
    ```
    아래의 인코더 중 하나를 선택해서 객체를 생성함.
    OneHotEncoder, OrdinalEncoder, TargetEncoder

    encoder 명은 반드시 아래 리스트에 포함되어 있어야 함.
    ["onehot_encoder", "ordinal_encoder", "target_encoder"]

    해당 인코더의 파라미터를 item에 dictionary(JSON) 형태로 보내야 함.

    생성된 모델은 S3 bucket의 key 경로에 name을 가진 파일로 저장됨.(pickle)
    ```
    Args:
    ```
    item   (Request, required): JSON, 파라미터
    name   (str    , required): 생성한 인코더를 저장할 파일명
    bucket (str    , required): 버켓을 함수에 내장해야 하는지, 쿼리로 날려야 하는지 논의 필요!
    key    (str    , required): 키를 생성해서 리턴해야 하는 지 논의 필요!
    encoder(str    , required): 인코더 1개(해당 함수는 하나의 모델만 생성)
    ```
    Returns:
    ```
    str: 생성 완료 메시지 +  (opt.저장된 위치 key를 해시로 내부에서 생성해서 리턴?))
    ```
    """
    params = await item.json()
    
    if encoder not in ENCODERS:
        return f'"encoder" should be in {list(ENCODERS)}. current {encoder}'
    
    # encoders[encoder](**params["encoders"][encoder])
    # encoders[encoder](**params[encoder])
    x = ENCODERS[encoder](**params)

    # # 테스트용
    # name   = "test_pipe.pickle"
    # bucket = "aiplay-test-bucket"
    # key    = "test"

    # AWS S3 에 pickle 저장
    key = key+"/"+name
    s3_model_save(bucket, key, x)

    return f"Generation Complete: {x}"


async def make_scaler(
    item  : Request,
    name  : str, # 저장 될 이름.
    bucket: str,
    key   : str, # 키를 생성해서 리턴해야 하는지 생각중입니다!
    scaler: str,
) -> str:
    """
    ```
    아래의 스케일러 중 하나를 선택해서 객체를 생성함.
    StandardScaler, MinMaxScaler

    scaler 명은 반드시 아래 리스트에 포함되어 있어야 함.
    ["standard_scaler", "minmax_scaler"]

    해당 스케일러의 파라미터를 item에 dictionary(JSON) 형태로 보내야 함.

    생성된 모델은 S3 bucket의 key 경로에 name을 가진 파일로 저장됨.(pickle)
    ```
    Args:
    ```
    item  (Request, required): JSON, 파라미터
    name  (str    , required): 생성한 스케일러를 저장할 파일명
    bucket(str    , required): 버켓을 함수에 내장해야 하는지, 쿼리로 날려야 하는지 논의 필요!
    key   (str    , required): 키를 생성해서 리턴해야 하는 지 논의 필요!
    scaler(str    , required): 스케일러 1개
    ```
    Returns:
    ```
    str: 생성 완료 메시지 +  (opt.저장된 위치 key를 해시로 내부에서 생성해서 리턴?))
    ```
    """
    params = await item.json()

    if scaler not in SCALERS:
        return f'"scaler" should be in {list(SCALERS)}. current {scaler}'

    # scalers[scaler](**params[scaler])
    x = SCALERS[scaler](**params)

    # # 테스트용
    # name   = "test_pipe.pickle"
    # bucket = "aiplay-test-bucket"
    # key    = "test"

    # AWS S3 에 pickle 저장
    key = key+"/"+name
    s3_model_save(bucket, key, x)

    return f"Generation Complete: {x}"


async def make_model(
    item  : Request,
    name  : str, # 저장 될 이름.
    bucket: str,
    key   : str, # 키를 생성해서 리턴해야 하는지 생각중입니다!
    model : str,
) -> str:
    """
    ```
    아래 모델 중 하나를 선택해서 객체를 생성함.
    LinearRegression, LogisticRegression

    model 명은 반드시 아래 리스트에 포함되어 있어야 함.
    ["linear_regression", "logistic_regression"]

    해당 모델의 파라미터를 item에 dictionary(JSON) 형태로 보내야 함.

    생성된 모델은 S3 bucket의 key 경로에 name을 가진 파일로 저장됨.(pickle)
    ```
    Args:
    ```
    item  (Request, required): JSON, 파라미터
    name  (str    , required): 생성한 모델를 저장할 파일명
    bucket(str    , required): 버켓을 함수에 내장해야 하는지, 쿼리로 날려야 하는지 논의 필요!
    key   (str    , required): 키를 생성해서 리턴해야 하는 지 논의 필요!
    model (str    , required): 모델 1개
    ```
    Returns:
    ```
    str: 생성 완료 메시지 +  (opt.저장된 위치 key를 해시로 내부에서 생성해서 리턴?))
    ```
    """
    params = await item.json()
    
    if model not in MODELS:
        return f'"model" should be in {list(MODELS)}. current {model}'
    
    # models[model](**params[model])
    x = MODELS[model](**params)

    # # 테스트용
    # name   = "test_pipe.pickle"
    # bucket = "aiplay-test-bucket"
    # key    = "test"

    # AWS S3 에 pickle 저장
    key = key+"/"+name
    s3_model_save(bucket, key, x)

    return f"Generation Complete: {x}"


async def make_pipeline(
    item   : Request,
    name   : str, # 저장 될 이름.
    bucket : str,
    key    : str, # 키를 생성해서 리턴해야 하는지 생각중입니다!
    *,
    encoder: Optional[str] = Query(None,    max_length=50),
    scaler : Optional[str] = Query(None,    max_length=50),
    model  : Optional[str] = Query(None,    max_length=50),
    memory : Optional[str] = Query(None,    max_length=50),
    verbose: Optional[str] = Query("false", max_length=50),
) -> str:
    """
    ```
    
    아래의 인코더 리스트 중 여러개를 선택함(쉼표로 구분된 어레이)
    ["onehot_encoder", "ordinal_encoder", "target_encoder"]

    아래의 스케일러 중 하나를 선택함
    ["standard_scaler", "minmax_scaler"]

    아래 모델 중 하나를 선택함
    ["linear_regression", "logistic_regression"]

    해당 모델의 파라미터를 item에 dictionary(JSON) 형태로 보내야 함.
    params = {
        "encoders":{
            encoder1:{
                ...
            }, 
            encoder2:{
                ...
            }
        },
        "scaler":{
            ...
        },
        "model":{
            ...
        },
    }
    선택된 인코더, 스케일러, 모델로 파이프를 생성,
    생성된 pipe는 S3 bucket의 key 경로에 name을 가진 파일로 저장됨.(pickle)
    ```
    Args:
    ```
    item   (Request, required): JSON, 파라미터
    name   (str,     required): 생성한 모델를 저장할 파일명
    bucket (str,     required): 버켓을 함수에 내장해야 하는지, 쿼리로 날려야 하는지 논의 필요!
    key    (str,     required): 키를 생성해서 리턴해야 하는 지 논의 필요!
    *
    encoder(str,     optional): Defaults = None,    쉼표로 구분된 인코더 이름 어레이
    scaler (str,     optional): Defaults = None,    스케일러 이름(1개)
    model  (str    , optional): Defaults = None,    모델 이름(1개)
    memory (str,     optional): Defaults = None,    ??? 사용할 일 없어 보임.
    verbose(str    , optional): Defaults = "false", true, false 이 방식으로는 당장은 의미가 없어 보임.
    ```
    Returns:
    ```
    str: 생성 완료 메시지 +  (opt.저장된 위치 key를 해시로 내부에서 생성해서 리턴?))
    ```
    """
    encoder = None    if encoder == "" else encoder
    scaler  = None    if scaler  == "" else scaler
    model   = None    if model   == "" else model
    memory  = None    if memory  == "" else memory
    verbose = "false" if verbose == "" else verbose
    
    params = await item.json()

    ## validation check
    if encoder is not None:
            try   : 
                encoder = [i.strip() for i in encoder.split(",") if i.strip() != ""]
                if not set(encoder) <= set(ENCODERS):
                    return f'"encoder" should be in {list(ENCODERS)}. current {encoder}'
            except: return '"encoder" should be array(column names) divied by ","'
    
    if scaler is not None and scaler not in SCALERS:
        return f'"scaler" should be in {list(SCALERS)}. current {scaler}'
    
    if model is not None and model not in MODELS:
        return f'"model" should be in {list(MODELS)}. current {model}'

    if not (encoder or scaler or model):
        return "At least an encoder, a scaler or a model is needed."

    verbose = boolean(verbose)
    if verbose is None: return '"verbose" should be bool, "true" or "false"'

    # 테스트용
    # encoder = ["onehot_encoder"]
    # scaler = "standard_scaler"
    # model = "logistic_regression"
    # params = {
    #     "encoders":{
    #         "onehot_encoder":{
    #             "verbose"       :0, 
    #             "cols"          :["Sex", "Pclass", "Embarked"], 
    #             "drop_invariant":False, 
    #             "return_df"     :True,
    #             "handle_missing":'value', 
    #             "handle_unknown":'value', 
    #             "use_cat_names" :True
    #         },
    #         "target_encoder":{
                
    #         },
    #     },
    #     scaler:{
    #         "copy"     :True, 
    #         "with_mean":True, 
    #         "with_std" :True
    #     },
    #     model:{
    #         "penalty"          :"l2",
    #         "dual"             :False,
    #         "tol"              :1e-4,
    #         "C"                :1.0,
    #         "fit_intercept"    :True,
    #         "intercept_scaling":1,
    #         "class_weight"     :None,
    #         "random_state"     :None,
    #         "solver"           :"lbfgs",
    #         "max_iter"         :100,
    #         "multi_class"      :"auto",
    #         "verbose"          :0,
    #         "warm_start"       :False,
    #         "n_jobs"           :None,
    #         "l1_ratio"         :None,
    #     },
    # }
    

    steps = []
    if encoder is not None: 
        for i in encoder:
            try: steps.append((i, ENCODERS[i](**params["encoders"][i]))) 
            except: return "올바르지 않은 인코더 이름"
    
    if scaler is not None:
        try: steps.append((scaler, SCALERS[scaler](**params[scaler])))
        except: return "올바르지 않은 스케일러 이름"
    
    if model is not None:
        try: steps.append((model, MODELS[model](**params[model])))
        except: return "올바르지 않은 모델 이름"

    pipe = Pipeline(
        steps   = steps,
        memory  = memory,
        verbose = verbose
    )

    # # 테스트용
    # name   = "test_pipe.pickle"
    # bucket = "aiplay-test-bucket"
    # key    = "test"

    # ## AWS S3 에 pickle 저장
    key = key+"/"+name
    s3_model_save(bucket, key, pipe)

    # ## 로컬에 저장(테스트용)
    # with open("test_pipe.pickle", "wb") as f:
    #     pickle.dump(pipe, f)

    return f"Generated Model: {pipe}"

