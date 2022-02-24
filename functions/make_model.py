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
    # LinearRegression, 
    LogisticRegression,
    # RidgeClassifier,
    # Ridge,
    # Lasso,
)
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.ensemble import (
#     RandomForestClassifier, 
#     RandomForestRegressor,
#     AdaBoostClassifier,
#     AdaBoostRegressor,
#     GradientBoostingClassifier,
#     GradientBoostingRegressor,
# )



encoders = {
    "onehot_encoder" : OneHotEncoder,
    "ordinal_encoder": OrdinalEncoder,
    "target_encoder" : TargetEncoder
}

scalers = {
    "standard_scaler": StandardScaler,
    "minmax_scaler"  : MinMaxScaler
}

models = {
    # "linear_regression"  : LinearRegression,
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

    params = json.loads(await item.json())

    if encoder not in encoders:
        return f'"encoder" should be in {list(encoders)}. current {encoder}'
    
    # encoders[encoder](**params["encoders"][encoder])
    # encoders[encoder](**params[encoder])
    x = encoders[encoder](**params)

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

    params = json.loads(await item.json())

    if scaler not in scalers:
        return f'"scaler" should be in {list(scalers)}. current {scaler}'

    # scalers[scaler](**params[scaler])
    x = scalers[scaler](**params)

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

    params = json.loads(await item.json())
    
    if model not in models:
        return f'"model" should be in {list(models)}. current {model}'
    
    # models[model](**params[model])
    x = models[model](**params)

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
    encoder = None    if encoder == "" else encoder
    scaler  = None    if scaler  == "" else scaler
    model   = None    if model   == "" else model
    memory  = None    if memory  == "" else memory
    verbose = "false" if verbose == "" else verbose
    
    params = json.loads(await item.json())

    ## validation check
    if encoder is not None:
            try   : 
                encoder = [i.strip() for i in encoder.split(",") if i.strip() != ""]
                if not set(encoder) <= set(encoders):
                    return f'"encoder" should be in {list(encoders)}. current {encoder}'
            except: return '"encoder" should be array(column names) divied by ","'
    
    if scaler is not None and scaler not in scalers:
        return f'"scaler" should be in {list(scalers)}. current {scaler}'
    
    if model is not None and model not in models:
        return f'"model" should be in {list(models)}. current {model}'

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
            try: steps.append((i, encoders[i](**params["encoders"][i]))) 
            except: return "올바르지 않은 인코더 이름"
    
    if scaler is not None:
        try: steps.append((scaler, scalers[scaler](**params[scaler])))
        except: return "올바르지 않은 스케일러 이름"
    
    if model is not None:
        try: steps.append((model, models[model](**params[model])))
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

