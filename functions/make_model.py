from typing import Optional
from fastapi import Request, Query

# import json
# import numpy as np
from scipy.stats import randint, uniform
# import pandas as pd
# import modin.pandas as pd

from sklearn.pipeline import Pipeline


from .internal_func import (
    s3_model_save, 
    s3_model_load, 
    boolean,
    isint,
    check_error,
    ENCODERS,
    SCALERS,
    MODELS,
    METRICS,
    OPTIMIZERS,
)


@check_error
async def make_encoder(
    item   : Request,
    name   : str, # 저장 될 이름.
    key    : str, # 키를 생성해서 리턴해야 하는지 생각중입니다!
    encoder: str,
) -> tuple:
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
        return False, {"result": False, "message": f'"encoder" should be in {list(ENCODERS)}. current {encoder}'}
    
    # encoders[encoder](**params["encoders"][encoder])
    # encoders[encoder](**params[encoder])
    x = ENCODERS[encoder](**params)

    # # 테스트용
    # name   = "test_pipe.pickle"
    # key    = "test"

    # AWS S3 에 pickle 저장
    key = key+"/"+name
    s3_model_save(key, x)

    return True, {"result": True, "message":f"Generation Complete: {x}"}


@check_error
async def make_scaler(
    item  : Request,
    name  : str, # 저장 될 이름.
    key   : str, # 키를 생성해서 리턴해야 하는지 생각중입니다!
    scaler: str,
) -> tuple:
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
        return False, {"result": False, "message": f'"scaler" should be in {list(SCALERS)}. current {scaler}'}

    # scalers[scaler](**params[scaler])
    x = SCALERS[scaler](**params)

    # # 테스트용
    # name   = "test_pipe.pickle"
    # key    = "test"

    # AWS S3 에 pickle 저장
    key = key+"/"+name
    s3_model_save(key, x)

    return True, {"result": True, "message": f"Generation Complete: {x}"}



@check_error
async def make_model(
    item  : Request,
    name  : str, # 저장 될 이름.
    key   : str, # 키를 생성해서 리턴해야 하는지 생각중입니다!
    model : str,
) -> tuple:
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
        return False, {"result": False, "message": f'"model" should be in {list(MODELS)}. current {model}'}
    
    # models[model](**params[model])
    x = MODELS[model](**params)

    # # 테스트용
    # name   = "test_pipe.pickle"
    # key    = "test"

    # AWS S3 에 pickle 저장
    key = key+"/"+name
    s3_model_save(key, x)

    return True, {"result": True, "message": f"Generation Complete: {x}"}


@check_error
async def make_pipeline(
    item   : Request,
    name   : str, # 저장 될 이름.
    key    : str, # 키를 생성해서 리턴해야 하는지 생각중입니다!
    *,
    encoder: Optional[str] = Query(None,    max_length=50),
    scaler : Optional[str] = Query(None,    max_length=50),
    model  : Optional[str] = Query(None,    max_length=50),
    memory : Optional[str] = Query(None,    max_length=50),
    verbose: Optional[str] = Query("false", max_length=50),
) -> tuple:
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
                    return False, {"result": False, "message": f'"encoder" should be in {list(ENCODERS)}. current {encoder}'}
            except: return False, {"result": False, "message": '"encoder" should be array(column names) divied by ","'}
    
    if scaler is not None and scaler not in SCALERS:
        return False, {"result": False, "message": f'"scaler" should be in {list(SCALERS)}. current {scaler}'}
    
    if model is not None and model not in MODELS:
        return False, {"result": False, "message": f'"model" should be in {list(MODELS)}. current {model}'}

    if not (encoder or scaler or model):
        return False, {"result": False, "message": "At least an encoder, a scaler or a model is needed."}

    verbose = boolean(verbose)
    if verbose is None: return False, {"result": False, "message": '"verbose" should be bool, "true" or "false"'}

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
            except: return False, {"result": False, "message": "올바르지 않은 인코더 이름"}
    
    if scaler is not None:
        try: steps.append((scaler, SCALERS[scaler](**params["scaler"])))
        except: return False, {"result": False, "message": "올바르지 않은 스케일러 이름"}
    
    if model is not None:
        try: steps.append((model, MODELS[model](**params["model"])))
        except: return False, {"result": False, "message": "올바르지 않은 모델 이름"}

    pipe = Pipeline(
        steps   = steps,
        memory  = memory,
        verbose = verbose
    )

    # # 테스트용
    # name   = "test_pipe.pickle"
    # key    = "test"

    # ## AWS S3 에 pickle 저장
    key = key+"/"+name
    s3_model_save(key, pipe)

    # ## 로컬에 저장(테스트용)
    # with open("test_pipe.pickle", "wb") as f:
    #     pickle.dump(pipe, f)

    return True, {"result": True, "message": f"Generated Model: {pipe}"}


@check_error
async def make_optimizer(
    item              : Request,
    name              : str, # 불러올 모델 이름.
    key               : str, 
    optimizer         : str,
    *,
    save_name         : Optional[str] = Query(None,    max_length=50), # 다른 이름으로 저장
    n_iter            : Optional[str] = Query(10,      max_length=50),
    scoring           : Optional[str] = Query(None,    max_length=50),
    n_jobs            : Optional[str] = Query(None,    max_length=50), # 멀티 유저를 상정하기 때문에 1로 고정하는 것이 어떨까요?
    cv                : Optional[str] = Query(5,       max_length=50), # None, to use the default 5-fold cross validation
    random_state      : Optional[str] = Query(None,    max_length=50),
    return_train_score: Optional[str] = Query("false", max_length=50),
    # refit
    # verbose
    # pre_dispatch
    # error_score
) -> tuple:

    if optimizer not in OPTIMIZERS:
        return False, "지원되지 않는 optimizer"

    save_name          = None    if save_name          == "" else save_name
    n_iter             = 10      if n_iter             == "" else n_iter
    scoring            = None    if scoring            == "" else scoring
    n_jobs             = None    if n_jobs             == "" else n_jobs
    cv                 = 5       if cv                 == "" else cv
    random_state       = None    if random_state       == "" else random_state
    return_train_score = "false" if return_train_score == "" else return_train_score
    
    # n_iter
    if isint(n_iter): n_iter = int(n_iter)
    else            : return False, {"result": False, "message":"n_iter는 정수만 가능!"}

    # scoring
    if scoring is not None and scoring not in METRICS:
        return False, {"result": False, "message": f"scoring은 반드시 {list(METRICS)}안에 포함되어야 함!"}

    # n_jobs
    if isint(n_jobs): n_jobs = int(n_jobs)
    else            : return False, {"result": False, "message": "n_jobs는 정수만 가능!"}

    # cv
    if isint(cv): cv = int(cv)
    else        : return False, {"result": False, "message": "cv는 정수만 가능!"}

    # random_state
    if random_state is not None:
        if isint(random_state): random_state = int(random_state)
        else                  : return False, {"result": False, "message": "random_state는 정수만 가능!"}

    # return_train_score
    return_train_score = boolean(return_train_score)
    if return_train_score is None:
        return False, {"result": False, "message": "return_train_score should be true or false"}

    # 사용하지 않는 파라미터
    # refit
    # verbose
    # pre_dispatch
    # error_score
    {
        "onehot_encoder":{
            "param1":...,
            "param2":...
        },
        "standard_scaler":{
            "param1":...,
        }
    }

    {
        "onehot_encoder__cat_use_named": True,
        # "onehot_encoder__columns": [[...], [...], [...]],
        "standard_scaler__params1": "0,1,2",
        "encoder__max": "_randint,10,100",
        "encoder__max": "_randexp,10,-10,-4"
    }

    i_params:dict = await item.json()
    params:dict = {}
    for model_name, model_params in i_params.items():
        for param_name, param in model_params.items():
            i = model_name + "__" + param_name
            if   param[0] == "_randint": # ex) "randint,-3,3" => [-3,-2,-1,0,1,2,3]
                params[i] = randint(int(param[1]),int(param[2]))

            elif param[0] == "_randexp": # ex) "randexp,10,-3,3" => [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                params[i] = [int(param[1])**x.rvs() for x in uniform(int(param[2]),int(param[3]) - int(param[2]))]

            elif param[0] == "_uniform": # ex) "uniform,0,1" => 0.0 ~ 1.0 사이의 값을 균일 확률로...?
                params[i] = uniform(int(param[1]),int(param[2]) - int(param[1]))
                
            elif param[0] == "_discrete":
                try: # 숫자형일 경우 숫자형 리스트로 변환
                    params[i] = [float(k) for k in param[1].split(",")]
                except:
                    return False, {"result": False, "message": "입력된 값이 숫자가 아닙니다."}
            else:
                params[i] = param

    print(params)

    kwargs = {
        "n_iter"            : n_iter,
        "scoring"           : scoring,
        "n_jobs"            : n_jobs,
        "cv"                : cv,
        "random_state"      : random_state,
        "return_train_score": return_train_score,
        # "refit"             : True,
        # "verbose"           : 0,
        # "pre_dispatch"      : "2*n_jobs",
        # "error_score"       : np.nan,
    }

    if optimizer == "grid_search_cv":
        del kwargs["n_iter"]
    
    key = key+"/"+name
    pipe = s3_model_load(key)

    op_model = OPTIMIZERS[optimizer](
        estimator           = pipe,
        param_distributions = params,
        **kwargs        
    )
    
    if save_name is not None:
        key = key+"/"+save_name
    
    s3_model_save(key, op_model)

    return True, {"result": True, "message": f"Generated Optimizer: {op_model}"}

