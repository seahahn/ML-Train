import pickle, boto3, io, json, os
from dotenv import load_dotenv
load_dotenv()


from sklearn.impute import SimpleImputer, KNNImputer
IMPUTER = {
    "simple_imputer": SimpleImputer,
    "knn_imputer"   : KNNImputer,
}


from category_encoders import OneHotEncoder, OrdinalEncoder, TargetEncoder
ENCODERS = {
    "onehot_encoder" : OneHotEncoder,
    "ordinal_encoder": OrdinalEncoder,
    "target_encoder" : TargetEncoder
}


from sklearn.preprocessing import StandardScaler, MinMaxScaler
SCALERS = {
    "standard_scaler": StandardScaler,
    "minmax_scaler"  : MinMaxScaler
}


from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    # RidgeClassifier,
    Ridge,
    # Lasso,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
MODELS = {
    "linear_regression"       : LinearRegression,
    "logistic_regression"     : LogisticRegression,
    "ridge"                   : Ridge,
    "decision_tree_classifier": DecisionTreeClassifier,
    "decision_tree_regressor" : DecisionTreeRegressor,
    "random_forest_classifier": RandomForestClassifier,
    "random_forest_regressor" : RandomForestRegressor,
}


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

METRICS = {
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


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
OPTIMIZERS = {
    "randomized_search_cv": RandomizedSearchCV,
    "grid_search_cv"      : GridSearchCV,
}

BUCKET = "aiplay-test-bucket"

def s3_model_save(key, body):
    s3 = boto3.client('s3', aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))
    s3.put_object(
        Bucket = BUCKET,
        Key    = key,
        Body   = pickle.dumps(body)
    )


def s3_model_load(key):
    s3 = boto3.client('s3', aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))
    file = io.BytesIO(
        s3.get_object(
            Bucket = BUCKET,
            Key    = key
        )["Body"].read()
    )
    return pickle.load(file)


def boolean(x) -> bool:
    if   x.lower() == "true" : return True
    elif x.lower() == "false": return False


def isint(x:str) -> bool:
    if type(x) == str:
        if x.isnumeric(): return True
        else            : return False
    elif type(x) == int:
        return True
    else:
        return False


# user_idx INTEGER NOT NULL, -- 사용자 고유번호
# func_code VARCHAR(255) NOT NULL, -- 함수 기능 코드
# is_worked INTEGER NOT NULL, -- 정상 작동 여부
# error_msg TEXT NOT NULL DEFAULT '', -- Unexpected Error일 경우 저장되는 에러 메시지
# start_time TIMESTAMP NOT NULL, -- 작동 시작 시각
# end_time TIMESTAMP NOT NULL, -- 작동 종료 시각
# created_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW() -- 생성 시점

import psycopg2

def save_log(query):
    db = psycopg2.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PW"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME")
    )
    cursor = db.cursor()
    cursor.execute(query)
    print(query)
    cursor.close()
    db.commit()
    db.close()

from typing import Optional
from fastapi import Header, Cookie
import datetime, inspect, traceback, jwt

SECRET_KEY=os.getenv("SECRET_KEY")

def check_error(func):
    async def wrapper(*args, user_id: Optional[str] = Header(None), token: Optional[str] = Header(None), **kwargs):
        try:
            # 토큰을 검증하여 유효한 토큰인지 확인
            # JWT 토큰 인증되지 않으면 기능 작동 X (정상적인 사용자가 아닌 것으로 간주)
            at = token
            jwt.decode(at, SECRET_KEY, algorithms="HS256")
        except Exception as e:
            return {"result":False, "token_state":False, "message":str(e)}

        name = func.__name__
        start = datetime.datetime.now(datetime.timezone.utc)
        try:
            tf, return_value = await func(*args, **kwargs)
            end = datetime.datetime.now(datetime.timezone.utc)
            is_worked = 0 if tf else 1

            query = f"""INSERT INTO
                public.func_log (user_idx, func_code, is_worked, start_time, end_time)
                VALUES ({user_id},'{name}',{is_worked}, '{start}', '{end}')"""
            save_log(query)
            return return_value
        except:
            error = traceback.format_exc()
            end = datetime.datetime.now(datetime.timezone.utc)
            is_worked = 2
            # Unexpected error
            query = f"""INSERT INTO
                public.func_log (user_idx, func_code, is_worked, error_msg, start_time, end_time)
                VALUES ({user_id},'{name}',{is_worked}, '{error}', '{start}', '{end}')"""
            save_log(query)
            return traceback.format_exc()

    ## FastAPI 에서 데코레이터를 사용할 수 있도록 파라미터 수정
    wrapper.__signature__ = inspect.Signature(
        parameters = [
            # Use all parameters from function
            *inspect.signature(func).parameters.values(),
            # Skip *args and **kwargs from wrapper parameters:
            *filter(
                lambda p: p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD),
                inspect.signature(wrapper).parameters.values()
            ),
        ],
        return_annotation = inspect.signature(func).return_annotation,
    )

    # 나머지 요소를 func으로부터 가져오기
    wrapper.__module__ = func.__module__
    wrapper.__doc__  = func.__doc__
    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__

    return wrapper