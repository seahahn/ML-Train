import pickle, boto3, io


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


import psycopg2
def save_log(query):
    # params = json.load(".env")
    # db = psycopg2.connect(
    #     **params
    # )
    # cursor = db.cursor()
    # insert_into = """INSERT INTO {schema}.{table}({column}) VALUES ('{data}')"""

    # cursor.execute(query)
    print(query)


from typing import Optional
from fastapi import Header
import datetime, inspect
import traceback

def check_error(func):
    async def wrapper(*args, user_id: Optional[str] = Header(None), **kwargs):
        name = func.__name__
        start = datetime.datetime.now()
        print(user_id)
        try:
            tf, return_value = await func(*args, **kwargs)
            end = datetime.datetime.now()
            is_worked = 0 if tf else 1
            
            query = """INSERT INTO {}.{}({}) VALUES ("{}")"""
            save_log(query)
            return return_value
        except:
            print(traceback.format_exc())
            end = datetime.datetime.now()
            # Unexpected error
            query = """비정상적인 동작"""
            is_worked = 2
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