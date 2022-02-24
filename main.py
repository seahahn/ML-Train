from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000" # 포트 지정 안 하면 CORS 에러 발생
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from functions import (
    model_transform,
    model_fit_transform,
    model_fit,
    model_predict,
    model_score,
    model_fit_predict,
    model_predict_score,
    model_fit_predict_score,

    make_encoder,
    make_scaler,
    make_model,
    make_pipeline,
)

model_transform         = app.post("/model/transform")        (model_transform)
model_fit_transform     = app.post("/model/fit_transform")    (model_fit_transform)
model_fit               = app.post("/model/fit")              (model_fit)
model_predict           = app.post("/model/predict")          (model_predict)
model_score             = app.post("/model/score")            (model_score)
model_fit_predict       = app.post("/model/fit_predict")      (model_fit_predict)
model_predict_score     = app.post("/model/predict_score")    (model_predict_score)
model_fit_predict_score = app.post("/model/fit_predict_score")(model_fit_predict_score)


make_encoder  = app.post("/model/make_encoder") (make_encoder)
make_scaler   = app.post("/model/make_scaler")  (make_scaler)
make_model    = app.post("/model/make_model")   (make_model)
make_pipeline = app.post("/model/make_pipeline")(make_pipeline)