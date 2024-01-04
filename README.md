# AI Play ML-Train

API server for model training, one of the machine learning-related functions.

## :one: Stack

- Python 3.8.12
- FastAPI 0.73.0
- Pandas 1.4.1
- scikit-learn 1.0.2
- JWT
- Swagger

<br/>

## 2️⃣ Deployment Platform and Server Address

- Platform: Heroku
- Address: [https://aiplay-mltrain.herokuapp.com/](https://aiplay-mltrain.herokuapp.com/)

<br/>

## :three: API Specification

- DOCS: [https://aiplay-mltrain.herokuapp.com/docs](https://aiplay-mltrain.herokuapp.com/docs)

<details>
  <summary>Expand</summary>

| Method | URL                      | Description                                                                             |
| ------ | ------------------------ | --------------------------------------------------------------------------------------- |
| GET    | /model/steps             | Display the list of model pipeline components (encoder, scaler, model) with step titles |
| GET    | /model/steps_detail      | Display detailed information about model pipeline components (encoder, scaler, model)   |
| POST   | /model/transform         | Display the DataFrame transformation result by encoder and scaler                       |
| POST   | /model/fit_transform     | Display the DataFrame transformation result after model training                        |
| POST   | /model/fit               | Train the model                                                                         |
| POST   | /model/predict           | Generate target predictions from the model                                              |
| POST   | /model/score             | Measure the prediction performance of the model                                         |
| POST   | /model/fit_predict       | Train the model and generate target predictions                                         |
| POST   | /model/predict_score     | Generate target predictions and measure prediction performance                          |
| POST   | /model/fit_predict_score | Train the model, generate predictions, and measure model performance                    |
| POST   | /model/make_encoder      | Create and save encoder object                                                          |
| POST   | /model/make_scaler       | Create and save scaler object                                                           |
| POST   | /model/make_model        | Create and save model object                                                            |
| POST   | /model/make_pipeline     | Create and save model pipeline object                                                   |
| POST   | /model/make_optimizer    | Create and save model optimizer object                                                  |

</details>

<br/>

## :four: Troubleshooting Records

- [https://github.com/AI-Play/ML-Train/discussions](https://github.com/AI-Play/ML-Train/discussions)

<br/>

## :five: Development Environment Preparation

<details>
  <summary>Expand</summary>

```
// Create a new virtual environment
// 1. Move to the directory which has python version we need to use
// 2. Create a new virtual environment
python -m venv /path/to/new/virtual/environment

// 3. Activate the virtual environment
source /path/to/new/virtual/environment/bin/activate

// 4. Install required packages
pip install -r requirements.txt
```

##### Run

```
uvicorn main:app --reload
```

</details>
