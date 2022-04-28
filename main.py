from fastapi import Request, FastAPI
from sklearn.metrics import r2_score
import pandas as pd
import json
import Models


app = FastAPI()
ML_model = Models.LinearRegressionModel()


@app.post("/predict")
def make_predictions(prediction_model: Models.DataModel):
    df = pd.DataFrame(prediction_model.dict(), columns=prediction_model.dict().keys(), index=[0])
    df.columns = Models.prediction_columns()
    result = ML_model.make_predictions(df)
    return result.tolist()


@app.post("/r2")
async def calculate_r2(request_body: Request):
    req_info = await request_body.json()
    json_string = json.dumps([ob for ob in req_info])
    df = pd.read_json(json_string)
    df.columns = Models.r2_columns()

    x = df.drop('Life expectancy', axis=1)
    y = df['Life expectancy']

    y_prediction = ML_model.make_predictions(x)
    r2 = r2_score(y, y_prediction)

    return "'R²: %.2f" % r2


@app.get("/")
def read_root():
    return {"Hello": "World"}
