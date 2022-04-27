from fastapi import FastAPI
from joblib import load
import pandas as pd
from pydantic import BaseModel
app = FastAPI()


class DataModel(BaseModel):

    adult_mortality: float
    infant_deaths: float
    alcohol: float
    percentage_expenditure: float
    hepatitis_B: float
    measles: float
    bmi: float
    under_five_deaths: float
    polio: float
    total_expenditure: float
    diphtheria: float
    hiv_aids: float
    gdp: float
    population: float
    thinness_10_19_years: float
    thinness_5_9_years: float
    income_composition_of_resources: float
    schooling: float


def columns():
    return ["Adult Mortality", "infant deaths", "Alcohol", "percentage expenditure", "Hepatitis B", "Measles",
            "BMI", "under-five deaths", "Polio", "Total expenditure", "Diphtheria", "HIV/AIDS", "DGP", "Population",
            "thinness 10-19 years", "thinness 5-9 years", "Income composition of resources", "Schooling"]


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def make_predictions(data_model: DataModel):
    df = pd.DataFrame(data_model.dict(), columns=data_model.dict().keys(), index=[0])
    df.columns = columns()
    model = load("assets/modelo.joblib")
    result = model.predict(df)
    return result.tolist()
