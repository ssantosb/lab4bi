from joblib import load
from pydantic import BaseModel


class LinearRegressionModel:

    def __init__(self):
        self.model = load("assets/modelo.joblib")

    def make_predictions(self, data):
        result = self.model.predict(data)
        return result


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


def prediction_columns():
    return ["Adult Mortality", "infant deaths", "Alcohol", "percentage expenditure", "Hepatitis B", "Measles",
            "BMI", "under-five deaths", "Polio", "Total expenditure", "Diphtheria", "HIV/AIDS", "DGP", "Population",
            "thinness 10-19 years", "thinness 5-9 years", "Income composition of resources", "Schooling"]


def r2_columns():
    return ["Life expectancy", "Adult Mortality", "infant deaths", "Alcohol", "percentage expenditure", "Hepatitis B",
            "Measles", "BMI", "under-five deaths", "Polio", "Total expenditure", "Diphtheria", "HIV/AIDS", "DGP",
            "Population", "thinness 10-19 years", "thinness 5-9 years", "Income composition of resources", "Schooling"]

