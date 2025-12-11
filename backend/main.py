from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.helper_fun import load_config, load_model, load_x_t 
from Preproessing.Preprocessing import Preprocessing

app = FastAPI(title="Car Price Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG = load_config()
MODEL_NAME = CONFIG['Model']['model_name']

@app.on_event("startup")
def load_artifacts():
    global MODEL, POLY, SCALER, ENCODERS
    try:
        MODEL, POLY, SCALER, ENCODERS = load_model(MODEL_NAME, 'test')
    except Exception as e:
        raise RuntimeError(f"Failed to load model artifacts: {e}")

class CarFeatures(BaseModel):
    Manufacturer: str
    Model: str
    Category: str
    Leather_interior: Optional[str] = Field(None, alias="Leather interior")
    Fuel_type: str = Field(..., alias="Fuel type")
    Gear_box_type: str = Field(..., alias="Gear box type")
    Drive_wheels: str = Field(..., alias="Drive wheels")
    Doors: str
    Wheel: str
    Color: str
    Price: Optional[float] = 0.0  
    Levy: Optional[float] = 0.0
    Engine_volume: Optional[float] = Field(None, alias="Engine volume")
    Mileage: Optional[int] = 0
    Cylinders: Optional[int] = None
    Prod_year: Optional[int] = Field(None, alias="Prod. year")
    Airbags: Optional[int] = None
    Random_notes: Optional[str] = None

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "Manufacturer": "Toyota",
                "Model": "Corolla",
                "Category": "Sedan",
                "Leather interior": "Yes",
                "Fuel type": "Petrol",
                "Gear box type": "Automatic",
                "Drive wheels": "Front",
                "Doors": "4",
                "Wheel": "Left",
                "Color": "White",
                "Price": 0,
                "Levy": 0,
                "Engine volume": 1.6,
                "Mileage": 50000,
                "Cylinders": 4,
                "Prod. year": 2018,
                "Airbags": 6
            }
        }

def _format_raw_df(input: CarFeatures) -> pd.DataFrame:
    """
    Format a single-row raw dataframe that matches expected raw CSV formats
    used by the Preprocessing.fix_data_type() method.
    The preprocessing code expects strings for some columns (e.g., 'Mileage' with 'KM', 'Engine volume' as str, 'Levy' possibly '-')
    We convert user-friendly inputs into the raw-ish form that your Preprocessing expects.
    """
    d = {}

    d['Price'] = f"${int(input.Price or 0)}"

    levy_val = input.Levy if input.Levy is not None else 0
    d['Levy'] = str(int(levy_val))

    mileage_val = int(input.Mileage or 0)
    d['Mileage'] = f"{mileage_val}KM"

    d['Prod. year'] = str(int(input.Prod_year)) if input.Prod_year is not None else 'unknown'

    d['Cylinders'] = int(input.Cylinders) if input.Cylinders is not None else pd.NA

    if input.Engine_volume is not None:
        d['Engine volume'] = str(input.Engine_volume)
    else:
        d['Engine volume'] = pd.NA

    d['Manufacturer'] = input.Manufacturer
    d['Model'] = input.Model
    d['Category'] = input.Category
    d['Leather interior'] = input.__dict__.get('Leather interior') or input.Leather_interior or 'No'
    d['Fuel type'] = input.__dict__.get('Fuel type') or input.Fuel_type
    d['Gear box type'] = input.__dict__.get('Gear box type') or input.Gear_box_type
    d['Drive wheels'] = input.__dict__.get('Drive wheels') or input.Drive_wheels
    d['Doors'] = str(input.Doors)
    d['Wheel'] = input.Wheel
    d['Color'] = input.Color
    d['Airbags'] = int(input.Airbags) if input.Airbags is not None else pd.NA
    d['Random_notes'] = input.Random_notes or ""

    return pd.DataFrame([d])

@app.post("/predict")
def predict(features: CarFeatures):
    try:
        raw_df = _format_raw_df(features)

        prep = Preprocessing(raw_df)
        prep.encoders = ENCODERS
        if "_impute" in ENCODERS:
            prep._impute = ENCODERS.get("_impute", None)
        if "_outlier_bounds" in ENCODERS:
            prep._outlier_bounds = ENCODERS.get("_outlier_bounds", None)

        df_prepared = prep.prepare_data(fit_encoders=False)

        _, x, _ = load_x_t(df_prepared)

        x_transformed = prep.transform(x, POLY, SCALER)

        pred = MODEL.predict(x_transformed)
        if isinstance(pred, (list, np.ndarray)):
            pred_val = float(pred[0])
        else:
            pred_val = float(pred)

        apply_log = ENCODERS.get("_apply_log", False)
        if apply_log:
            price_pred = float(np.exp(pred_val))
        else:
            price_pred = float(pred_val)

        return {
            "predicted_price": price_pred,
            "currency": "USD (same unit as original Price column)",
            "raw_prediction_model_output": pred_val
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metadata")
def metadata():
    """
    Return expected categorical values from the saved encoders (helpful for frontend select boxes).
    """
    try:
        cats = {}
        for k, v in ENCODERS.items():
            if isinstance(v, (list, tuple, dict)):
                continue
            try:
                classes = list(v.classes_)
                cats[k] = classes
            except Exception:
                pass
        return {"categorical_classes": cats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
