import logging
import os
from pathlib import Path
import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sklearn.utils.validation import check_is_fitted
from pipeline_def import FEATURES

BUNDLE = None
try:
    loaded = joblib.load(Path(os.environ.get('MODEL_PATH', Path(__file__).with_name('pipeline.joblib'))))
    if loaded['metadata']['sklearn_version'] != sklearn.__version__:
        raise ValueError('Artifact/runtime sklearn version mismatch')
    check_is_fitted(loaded['pipeline'].named_steps['classifier'])
    if loaded['metadata']['expected_input_features'] != FEATURES:
        raise ValueError('Artifact feature schema mismatch')
    BUNDLE = loaded
except Exception:
    logging.exception('Model unavailable')

app = FastAPI(title='Injury Predict — fitted ML API', version='1.0.0')
origins = [o.strip() for o in os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000,http://localhost:5173').split(',') if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['GET','POST'], allow_headers=['Content-Type'])

class InjuryInput(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    Player_Age: int = Field(ge=18, le=39, description='Player age, assumed years')
    Player_Weight: float = Field(ge=40.191, le=104.651, description='Player weight, assumed kg')
    Player_Height: float = Field(ge=145.285, le=207.309, description='Player height, assumed cm')
    Previous_Injuries: int = Field(ge=0, le=1, description='Binary previous injury indicator')
    Training_Intensity: float = Field(ge=0, le=1, description='Normalized intensity score')
    Recovery_Time: int = Field(ge=1, le=6, description='Recovery time in dataset units; physical unit undocumented')

def require_model():
    if BUNDLE is None:
        raise HTTPException(503, 'Fitted model artifact is unavailable')
    return BUNDLE

@app.get('/health')
def health():
    require_model()
    return {'status':'ok','model_loaded':True}

@app.get('/pipeline')
def pipeline_metadata():
    return require_model()['metadata']

@app.post('/predict')
def predict(payload: InjuryInput):
    bundle = require_model()
    frame = pd.DataFrame([payload.model_dump()], columns=FEATURES)
    model = bundle['pipeline']
    prediction = int(model.predict(frame)[0])
    index = list(model.classes_).index(1)
    probability = float(model.predict_proba(frame)[0, index])
    return {'prediction':prediction,'label':bundle['metadata']['classes'][str(prediction)],'injury_probability':probability,'target':bundle['metadata']['target'],'model':'LogisticRegression','built_at':bundle['metadata']['built_at']}
