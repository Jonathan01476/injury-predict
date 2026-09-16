import importlib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
import serve
from pipeline_def import InjuryFeatureEngineer

PAYLOAD = dict(Player_Age=28, Player_Weight=75, Player_Height=180, Previous_Injuries=0, Training_Intensity=.5, Recovery_Time=4)

def test_real_pipeline_response():
    client = TestClient(serve.app)
    assert client.get('/health').status_code == 200
    metadata = client.get('/pipeline').json()
    assert metadata['sklearn_version'] == '1.7.2'
    response = client.post('/predict', json=PAYLOAD)
    assert response.status_code == 200
    frame = pd.DataFrame([PAYLOAD])
    assert response.json()['prediction'] == int(serve.BUNDLE['pipeline'].predict(frame)[0])
    assert response.json()['injury_probability'] == pytest.approx(serve.BUNDLE['pipeline'].predict_proba(frame)[0,1])
    assert client.get('/docs').status_code == 200

@pytest.mark.parametrize('feature,value', [('Player_Age',99),('Player_Weight',0),('Player_Height',400),('Previous_Injuries',2),('Training_Intensity',1.5),('Recovery_Time',0),('Recovery_Time',2.5)])
def test_out_of_range(feature,value):
    assert TestClient(serve.app).post('/predict', json={**PAYLOAD,feature:value}).status_code == 422

def test_missing_and_extra():
    client = TestClient(serve.app)
    assert client.post('/predict',json={}).status_code == 422
    assert client.post('/predict',json={**PAYLOAD,'unknown':1}).status_code == 422

def test_artifact_unavailable(monkeypatch):
    monkeypatch.setattr(serve,'BUNDLE',None)
    client = TestClient(serve.app)
    for path in ['/health','/pipeline']:
        assert client.get(path).status_code == 503
    assert client.post('/predict',json=PAYLOAD).status_code == 503

def test_missing_and_corrupt_import(tmp_path,monkeypatch):
    for path in [tmp_path/'missing.joblib',tmp_path/'corrupt.joblib']:
        if path.name.startswith('corrupt'):
            path.write_bytes(b'not a joblib model')
        monkeypatch.setenv('MODEL_PATH',str(path))
        importlib.reload(serve)
        assert TestClient(serve.app).get('/health').status_code == 503
    monkeypatch.delenv('MODEL_PATH')
    importlib.reload(serve)

def test_transformer_learns_only_training_medians():
    data = pd.DataFrame([PAYLOAD,{**PAYLOAD,'Player_Weight':85}])
    transformer = InjuryFeatureEngineer().fit(data)
    missing = pd.DataFrame([{**PAYLOAD,'Player_Weight':np.nan}])
    transformed = transformer.transform(missing)
    assert transformed.Player_Weight.iloc[0] == 80
    assert transformed.Body_Size_Index.iloc[0] == pytest.approx(80/1.8**2)
    assert transformed.Load_Recovery_Ratio.iloc[0] == .125
