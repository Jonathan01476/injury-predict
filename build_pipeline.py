import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from pipeline_def import FEATURES, InjuryFeatureEngineer

ROOT = Path(__file__).parent
TARGET = 'Likelihood_of_Injury'

def build(dataset):
    df = pd.read_csv(dataset)
    if list(df.columns) != FEATURES + [TARGET]:
        raise ValueError('Unexpected dataset schema')
    if not set(df[TARGET].unique()) == {0, 1}:
        raise ValueError('Target must be binary 0/1')
    X_train, X_test, y_train, y_test = train_test_split(df[FEATURES], df[TARGET], test_size=.2, random_state=42, stratify=df[TARGET])
    pipeline = Pipeline([('injury_features', InjuryFeatureEngineer()), ('scale', StandardScaler()), ('classifier', LogisticRegression(C=1.0, max_iter=2000, random_state=42))])
    cv = cross_val_score(pipeline, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc')
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    baseline = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    metrics = dict(accuracy=accuracy_score(y_test, predictions), precision=precision_score(y_test, predictions, zero_division=0), recall=recall_score(y_test, predictions, zero_division=0), f1=f1_score(y_test, predictions, zero_division=0), roc_auc=roc_auc_score(y_test, probabilities), confusion_matrix=confusion_matrix(y_test, predictions).tolist(), baseline_accuracy=accuracy_score(y_test, baseline.predict(X_test)), train_cv_roc_auc_mean=float(cv.mean()), train_cv_roc_auc_std=float(cv.std()))
    metadata = dict(steps=[{'name': n, 'class': type(s).__name__} for n,s in pipeline.steps], built_at=datetime.now(timezone.utc).isoformat(), sklearn_version=sklearn.__version__, expected_input_features=FEATURES, target=TARGET, classes={'0': 'No injury label', '1': 'Injury label'}, metrics=metrics, rows=len(df), training_rows=len(X_train), test_rows=len(X_test), random_state=42, dataset_sha256=hashlib.sha256(Path(dataset).read_bytes()).hexdigest(), feature_ranges={c:{'min':float(df[c].min()),'max':float(df[c].max())} for c in FEATURES})
    joblib.dump({'pipeline': pipeline, 'metadata': metadata}, ROOT/'pipeline.joblib')
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports'/'evaluation.json').write_text(json.dumps(metadata, indent=2))
    (ROOT/'reports'/'dataset_profile.json').write_text(json.dumps({'shape':list(df.shape),'missing':df.isna().sum().to_dict(),'duplicates':int(df.duplicated().sum()),'summary':df.describe().to_dict(),'target_counts':df[TARGET].value_counts().to_dict()},indent=2))
    print(json.dumps(metadata, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=ROOT/'data'/'injury_data.csv')
    build(parser.parse_args().dataset)
