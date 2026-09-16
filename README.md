# Injury Predict — Cyberpunk AI Injury Analysis Lab

A real trained classification application: CSV → stratified split → custom feature engineering → StandardScaler → LogisticRegression → fitted joblib bundle → FastAPI → Modal → React/Vite frontend on Vercel. Browser predictions come exclusively from POST /predict. No frontend model, mocks, or hard-coded predictions.

## Dataset and interpretation

Exact source: `C:\Users\obele\Downloads\archive (3)\injury_data.csv`. A byte-identical rebuild copy is `data/injury_data.csv`. SHA-256: `cc091b4398465db5645fb6134657b2557e6ff584aa78ecd137a3875a0476f7d1`.

1,000 rows, seven numeric columns, no missing cells, no duplicate rows. The CSV contains no data dictionary, collection method, prediction horizon, or outcome timestamps. Do not assume these records are clinically verified or representative of a real player population.

| Column | Interpretation and form bounds | Role |
|---|---|---|
| Player_Age | Player age, inferred years; integer 18–39 | Feature |
| Player_Weight | Player weight, inferred kg; 40.191–104.651 | Feature |
| Player_Height | Player height, inferred cm; 145.285–207.309 | Feature |
| Previous_Injuries | Binary previous-injury indicator, 0/1; not a count | Feature |
| Training_Intensity | Normalized numeric score, 0–1 | Feature |
| Recovery_Time | Integer 1–6, physical unit unspecified | Feature, with timing assumption |
| Likelihood_of_Injury | Binary outcome label, 500 zeros and 500 ones | Target only |

Despite its name, the target is a binary class rather than a continuous probability. Class 1 is presented as “Injury label,” class 0 as “No injury label”; this interpretation follows column naming, not an independently supplied label definition. Exact observed minima/maxima are recorded in `reports/evaluation.json`; decimal form bounds are rounded outward. Intensity bounds use its normalized 0–1 domain.

All six non-target columns are used. There are no identifier columns. The target is excluded from inputs to prevent leakage. Recovery_Time is retained only under the assumption that it describes recovery already known at inference time, before the outcome. If it instead records recovery from the very injury being predicted, it is a leakage variable: remove it and rebuild the training, API schema and form before claiming prospective prediction. The CSV alone cannot settle this ambiguity.

## Training and fitted state

Regularized logistic regression suits the small, balanced binary dataset, gives direct class probabilities, and limits complexity where signal is weak. No model or hyperparameter choices are made using the held-out test labels. Fixed seed 42, stratified 80/20 split (800 training, 200 test); five-fold stratified cross-validation runs on training data only.

`InjuryFeatureEngineer` in `pipeline_def.py` inherits BaseEstimator and TransformerMixin. Its constructor only assigns its argument; fit learns training-only medians and returns self. Transform imputes missing features and adds body-size index (weight / height-in-metres squared), intensity/recovery ratio, and prior-injury × intensity interaction. These are plausible engineering hypotheses, not established causal injury factors. StandardScaler learns means/variances; LogisticRegression learns coefficients. All are fitted on training rows only. No missing cells were observed, but the learned imputation makes the transformer reusable. API requests must still supply all six values.

`pipeline.joblib` is a dictionary with `pipeline` and `metadata`, including step names/classes, UTC timestamp, exact sklearn version **1.7.2**, features, metrics, class labels and dataset hash. The tested training model is the saved serving model; it is not refitted on the test set. `serve.py` loads this trusted local artifact once at module import and never fits a model.

| Held-out metric | Value |
|---|---:|
| Accuracy | 0.525 |
| Precision | 0.518519 |
| Recall | 0.700 |
| F1 | 0.595745 |
| ROC-AUC | 0.543 |
| Majority baseline accuracy | 0.500 |
| Training CV ROC-AUC mean ± SD | 0.480656 ± 0.027985 |

Confusion matrix, rows true and columns predicted, label order [0,1]: `[[35,65],[30,70]]`. Performance is close to chance; cross-validation does not support reliable generalization. This is an end-to-end ML engineering demonstration, not a validated injury-risk tool. The displayed probability is a logistic model estimate, not a calibrated clinical probability.

## Local setup (PowerShell)

Use a working CPython 3.13 interpreter. An isolated `.venv` has been created in this project.

```powershell
cd C:\Users\obele\injury-predict
# For a fresh installation:
& C:\Users\obele\anaconda3\python.exe -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe build_pipeline.py
# Or explicitly rebuild from the original:
& .\.venv\Scripts\python.exe build_pipeline.py --dataset 'C:\Users\obele\Downloads\archive (3)\injury_data.csv'
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m uvicorn serve:app --reload
```

With the environment activated, `uvicorn serve:app --reload` also works. Local docs: http://localhost:8000/docs. No training occurs during API startup. Missing, corrupt or incompatible artifacts yield 503, including /health. Unexpected fields, missing fields, fractional integer fields and out-of-range values yield 422.

| Endpoint | Purpose |
|---|---|
| GET /health | 200 when artifact is loaded, 503 otherwise |
| GET /pipeline | Steps, exact version, timestamp, features and measured metrics |
| POST /predict | Six validated features → actual class, label and class-1 probability |

Valid body:
```json
{"Player_Age":28,"Player_Weight":75,"Player_Height":180,"Previous_Injuries":0,"Training_Intensity":0.5,"Recovery_Time":4}
```

## Deployment

Modal image includes serve.py, pipeline_def.py and pipeline.joblib. The exact sklearn pin is read from the build report and checked again against the artifact at API import. Keep artifact and report together when rebuilding.

```powershell
cd C:\Users\obele\injury-predict
& .\.venv\Scripts\modal.exe token new
& .\.venv\Scripts\modal.exe secret create injury-predict-config 'ALLOWED_ORIGINS=http://localhost:3000,https://YOUR-PRODUCTION-SITE.vercel.app'
& .\.venv\Scripts\modal.exe deploy modal_serve.py
```

Copy the exact public API URL printed by Modal. CORS uses explicit origins, not a wildcard; replace the production origin with the actual stable Vercel origin. Preview origins require explicit addition if needed. Modal cold starts can take longer; the frontend allows a 120-second request timeout.

```powershell
cd frontend
npm.cmd install
# Copy .env.example to .env.local and set VITE_API_URL to the real Modal URL.
npm.cmd run dev
npm.cmd run build
npx.cmd vercel login
npx.cmd vercel --prod --yes --name injury-predict-lab --build-env VITE_API_URL=https://YOUR-MODAL-API.modal.run
```

Frontend is React/Vite; its public environment variable is **VITE_API_URL** (the Vite equivalent of NEXT_PUBLIC_API_URL). Set it in Vercel project settings for production and preview, then rebuild. An unset URL produces a real configuration error; production rejects localhost URLs. Status and evaluation cards fetch actual /health and /pipeline responses. The six form fields match the API and trained model. Set Modal's ALLOWED_ORIGINS to the resulting Vercel origin, then redeploy Modal. Deployment procedure follows [Modal web-function documentation](https://modal.com/docs/guide/webhooks) and [Vercel CLI deployment documentation](https://vercel.com/docs/cli/deploy).

## Submission and Postman

Modal API URL: **Pending authenticated deployment**

API Docs URL: **Pending — Modal API URL + /docs**

Vercel URL: **Pending authenticated deployment**

Import `postman/collection.json` in Postman. Set collection variable `base_url` to the deployed Modal URL, without trailing slash. Run all four requests or use Collection Runner. The health and metadata requests assert 200; valid prediction asserts 200 and actual prediction/probability fields; invalid prediction asserts 422 and validation details. No screenshot or cloud-testing result is claimed until performed.

For your screenshots, run “Predict VALID — 200 (SCREENSHOT)” and “Predict INVALID — 422 (SCREENSHOT)”. Capture each with the full public Modal request URL, request JSON, response JSON and visible status code. Capture passing test results too if required. The invalid request uses age 99, intensity 1.5 and recovery 0. Save your screenshots in `postman/screenshots/`.

If desired, run the same deployed collection headlessly: `npx.cmd newman run postman/collection.json --env-var base_url=https://YOUR-MODAL-API.modal.run`. Screenshots must still be taken in the actual Postman application.

## Verification completed locally

12 Python tests passed, including real pipeline-response equality, all feature bounds, missing/extra fields, and missing/corrupt artifact imports. Actual HTTP checks passed for /health, /pipeline, /docs, valid prediction (200) and invalid prediction (422); results are `reports/local_http_checks.json`. Two Playwright tests passed against the real localhost API: desktop prediction equality and mobile interaction/validation. UI captures: `reports/frontend-desktop.png` and `reports/frontend-mobile.png`. These captures are local UI QA, not Postman screenshots or deployed evidence. The production Vercel build passed with Vite 7.3.6; npm audit reported zero vulnerabilities after updating Vite.

To rerun browser checks, first run FastAPI on port 8000, then run `npx.cmd playwright test` inside frontend. Playwright starts a development frontend with the local API configured only for these local tests.

After cloud deployment, run:

```powershell
& .\.venv\Scripts\python.exe verify_deployment.py --api-url https://YOUR-MODAL-API.modal.run --vercel-url https://YOUR-PRODUCTION-SITE.vercel.app
```

This checks actual public API responses, production CORS and website availability, saves the cloud test evidence, fills the Postman collection URL and records all three submission URLs above. Also open the Vercel site and run a prediction to verify the production browser flow before capturing your Postman screenshots.
