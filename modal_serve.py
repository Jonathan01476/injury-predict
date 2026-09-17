import json
from pathlib import Path
import modal

ROOT = Path(__file__).parent
VERSION = json.loads((ROOT/'reports'/'evaluation.json').read_text())['sklearn_version']
app = modal.App('injury-predict')
image = modal.Image.debian_slim(python_version='3.13').pip_install(
    f'scikit-learn=={VERSION}', 'pandas==2.3.3', 'joblib==1.5.2', 'fastapi==0.135.1', 'numpy==2.5.3', 'scipy==1.18.1'
).add_local_file(ROOT/'serve.py','/root/serve.py').add_local_file(ROOT/'pipeline_def.py','/root/pipeline_def.py').add_local_file(ROOT/'pipeline.joblib','/root/pipeline.joblib').add_local_file(ROOT/'reports'/'evaluation.json','/root/reports/evaluation.json')

@app.function(image=image, secrets=[modal.Secret.from_name('injury-predict-config')], scaledown_window=300)
@modal.asgi_app()
def api():
    from serve import app as web_app
    return web_app
