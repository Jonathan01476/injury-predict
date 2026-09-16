"""Verify real cloud endpoints and record submission URLs after deployment."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse
import httpx

ROOT = Path(__file__).parent
PAYLOAD = dict(Player_Age=28,Player_Weight=75,Player_Height=180,Previous_Injuries=0,Training_Intensity=.5,Recovery_Time=4)

def verify(api_url, vercel_url):
    api_url, vercel_url = api_url.rstrip('/'), vercel_url.rstrip('/')
    for url in [api_url,vercel_url]:
        parsed = urlparse(url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.hostname in ['localhost','127.0.0.1']:
            raise ValueError('Submission requires real public HTTPS URLs')
    with httpx.Client(timeout=180) as client:
        health = client.get(api_url+'/health')
        assert health.status_code == 200 and health.json()['model_loaded'] is True
        metadata = client.get(api_url+'/pipeline')
        assert metadata.status_code == 200
        assert metadata.json()['dataset_sha256'] == json.loads((ROOT/'reports/evaluation.json').read_text())['dataset_sha256']
        docs = client.get(api_url+'/docs')
        assert docs.status_code == 200
        valid = client.post(api_url+'/predict',json=PAYLOAD)
        assert valid.status_code == 200 and valid.json()['prediction'] in [0,1]
        assert 0 <= valid.json()['injury_probability'] <= 1
        invalid = client.post(api_url+'/predict',json={**PAYLOAD,'Player_Age':99,'Training_Intensity':1.5,'Recovery_Time':0})
        assert invalid.status_code == 422
        cors = client.options(api_url+'/predict',headers={'Origin':vercel_url,'Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'content-type'})
        assert cors.status_code == 200 and cors.headers.get('access-control-allow-origin') == vercel_url
        site = client.get(vercel_url,follow_redirects=True)
        assert site.status_code == 200 and 'Injury Predict' in site.text
    report = {'modal_api_url':api_url,'api_docs_url':api_url+'/docs','vercel_url':vercel_url,'health_status':health.status_code,'metadata_status':metadata.status_code,'valid_status':valid.status_code,'valid_response':valid.json(),'invalid_status':invalid.status_code,'invalid_response':invalid.json(),'cors_origin':cors.headers['access-control-allow-origin'],'website_status':site.status_code}
    (ROOT/'reports/deployed_checks.json').write_text(json.dumps(report,indent=2))
    collection_path = ROOT/'postman/collection.json'
    collection = json.loads(collection_path.read_text())
    collection['variable'][0]['value'] = api_url
    collection_path.write_text(json.dumps(collection,indent=2))
    readme_path = ROOT/'README.md'
    readme = readme_path.read_text()
    readme = readme.replace('Modal API URL: **Pending authenticated deployment**',f'Modal API URL: {api_url}')
    readme = readme.replace('API Docs URL: **Pending — Modal API URL + /docs**',f'API Docs URL: {api_url}/docs')
    readme = readme.replace('Vercel URL: **Pending authenticated deployment**',f'Vercel URL: {vercel_url}')
    readme_path.write_text(readme)
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-url',required=True)
    parser.add_argument('--vercel-url',required=True)
    args = parser.parse_args()
    verify(args.api_url,args.vercel_url)
