import json

import requests


def intent_classifier(text):
    url = 'http://127.0.0.1:60061/service/api/medical_ner'
    data = {"text":text}
    headers = {'Content-Type':'application/json;charset=utf8'}
    reponse = requests.post(url,data=json.dumps(data),headers=headers)
    if reponse.status_code == 200:
        reponse = json.loads(reponse.text)
        return reponse['data']
    else:
        return -1