import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from schemes import MarkupResponse
import importlib
import os
from os.path import dirname
import sys

# adding models on backend
models = {}
sys.path.append(dirname('/model'))
for model_path in os.listdir('/models'):
    spec = importlib.util.spec_from_file_location("model_api", f"/models/{model_path}/model/model_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    models[model_path] = module.model_api

BACKEND_URL = os.getenv('BACKEND_URL', default='/')

app = FastAPI(root_path='/')
app.add_middleware(
    CORSMiddleware,
    allow_origins=["0.0.0.0", "localhost", "127.0.0.1"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/predict/human-values')
async def train_on_syntetic_dataset(text: str, model: str = 'gliner'):
    from fastapi.responses import JSONResponse
    response = models[model]('23 февраля в войсковой части 5526 прошло чествование воспитанников военно-патриотического клуба «Крепость». Заместитель командира части по идеологической работе майор Олег Ляшук отметил, что воспитание подрастающего поколения в патриотическом ключе, в стремлении к здоровому образу жизни, уважению к традициям, культурным ценностям и исторической памяти государства является главным профилактическим фактором безнравственности и аморальности. «Вместе мы вносим огромный вклад в будущее страны и нравственное здоровье нашего общества. Служба Родине во все времена была почетна. А служить можно по-разному, и не обязательно с оружием в руках. Служить можно в любом возрасте, служить можно и парню, и девушке. Служить можно и словом, и делом!» – подытожил Олег Ляшук.')
    return JSONResponse(content=response)


if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0')
