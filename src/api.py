import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    print(model_path)
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


@app.post('/predict/human-values', response_model=MarkupResponse)
async def train_on_syntetic_dataset(text: str, model: str = 'gliner'):
    response = models[model](text)
    return JSONResponse(content=response)

@app.get('available_models')
async def get_available_models():
    return models.keys()


if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0')
