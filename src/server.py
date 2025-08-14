import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer

from utils import load_model, calculate_perplexity, compute_burstiness
from config import MODEL_PATH, HF_TOKEN, PORT

from huggingface_hub._login import _login

print(HF_TOKEN, type(HF_TOKEN))

_login(token=HF_TOKEN, add_to_git_credential=False)

app = FastAPI()
templates = Jinja2Templates(directory="src/templates")


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post("/", response_class=HTMLResponse)
def predict(request: Request, text: str = Form(...)):
    model = load_model(MODEL_PATH)
    emb_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    embedding = emb_model.encode(text)
    perplexity = calculate_perplexity(text)
    burstiness = compute_burstiness(text)
    prediction = model.predict([[perplexity, burstiness, embedding.mean()]])
    if prediction == 0:
        result = 'Humain'
    else:
        result = 'ia'
    return templates.TemplateResponse('index.html', context={"request": request, 'result': result})

if __name__ == "__main__":
    config = uvicorn.Config("server:app", host="0.0.0.0", port=int(PORT), log_level="info")
    server = uvicorn.Server(config)
    server.run()