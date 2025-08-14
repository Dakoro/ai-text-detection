import os
import torch


MODEL_PATH = os.path.join('..', 'models', 'ai_detection.pkl')

MODEL_CONFIG = {
    "id": "meta-llama/Llama-3.2-1B",
    "name": "Llama 3.2 1B",
    "torch_dtype": torch.float16
}

HF_TOKEN = os.environ.get('HF_TOKEN')
PORT = os.environ.get('PORT')