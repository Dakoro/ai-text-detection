import os
import re
import tempfile
import torch
import glob
import pickle
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Configure the model you want to use
MODEL_CONFIG = {
    "id": "meta-llama/Llama-3.2-1B",
    "name": "Llama 3.2 1B",
    "torch_dtype": torch.float16
}

CHECKPOINT_DIR = os.path.join('data', 'checkpoint')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

@dataclass
class DataLoader:
    data: pd.DataFrame
    
    def load_parquet(self):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"Created temporary directory: {temp_dir}")
                temp_file_path = os.path.join(temp_dir, 'intermediate_data.parquet')

                # Save intermediate results
                self.data.to_parquet(temp_file_path)
                print(f"Saved intermediate data to {temp_file_path}")

                # Load and perform further processing...
                loaded_data = pd.read_parquet(temp_file_path)
                print("Loaded intermediate data for next step.")
                return loaded_data

        except Exception as e:
            print(f"An error occurred during processing: {e}")


@dataclass
class ModelLoader:
    config: dict
    
    def load_model_and_tokenizer(self) -> tuple:
        """Loads the specified model and tokenizer from Hugging Face."""
        model_id = self.config["id"]
        try:
            # Load tokenizer and add a padding token if it's missing
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load the model with device_map for automatic hardware placement (GPU/CPU)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=self.config["torch_dtype"],
                device_map="auto",
                low_cpu_mem_usage=True
            )
            return model, tokenizer
        except Exception as e:
            print(f"Error loading {self.config['name']}: {e}")
            return None, None

    
def save_checkpoint(filename, pyobject):
    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)
    fp = os.path.join(CHECKPOINT_DIR, f'{filename}.pkl',)
    with open(fp, mode='wb') as pklf:
        pickle.dump(pyobject, pklf, protocol=pickle.HIGHEST_PROTOCOL)


def load_last_checkpoint():
    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)
    
    file_idx = []
    for fp in glob.glob(f'{CHECKPOINT_DIR}/*.pkl'):
        fn = os.path.basename(fp)
        file_idx.append(int(re.findall(r'\d+', fn)[0]))
        
    last_checkpoint_idx = max(file_idx)
    fp = os.path.join(CHECKPOINT_DIR, f'perplexity_{last_checkpoint_idx}.pkl')
    with open(fp, mode='rb') as pklf:
        last_checkpoint_list = pickle.load(pklf)
    return last_checkpoint_list

def calculate_perplexity(text, model, tokenizer):
    """Calculates the perplexity of a given text using a specified model."""
    if not text.strip() or not model or not tokenizer:
        return float('inf')

    try:
        # Move model to the correct device
        device = next(model.parameters()).device
        
        # Tokenize the text
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        
        # Calculate the loss, which is the negative log-likelihood
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            neg_log_likelihood = outputs.loss
            
        # Exponentiate the loss to get perplexity
        perplexity = torch.exp(neg_log_likelihood)
        torch.cuda.empty_cache()
        return perplexity.item()
    except Exception as e:
        print(f"Error in perplexity calculation: {e}")
        return float('inf')


if __name__ == '__main__':
    data_path = os.path.join('data', 'AI_Human.csv')
    df = pd.read_csv(data_path)
    df = DataLoader(df).load_parquet()
    
    if len(glob.glob(f'{CHECKPOINT_DIR}/*.pkl')) == 0:
        perplexity_list = []
    else:
        perplexity_list = load_last_checkpoint()
    
    model, tokenizer = ModelLoader(MODEL_CONFIG).load_model_and_tokenizer()
    if model and tokenizer:
        for idx, txt in enumerate(tqdm(df['text'])):
            if idx <= len(perplexity_list):
                continue
            
            perplexity = calculate_perplexity(txt, model, tokenizer)
            perplexity_list.append(perplexity)
            
            if idx %  10000 == 0:
                print(f'Creating checkpoint_{idx}')
                filename = f'perplexity_{idx}'
                save_checkpoint(filename, perplexity_list)
                print(f"{filename} created")