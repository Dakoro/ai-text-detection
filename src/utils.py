import re
import pickle
import torch
import statistics
from config import MODEL_CONFIG
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass

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

def load_model(filepath):
    with open(filepath, mode='rb') as pklf:
        model = pickle.load(pklf)
    return model

def calculate_perplexity(text: str):
    """Calculates the perplexity of a given text using a specified model."""
    model, tokenizer = ModelLoader(MODEL_CONFIG).load_model_and_tokenizer()
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
    


def compute_burstiness(text: str) -> float:
    """
    Computes the burstiness of a text, defined as the coefficient of variation
    of sentence lengths (standard deviation / mean).

    A higher score indicates more variation in sentence length, while a lower score
    suggests more uniform sentence lengths.

    Args:
        text: The input string to analyze.

    Returns:
        A float representing the burstiness score. Returns 0.0 for texts with
        fewer than two sentences, as variation cannot be calculated.
    """
    # 1. Handle empty or invalid input
    if not isinstance(text, str) or not text.strip():
        return 0.0

    # 2. Split text into sentences using punctuation as delimiters.
    # This simple regex handles periods, question marks, and exclamation points.
    sentences = [s.strip() for s in re.split(r'[.?!]', text) if s.strip()]

    # 3. Variation requires at least two data points (sentences).
    if len(sentences) < 2:
        return 0.0

    # 4. Calculate the length (in words) of each sentence.
    sentence_lengths = [len(s.split()) for s in sentences]

    # 5. Calculate mean and standard deviation of sentence lengths.
    mean_length = statistics.mean(sentence_lengths)
    stdev_length = statistics.stdev(sentence_lengths)

    # 6. Avoid division by zero if all sentences are empty.
    if mean_length == 0:
        return 0.0

    # 7. Compute burstiness (coefficient of variation).
    burstiness_score = stdev_length / mean_length
    
    return burstiness_score