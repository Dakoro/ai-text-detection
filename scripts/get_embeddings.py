import os
import re
import pickle
import glob
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

CHECKPOINT_DIR = os.path.join('data', 'checkpoint')
EMBEDDINGS_MODEL = "Qwen/Qwen3-Embedding-0.6B"

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

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
    fp = os.path.join(CHECKPOINT_DIR, f'embeddings_{last_checkpoint_idx}.pkl')
    with open(fp, mode='rb') as pklf:
        last_checkpoint_list = pickle.load(pklf)
    return last_checkpoint_list

def compute_embedding(model, sentences):
    if len(glob.glob(f'{CHECKPOINT_DIR}/*.pkl')) == 0:
        embeddings = []
    else:
        embeddings = load_last_checkpoint()
    for idx, sentence in enumerate(tqdm(sentences)):
        if idx <= len(embeddings) and idx != 0:
            continue
        
        embedding = model.encode(sentence)
        embeddings.append(embedding)
        
        if idx %  10000 == 0:
            print(f'Creating checkpoint_{idx}')
            filename = f'embeddings_{idx}'
            save_checkpoint(filename, embeddings)
            print(f"{filename} created")
        

def main():
    data_path = os.path.join('data', 'clean_data.parquet')
    df = pd.read_parquet(data_path, engine='pyarrow')
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    compute_embedding(model, df['text'])
    


if __name__ == '__main__':
    main()