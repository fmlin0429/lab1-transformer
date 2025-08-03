from pathlib import Path

def config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4, # learning rate
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "ar",
        "lang_tgt": "de",
        "dataset_name": "Helsinki-NLP/opus-100",
        "model_folder": "weights",
        "model_basename": "tmodel__",
        "tokenizer_file": "tokenizer_{0}.json",
        'preload': 'latest', # None or 'latest'
        'experiment_name': 'runs/tmodel'
    }



def get_weights_file_path(config, epoch:str):
    model_folder = f"{config['dataset_name']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{config['lang_src']}-{config['lang_tgt']}__{epoch}.pt"
    return str(Path(model_folder) / model_filename)

def latest_weights_file_path(config):
    model_folder = f"{config['dataset_name']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{config['lang_src']}-{config['lang_tgt']}*.pt"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_file = sorted(weights_files)[-1]
    return str(weights_file)