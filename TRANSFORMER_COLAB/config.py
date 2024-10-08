from pathlib import Path

def get_config():
    base_path = '/content/drive/My Drive/Transformer'
    return {
        "batch_size": 20,
        "num_epochs": 10,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": base_path,
        "lang_src": "en",
        "lang_tgt": "hi_ng",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": f"{base_path}/tokenizer_{{0}}.json",
        "experiment_name": f"{base_path}/runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}/{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(model_folder) / model_filename)

def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}/{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
