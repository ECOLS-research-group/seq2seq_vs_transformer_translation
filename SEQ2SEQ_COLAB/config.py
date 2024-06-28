from pathlib import Path

def get_config():
    return {
        "batch_size": 16,
        "num_epochs": 5,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": '/content/drive/My Drive/Seq2Seq',  # Change this path to the directory where you want to store your dataset
        "lang_src": "en",
        "lang_tgt": "hi_ng",
        "model_folder": "weights",
		    "hidden_size": 512,
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "/content/drive/My Drive/Seq2Seq/tokenizer_{0}.json",  # Change this path to where you want to save your tokenizer files
        "experiment_name": "/content/drive/My Drive/Seq2Seq/runs/tmodel"  # Change this path to where you want to save your model weights
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])