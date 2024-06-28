# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Add your code directory to Python path
import sys
sys.path.append('/content/drive/My Drive/Transformer')

# Import your modules
from pathlib import Path
import torch
import torch.nn as nn
from config import get_config, latest_weights_file_path
from train import get_model, get_ds, run_validation
from translate import translate

def main():
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load configuration
    config = get_config()

    # Get data loaders and tokenizers
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # Build the model
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Load the pretrained weights
    model_filename = latest_weights_file_path(config)
    if model_filename:
        # state = torch.load(model_filename, map_location=device)
        state = torch.load(model_filename, map_location=torch.device('cpu'))
        model.load_state_dict(state['model_state_dict'])
        print(f"Loaded model from {model_filename}")
    else:
        print("No pretrained model found. Please check the weights path and filenames.")
        return

    # Run validation
    run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: print(msg), 0, None, num_examples=10)

    # Translate a sentence (example usage)
    translation = translate("Where do you stay?")
    print(f"\nTranslation: {translation}")

if __name__ == "__main__":
    main()
