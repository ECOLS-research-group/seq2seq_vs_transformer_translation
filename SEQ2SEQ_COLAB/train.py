# train.py

import os
from google.colab import drive
drive.mount('/content/drive')

# Change directory to the folder where your files are located
os.chdir('/content/drive/My Drive/Seq2Seq')

import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from config import get_config, get_weights_file_path, latest_weights_file_path
from dataset import BilingualDataset, collate_fn
from model import Encoder, Decoder, Seq2Seq
import random 
import tqdm

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('findnitai/english-to-hinglish', split='train')
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'])
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    device = torch.device(device)
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    print("Config:", config)  # Debug print statement to check config contents

    # Initialize Encoder and Decoder
    encoder = Encoder(input_size=tokenizer_src.get_vocab_size(), hidden_size=config['hidden_size'])
    decoder = Decoder(output_size=tokenizer_tgt.get_vocab_size(), hidden_size=config['hidden_size'])

    # Initialize the Seq2Seq model with the Encoder and Decoder instances
    model = Seq2Seq(encoder, decoder).to(device)

    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm.tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            
            # Initialize hidden and cell states for LSTM
            hidden = torch.zeros(decoder.num_layers, encoder_input.size(0), decoder.hidden_size).to(device)
            cell = torch.zeros(decoder.num_layers, encoder_input.size(0), decoder.hidden_size).to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass through the Seq2Seq model
            outputs = model(encoder_input, decoder_input)
            
            # Shift the targets by one timestep
            target = decoder_input[:, 1:].contiguous().view(-1)
            outputs = outputs[:, 1:].contiguous().view(-1, outputs.shape[-1])
            
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()
            
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            global_step += 1
        
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []
    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_output = model.encoder(encoder_input)
            decoder_output = model.decoder(encoder_output, decoder_input)
            model_out = torch.argmax(decoder_output, dim=-1)
            model_out_text = tokenizer_tgt.decode(model_out.squeeze().detach().cpu().numpy())
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
                  
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")
            if count == num_examples:
                print_msg('-'*console_width)
                break

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
