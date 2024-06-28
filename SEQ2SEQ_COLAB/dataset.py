import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        
        # Tokenize source and target texts
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        return {
            "encoder_input": torch.tensor(enc_input_tokens, dtype=torch.int64),
            "decoder_input": torch.tensor(dec_input_tokens, dtype=torch.int64),
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def collate_fn(batch):
    encoder_inputs = [item["encoder_input"] for item in batch]
    decoder_inputs = [item["decoder_input"] for item in batch]
    src_texts = [item["src_text"] for item in batch]
    tgt_texts = [item["tgt_text"] for item in batch]
    
    encoder_inputs_padded = pad_sequence(encoder_inputs, batch_first=True, padding_value=0)
    decoder_inputs_padded = pad_sequence(decoder_inputs, batch_first=True, padding_value=0)
    
    return {
        "encoder_input": encoder_inputs_padded,
        "decoder_input": decoder_inputs_padded,
        "src_text": src_texts,
        "tgt_text": tgt_texts,
    }