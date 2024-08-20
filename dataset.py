import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset 


class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_source, tokenizer_target, source_lang, target_lang):
        super().__init__()
        self.ds = ds
        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target
        self.source_lang = source_lang
        self.target_lang = target_lang

        self.sos_token = torch.Tensor([tokenizer_source.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_source.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_source.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        source_target_pair = self.ds[idx]
        source_text = source_target_pair['translation'][self.source_lang]
        target_text = source_target_pair['translation'][self.target_lang]

        enc_input_tokens = self.tokenizer_source.encode(source_text).ids
        dec_input_tokens = self.tokenizer_target.encode(target_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # 2 for SOS and EOS tokens
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # 1 for EOS token

        