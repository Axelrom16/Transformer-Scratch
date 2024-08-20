import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


def get_all_sentences(ds, lang):
    """
    Get all sentences from the dataset for the given language.
    Args:
        ds: str, dataset name
        lang: str, language
    """
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    """
    Get or build a tokenizer for the given language.
    Args:
        config: dict, configuration dictionary
        ds: str, dataset name
        lang: str, language
    """
    tokenizer_path = Path(config['tokenizer_file']).format(lang)
    if not Path.exists(tokenizer_path):
        # Build tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        # Load tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    """
    Get the dataset.
    Args:
        config: dict, configuration dictionary
    """
    ds_raw = load_dataset('opus_books', f'{config["source_lang"]}-{config["target_lang"]}', split='train')
    
    # Build tokenizers 
    tokenizer_source = get_or_build_tokenizer(config, ds_raw, config['source_lang'])
    tokenizer_target = get_or_build_tokenizer(config, ds_raw, config['target_lang'])

    # Split train and validation
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    


























































