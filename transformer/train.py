import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from dataset import BilingualDataset

from pathlib import Path

def get_all_sentences(ds, lang):
    for item in ds:
        yield ds['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(tokenizer)
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)

    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split= 'train')

    # Build tokenizers
    src_tokenizer = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tgt_tokenizer = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Splitting raw data into train(90%) and val (10%)
    train_ds_size = int(0.9*len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size,val_ds_size])

    train_ds = BilingualDataset(train_ds_raw,src_tokenizer,tgt_tokenizer,config["lang_src"], config["lang_tgt"],config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw,src_tokenizer,tgt_tokenizer,config["lang_src"], config["lang_tgt"],config["seq_len"])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = src_tokenizer.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tgt_tokenizer.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size = config["batch_size"], shuffle= True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle=True)

    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer
