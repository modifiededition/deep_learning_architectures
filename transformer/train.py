import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from dataset import BilingualDataset,casual_mask
from model import build_transformer
from config import get_config,get_weights_file_path

from pathlib import Path
import warnings

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.encode('[SOS]')
    eos_idx = tokenizer_tgt.encode('[EOS]')

    # Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for the target
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get the next token
        prob = model.project(out[:,-1])

        # select the token with the max prob.
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item().to(device))],dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len,device, print_msg,global_state,writer,num_examples=2):
    model.eval()
    count = 0

    # size of the control window
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count +=1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src,tokenizer_tgt,max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # print to the console
            print_msg('-'*console_width)
            print_msg(f"source: {source_text}")
            print_msg(f'target: {target_text}')
            print_msg(f'predicted: {model_out_text}')

            if count == num_examples:
                break


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer))
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

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len,vocab_tgt_len,config['seq_len'],config['seq_len'],config['d_model'])
    return model


def train_model(config):

    # define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer = get_ds(config)

    model = get_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size()).to(device)

    # TensorBoard
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)

    initial_epoch = 0
    global_step = 0

    # this is used to load the model weigts and optmizer from where the last epoch, in case there is a crash happens.
    if config['preload']:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    # training loop
    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()

        batch_iterator = tqdm(train_dataloader, desc = f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch["encoder_input"].to(device) # (batch_size, seq_len)
            decoder_input = batch["decoder_input"].to(device) # (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch_size, 1,1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device) # (batch_size, 1, seq_len, seq_len)

            # Run the tensors through the transformer

            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size,seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch_size, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch_size, seq_len, tgt_vocab_size)

            label = batch["label"].to(device) # batch_size, seq_len

            loss = loss_fn(proj_output.view(-1, tgt_tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
            # log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # bachpropogate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # save the model at the end of every epoch

        model_filename = get_weights_file_path(config,f"{epoch:02d}")
        torch.save(
            {
                'epoch':epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step, # it helps tenserboard to keep track of the loss number to update on the plot
            },
            model_filename
        )
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)


