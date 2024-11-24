import torch
import torch.nn as nn

from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from datasets import load_dataset
from torch.utils.data import random_split, DataLoader

from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_weights_file_path, latest_weights_file_path, config
from tqdm import tqdm
import torchmetrics
import os
import warnings

from torch.utils.tensorboard import SummaryWriter

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

        
def get_dataset(config):
    ds_raw = load_dataset(config['dataset_name'], f'{config["lang_src"]}-{config["lang_tgt"]}', split='test')
    
    # build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    # keep 90% of the data for training and 10% for validation
    train_size = int(0.9 * len(ds_raw))
    val_size = len(ds_raw) - train_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_size, val_size])
    
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    # find max_len of soource and target sequences
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    return train_loader, val_loader, tokenizer_src, tokenizer_tgt

def get_model(config, src_vocab_size, tgt_vocab_size):
    model = build_transformer(src_vocab_size, tgt_vocab_size, config['seq_len'], config['seq_len'], config['d_model'])
    return model
    

def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() or torch.has_mps else "cpu"
    print(f"Using device: {device}")
    
    # Print the device information
    if device == 'cuda': # if cuda is available, print the device name and memory
        device_name = torch.cuda.get_device_name(device.index)
        device_memory = torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3
        print(f"Device name: {device_name}")
        print(f"Device memory: {device_memory} GB")
    elif device == 'mps': # if mps is available, print the device name
        print("Device name: <mps>")
    else: # if no gpu is available, print a message to the user
        print("Consider using a GPU for training if available.")
        print("For NVidia GPU on Windows, refer to: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("For Mac, install: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    
    device = torch.device(device)
    
    Path(f"{config['dataset_name']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    
    train_loader, val_loader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    # tensorboard
    writer = SummaryWriter(config['experiment_name'])
    
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload)
    if model_filename:
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
    else:
        print("No preloaded weights found.")
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_loader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device).long() # (batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device).long() # (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch_size, 1, seq_len, seq_len)
            
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch_size, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch_size, seq_len, target_vocab_size)
            
            label = batch['label'].to(device) # (batch_size, seq_len)
            
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix(loss=loss.item())
            
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            
            loss.backward() # backpropagate the loss
            
            optimizer.zero_grad() # zero out the gradients
            
            optimizer.step() # update the model parameters
            
            global_step += 1
        
        # run validation
        validation(model, val_loader, tokenizer_src, tokenizer_tgt, device, global_step, writer, config['seq_len'], lambda msg: batch_iterator.write(msg))
        
        # save model checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(checkpoint, model_filename)

def validation(model, val_loader, tokenizer_src, tokenizer_tgt, device, global_step, writer, seq_len, print_msg):
    model.eval()
    count = 0
    
    source_texts = []
    expected = []
    predicted = []
    
    try:
        # get console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80
    
    with torch.no_grad():
        for batch in val_loader:
            count += 1
            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, seq_len)
            
            # make sure that batch size is 1 for validation
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, device, max_len=seq_len)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            print_msg('-' * console_width)
            
            print_msg(f"Source: {source_text:>{console_width - 12}}")
            print_msg(f"Expected: {target_text:>{console_width - 12}}")
            print_msg(f"Predicted: {model_out_text:>{console_width - 12}}")
            
            print_msg('-' * console_width)
            
    if writer:
        # compute the character error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        
        # compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()
        
        # BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        
        writer.add_scalar('validation bleu', bleu, global_step)
        writer.flush()
            
def greedy_decode(model, src, src_mask, tokenizer_src, tokenizer_tgt, device, max_len):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    encoder_output = model.encode(src, src_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(src).to(device)
    
    while True:
        if decoder_input.size(1) == max_len:
            break
        
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)
        
        output = model.decode(encoder_output, src_mask, decoder_input, decoder_mask)
        
        # get the next token
        probabilities = model.project(output[:, -1]) # (batch_size, vocab_size)
        _, next_word = torch.max(probabilities, dim=-1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(src).fill_(next_word.item()).to(device)], dim=1
        )
        
        if next_word == eos_idx:
            break
        
    return decoder_input.squeeze(0)
        
        
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = config()
    train_model(config)
    
    
    

