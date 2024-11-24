import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, max_len):
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len

        # Move tensors to the correct device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id('[SOS]')], dtype=torch.int64).to(self.device)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id('[EOS]')], dtype=torch.int64).to(self.device)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id('[PAD]')], dtype=torch.int64).to(self.device)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Keep trying until we find a valid item or run out of items
        while idx < len(self.dataset):
            try:
                src_target_pair = self.dataset[idx]
                src_text = src_target_pair['translation'][self.src_lang]
                tgt_text = src_target_pair['translation'][self.tgt_lang]
                
                # Tokenize the texts
                enc_input_tokens = self.tokenizer_src.encode(src_text).ids
                dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
                
                # Check if sequence lengths are too long
                if len(enc_input_tokens) > self.max_len - 2 or len(dec_input_tokens) > self.max_len - 1:
                    idx += 1  # Try next item
                    continue
                
                enc_num_padding_tokens = self.max_len - len(enc_input_tokens) - 2
                dec_num_padding_tokens = self.max_len - len(dec_input_tokens) - 1
                
                # Get the device from one of the existing tensors
                device = self.sos_token.device
                
                # Move all tensors to the correct device
                encoder_input = torch.cat(
                    [
                        self.sos_token,
                        torch.tensor(enc_input_tokens, dtype=torch.int64, device=device),
                        self.eos_token,
                        torch.tensor([self.pad_token.item()] * enc_num_padding_tokens, dtype=torch.int64, device=device),
                    ],
                    dim=0
                )
                
                decoder_input = torch.cat(
                    [
                        self.sos_token,
                        torch.tensor(dec_input_tokens, dtype=torch.int64, device=device),
                        torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.int64, device=device),
                    ],
                    dim=0
                )
                
                label = torch.cat(
                    [
                        torch.tensor(dec_input_tokens, dtype=torch.int64, device=device),
                        self.eos_token,
                        torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.int64, device=device),
                    ],
                    dim=0
                )
                
                # make sure size of tensors are all max_len
                assert encoder_input.size(0) == self.max_len
                assert decoder_input.size(0) == self.max_len
                assert label.size(0) == self.max_len
                
                return {
                    "encoder_input": encoder_input,
                    "decoder_input": decoder_input,
                    "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, max_len)
                    "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, max_len) & (1, max_len, max_len)
                    "label": label,
                    "src_text": src_text,
                    "tgt_text": tgt_text,
                }
                
            except Exception as e:
                print(f"Error processing item {idx}: {str(e)}")
                idx += 1  # Try next item
                continue
                
        raise IndexError("No valid items found")
        
        
def causal_mask(size):
    # Create the mask on the same device as other tensors
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.int).to(device)
    return mask == 0

        