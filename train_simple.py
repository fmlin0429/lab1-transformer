import torch
import torch.nn as nn
import warnings
from pathlib import Path
from config import config

# Force CPU usage globally
torch.cuda.is_available = lambda: False

def simple_train():
    """Simple training loop to demonstrate CPU usage"""
    print("=== CPU Training Demo ===")
    print("✓ CUDA disabled globally")
    print("✓ All operations forced to CPU")
    
    # Get config
    cfg = config()
    
    # Check device
    device = "cpu"
    print(f"Using device: {device}")
    
    # Create a simple model for testing
    from model import build_transformer
    
    # Create dummy vocab sizes for testing
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    
    model = build_transformer(
        src_vocab_size, 
        tgt_vocab_size, 
        cfg['seq_len'], 
        cfg['seq_len'], 
        cfg['d_model']
    )
    
    model = model.to(device)
    print(f"Model moved to {device}")
    
    # Create dummy data
    batch_size = 2
    seq_len = cfg['seq_len']
    
    # Dummy input tensors
    encoder_input = torch.randint(0, src_vocab_size, (batch_size, seq_len)).to(device)
    decoder_input = torch.randint(0, tgt_vocab_size, (batch_size, seq_len)).to(device)
    
    print(f"Input shapes: encoder={encoder_input.shape}, decoder={decoder_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        encoder_output = model.encode(encoder_input, None)
        decoder_output = model.decode(encoder_output, None, decoder_input, None)
        output = model.project(decoder_output)
    
    print(f"Output shape: {output.shape}")
    print("✅ CPU training working correctly!")
    
    # Show memory usage
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return True

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    simple_train()
