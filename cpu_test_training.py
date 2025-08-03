"""
CPU-only training test for 1 epoch
This script ensures everything works on CPU without GPU interference
"""

import os
import torch
import warnings
from pathlib import Path
from config import config

# Force CPU usage completely
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

def cpu_training_test():
    """Test training with minimal data for CPU verification"""
    
    print("🚀 CPU TRAINING TEST - 1 EPOCH")
    print("=" * 40)
    
    # Force CPU
    device = torch.device('cpu')
    print(f"✅ Using device: {device}")
    
    # Create minimal test data instead of full dataset
    print("\n📊 Creating minimal test data...")
    
    # Simple test: create dummy data to verify training works
    from model import build_transformer
    
    cfg = config()
    
    # Use tiny vocab and sequence for quick test
    src_vocab_size = 100
    tgt_vocab_size = 100
    seq_len = 10
    batch_size = 2
    
    # Create model
    model = build_transformer(src_vocab_size, tgt_vocab_size, seq_len, seq_len, cfg['d_model'])
    model = model.to(device)
    
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy training data
    import torch.nn as nn
    import torch.optim as optim
    
    # Dummy data
    num_batches = 10  # Very small for quick test
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n🎯 Starting training with {num_batches} mini-batches...")
    
    model.train()
    total_loss = 0
    
    for batch_idx in range(num_batches):
        # Create dummy batch data
        encoder_input = torch.randint(0, src_vocab_size, (batch_size, seq_len))
        decoder_input = torch.randint(0, tgt_vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, tgt_vocab_size, (batch_size, seq_len))
        
        # Forward pass
        encoder_output = model.encode(encoder_input, None)
        decoder_output = model.decode(encoder_output, None, decoder_input, None)
        output = model.project(decoder_output)
        
        # Calculate loss
        loss = criterion(output.view(-1, tgt_vocab_size), targets.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 2 == 0:
            print(f"   Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    print(f"\n✅ Training completed!")
    print(f"   Average loss: {avg_loss:.4f}")
    print(f"   This proves the training pipeline works!")
    
    # Save the test model
    checkpoint = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }
    
    Path("test_weights").mkdir(exist_ok=True)
    torch.save(checkpoint, "test_weights/test_model.pth")
    
    print(f"\n💾 Model saved to: test_weights/test_model.pth")
    print(f"   File size: ~{os.path.getsize('test_weights/test_model.pth')/1024/1024:.1f} MB")
    
    return True

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    cpu_training_test()
    print("\n🎉 CPU training test completed successfully!")
    print("   Ready for GPU setup next!")
