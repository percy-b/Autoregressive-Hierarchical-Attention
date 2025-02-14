# Hierarchical-attention

# Hierarchical Attention Decoder

A PyTorch implementation of a novel hierarchical attention-based decoder architecture that combines tree-structured attention mechanisms with traditional transformer components for efficient sequence processing.

## Architecture Overview

This architecture introduces a hierarchical attention mechanism that processes sequences through a tree structure, offering potential benefits in both computational efficiency and modeling capability:

- **Leaf-level Processing**: Sequences are split into chunks and processed with local self-attention
- **Tree-structured Information Flow**: Binary tree architecture for efficient global information processing
- **Causal Attention**: Maintains autoregressive properties throughout the hierarchy
- **Position-aware**: Incorporates fixed positional embeddings for sequence awareness

## Requirements

```
torch>=1.8.0
numpy
transformers
datasets
tiktoken
```

## Installation

```bash
git clone https://github.com/percy-b/Hierarchical-attention.git
cd Hierarchical-attention
pip install -r requirements.txt
```

## Usage

Basic usage example:

```python
import torch
from decoder import Decoder

# Initialize the model
model = Decoder(
    vocab_size=32000,        # Size of your vocabulary
    sequence_length=1024,    # Maximum sequence length
    heads=8,                 # Number of attention heads
    d_model=512,            # Model dimension
    levels=3                 # Number of hierarchical levels
)

# Forward pass
batch_size = 32
seq_length = 512
input_ids = torch.randint(0, 32000, (batch_size, seq_length))
outputs = model(input_ids)

# Generate (inference mode)
with torch.no_grad():
    predictions = model(input_ids, training=False)
```

## Model Architecture Details

### Position Embedding
- Implements sinusoidal position encodings
- Combines with token embeddings additively
- Position weights are fixed during training

### Hierarchical Attention
1. **Leaf Attention**
   - Processes local chunks independently
   - Uses standard scaled dot-product attention
   - Maintains causal masking within chunks

2. **Node Attention**
   - Builds binary tree structure
   - Cross-attention between siblings
   - Preserves causal information flow

### Decoder Block
- Dual hierarchical attention layers
- Feed-forward network with 4x expansion
- Layer normalization and residual connections
- Automatic padding handling for arbitrary sequence lengths

## Sequence Length Requirements

The sequence length must be compatible with the number of hierarchical levels:
- Maximum levels = log2(sequence_length)
- The implementation handles padding for sequences not perfectly divisible by 2^levels

## Training Tips

1. Start with a smaller number of levels and gradually increase
2. Monitor attention patterns at different hierarchical levels
3. Consider using gradient checkpointing for longer sequences
4. Adjust chunk sizes based on your specific use case

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

bwowek4@gmail.com
