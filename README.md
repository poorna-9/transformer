# NeuroLite-Transformer

A complete Transformer architecture built from scratch using only **NumPy** — no TensorFlow, PyTorch, or external deep learning frameworks. Designed to demystify the internals of attention mechanisms, this project is ideal for learning, experimentation, and custom research.

## Features

- Implements full **Transformer Encoder and Decoder** architecture.
- All core components from scratch:
  - **Multi-Head Self Attention**
  - **Scaled Dot Product Attention**
  - **Positional Encoding**
  - **Feed Forward Network**
  - **Layer Normalization**
  - **Residual Connections**
- Modular design — define custom encoder/decoder stacks easily.
- Compatible with sequence-to-sequence tasks like:
  - **Machine Translation**
  - **Text Summarization**
  - **Code Generation**
- Built entirely using **NumPy**.

##  Example Usage

```python
from neuro_lite_transformer import Transformer, Tokenizer, CrossEntropyLoss

transformer = Transformer(
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    input_vocab_size=5000,
    target_vocab_size=5000,
    max_seq_len=100
)

transformer.compile(loss=CrossEntropyLoss(), learning_rate=0.0001)
transformer.fit(src_sentences, tgt_sentences, epochs=10, batch_size=64)
transformer.evaluate(src_val, tgt_val)
