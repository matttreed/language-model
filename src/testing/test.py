from src.configs.config import Config
from src.model.transformer import Transformer
from src.training.optimizer import AdamW
from src.model.util import crossEntropyLoss, load_model, get_batch, get_tokenizer, softmax
import torch
import numpy as np

def sample_from_model(version: str, from_checkpoint_k: int, max_tokens: int | None = 1000, temperature: float = 1.0, top_p: float = 0.9):
    config = Config(version)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Transformer(
        vocab_size=config.tokenizer.vocab_size,
        context_length=config.transformer.context_length,
        num_layers=config.transformer.num_layers,
        d_model=config.transformer.d_model,
        num_heads=config.transformer.num_heads,
        d_ff=config.transformer.d_ff,
        attn_pdrop=config.transformer.attn_pdrop,
        residual_pdrop=config.transformer.residual_pdrop
    ).to(device)
    model.eval()
    load_model(model, None, version, from_checkpoint_k)

    tokenizer = get_tokenizer(config)

    test = "hello this is me doing a test and I will continue here "
    test = torch.tensor(tokenizer.encode(test), device=device).unsqueeze(0)

    next_token_probs = softmax(model(test)[0, -1], temperature=temperature)

    print(next_token_probs)
