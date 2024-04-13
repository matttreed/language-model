from src.configs.config import Config
from src.model.transformer import Transformer
from src.training.optimizer import AdamW
from src.model.util import crossEntropyLoss, load_model, get_batch, get_tokenizer, softmax
import torch
import numpy as np

def sample_from_model(prompt: str, version: str, from_checkpoint_k: int, max_tokens: int = 1000, temperature: float = 1.0, top_p: float | None = None):
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

    
    tokens = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)

    while tokens.shape[1] < max_tokens:
        max_context_len = config.transformer.context_length
        in_tokens = tokens
        if tokens.shape[1] > max_context_len:
            in_tokens = tokens[:, -max_context_len:]
        next_token_probs = softmax(model(in_tokens)[0, -1], temperature=temperature)
        next_token = torch.multinomial(next_token_probs, 1).item()
        tokens = torch.cat((tokens, torch.tensor([[next_token]], device=device)), dim=1)

    return tokenizer.decode(tokens.squeeze(0).tolist())


