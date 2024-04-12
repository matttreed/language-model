from src.configs.config import Config
from src.model.transformer import Transformer
from src.training.optimizer import AdamW
from src.model.util import crossEntropyLoss, load_model, save_model, get_batch
import torch
import numpy as np

# TODO use gradient clipping and cosine ennealing

def train_model(version: str, from_checkpoint_k: int | None = None):
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

    model.train()

    optimizer = AdamW(
        params=model.parameters(), 
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        betas=config.training.betas,
        eps=config.training.eps
    )

    if from_checkpoint_k:
        load_model(model, optimizer, version, from_checkpoint_k)

    train_data_name = config.data.training_data
    valid_data_name = config.data.validation_data
    train_data = np.memmap(f"data/processed/{train_data_name}.npy", dtype=np.int16, mode="r", offset=16*8)
    valid_data = np.memmap(f"data/processed/{valid_data_name}.npy", dtype=np.int16, mode="r", offset=16*8)

    start_iteration = from_checkpoint_k * 1000 if from_checkpoint_k else 0
    for iteration in range(start_iteration, config.training.num_iterations):
        optimizer.zero_grad()
        x, y = get_batch(data=train_data,
                        batch_size=config.training.batch_size,
                        context_length=config.transformer.context_length,
                        device=device)
        y_hat = model(x)
        loss = crossEntropyLoss(y, y_hat).mean()
        loss.backward()

        optimizer.step()

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}")
        
        if iteration % 1000 == 0:
            save_model(model, optimizer, version, iteration)