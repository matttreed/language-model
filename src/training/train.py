from src.configs.config import Config
from src.model.transformer import Transformer
from src.training.optimizer import AdamW
from src.model.util import crossEntropyLoss, load_model, save_model, get_batch, log_validation_loss, log, build_model
import torch
import numpy as np

def train_model(version: str, from_checkpoint_k: int | None = None):
    config = Config(version)
    torch.random.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = build_model(config).to(device)

    model.train()

    optimizer = AdamW(
        params=model.parameters(), 
        weight_decay=config.training.weight_decay,
        betas=config.training.betas,
        eps=config.training.eps,
        max_grad_norm=config.training.max_grad_norm,
        alpha_min=config.training.alpha_min,
        alpha_max=config.training.alpha_max,
        T_warmup=config.training.T_warmup,
        T_cosine=config.training.T_cosine
    )

    if from_checkpoint_k:
        load_model(model, optimizer, version, from_checkpoint_k)

    train_data_name = config.data.training_data
    valid_data_name = config.data.validation_data
    train_data = np.memmap(f"data/processed/{train_data_name}.npy", dtype=np.int16, mode="r", offset=16*8)
    valid_data = np.memmap(f"data/processed/{valid_data_name}.npy", dtype=np.int16, mode="r", offset=16*8)

    iteration = from_checkpoint_k * 1000 if from_checkpoint_k else 0

    log(version, f"Training {version} from iteration {iteration}")
    while iteration < config.training.total_iterations:
        optimizer.zero_grad()
        x, y = get_batch(data=train_data,
                        batch_size=config.training.batch_size,
                        context_length=config.transformer.context_length,
                        device=device)
        y_hat = model(x)
        loss = crossEntropyLoss(y, y_hat).mean()
        loss.backward()

        optimizer.step()

        if iteration % config.training.log_every == 0:
            log_validation_loss(iteration, model, valid_data, version, config, device)

        if iteration != 0 and iteration % config.training.checkpoint_every == 0:
            save_model(model, optimizer, version, iteration, config)

        iteration += 1

    save_model(model, optimizer, version, iteration, config)