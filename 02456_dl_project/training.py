"""Shared training utilities for models in this repo.

Implements a training loop with checkpointing and metrics,
as well as helper functions to train several models with different configs.

# model = Seq2SeqLSTM(...) or TrajectoryTransformer30to10(...)
# train_loader, val_loader created via dataloader.load_train()/load_val(scaler)

history, best_path = training.train_model(
    "my_model_name",
    model,
    train_loader,
    val_loader,
    num_epochs=20,
    criterion=torch.nn.MSELoss(reduction='sum'), # loss criterion
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4),
    scheduler=None, # optional learning rate scheduler
)

"""
import inspect
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import math
import torch

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import tqdm

import dataloader
from paths import CHECKPOINTS_DIR, ROOT
from util import isonow


def determine_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
def checkpoint_model_path(model_name: str) -> Path:
    return (CHECKPOINTS_DIR / f"{model_name}_best.pt")


def takes_arg(f, arg: str):
    sig = inspect.signature(f)
    return arg in sig.parameters


def train_model(
    model_name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
) -> Tuple[Dict[str, list], str]:
    """Train `model` using train_loader and val_loader.

    Returns (history, best_checkpoint_path).

    history keys: train_mse, train_rmse, train_mae, val_mse, val_rmse, val_mae
    """
    device = determine_device()
    print(f"Using device: {device}")

    pass_tgt = takes_arg(model.forward, "tgt")

    model = model.to(device)

    best_val_rmse = float("inf")
    criterion = torch.nn.MSELoss()
    best_ckpt_path = checkpoint_model_path(model_name)
    epoch_since_best = 0

    history = {
        "train_loss": [],
        "train_mse": [],
        "train_rmse": [],
        "train_mae": [],
        "val_loss": [],
        "val_mse": [],
        "val_rmse": [],
        "val_mae": [],
    }

    for epoch in range(1, num_epochs + 1):
        # TRAIN
        model.train()
        epoch_train_loss = 0.0
        train_mse_acc = 0.0
        train_ae_acc = 0.0
        n_train = 0

        for X, Y in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - Training", leave=False):
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            if pass_tgt:
                preds = model(X, Y)
            else:
                preds = model(X)

            loss = criterion(preds, Y)
            loss.backward()
            optimizer.step()

            bs = X.size(0)
            # Calculate squared error and absolute error for the batch
            batch_loss = loss.item()
            mse_batch = torch.square(preds - Y).mean().item()
            mae_batch = torch.abs(preds - Y).mean().item()

            epoch_train_loss += batch_loss * bs
            train_mse_acc += mse_batch * bs
            train_ae_acc += mae_batch * bs
            n_train += bs

        train_loss = epoch_train_loss / n_train
        train_mse = train_mse_acc / n_train
        train_rmse = math.sqrt(train_mse)
        train_mae = train_ae_acc / n_train

        # VALIDATION
        model.eval()
        epoch_val_loss = 0.0
        val_se_acc = 0.0
        val_ae_acc = 0.0
        n_val = 0

        with torch.no_grad():
            for Xv, Yv in tqdm.tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} - Validation", leave=False):
                Xv = Xv.to(device)
                Yv = Yv.to(device)

                preds_v = model(Xv)
                loss_v = criterion(preds_v, Yv)

                bs = Xv.size(0)
                batch_loss = loss_v.item()
                mse_batch = torch.square(preds_v - Yv).mean().item()
                mae_batch = torch.abs(preds_v - Yv).mean().item()

                epoch_val_loss += batch_loss * bs
                val_se_acc += mse_batch * bs
                val_ae_acc += mae_batch * bs
                n_val += bs

        val_loss = epoch_val_loss / n_val
        val_mse = val_se_acc / n_val
        val_rmse = math.sqrt(val_mse)
        val_mae = val_ae_acc / n_val
        history["train_loss"].append(train_loss)
        history["train_mse"].append(train_mse)
        history["train_rmse"].append(train_rmse)
        history["train_mae"].append(train_mae)
        history["val_loss"].append(val_loss)
        history["val_mse"].append(val_mse)
        history["val_rmse"].append(val_rmse)
        history["val_mae"].append(val_mae)

        # save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), best_ckpt_path)
            epoch_since_best = 0
        else:
            epoch_since_best += 1

        print(f"Epoch {epoch}/{num_epochs} | Train RMSE={train_rmse:.4f} | Val RMSE={val_rmse:.4f} | Epochs since best: {epoch_since_best}")
        if epoch_since_best >= 20:
            print("Early stopping due to no improvement in validation RMSE for 20 epochs.")
            break

    # return history
    return history


def train_with_config(model_fn, config, train, val):
    name = config['name']
    model_kwargs = config['model_kwargs']
    epochs = config['epochs']
    batch_size = config['batch_size']
    optimizer_args = config['optimizer_args']
    model = model_fn(**model_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_args)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

    print(f"\n=== Training config: {name} ===")
    print("Time: ", isonow())
    print("Model args:", model_kwargs)
    print("Epochs :", epochs)
    print("Batch size:", batch_size)
    print("AdamW optimizer args: {}".format(optimizer_args))

    history = train_model(
        name,
        model,
        train_loader,
        val_loader,
        num_epochs=epochs,
        optimizer=optimizer,
    )
    print(f"Finished training model: {name} at {isonow()}\n")
    return model, history

def train_all(model_fn, configs, train=None, val=None):
    results = {}
    for config in configs:
        model, history = train_with_config(
            model_fn,
            config,
            train=train,
            val=val,
        )
        results[config['name']] = {
            'config': config,
            'model': model,
            'history': history,
            'checkpoint_path': checkpoint_model_path(config['name']),
        }
    return results
