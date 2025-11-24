#!/usr/bin/env python3
"""Runs training for several models and configurations.

Results are saved to `checkpoints/` directory.
"""

import json
import torch
import training
import dataloader
import paths
from model_lstm import LSTMModel
from model_autoregressive import Seq2SeqLSTM
from transformer_model import TrajectoryTransformer30to10
from model_baseline import LinearModel
from datetime import datetime

transformer_defaults = {
    "optimizer": torch.optim.AdamW,
    "optimizer_args": {
        "lr": 1e-3,
        "weight_decay": 1e-4,
    },
    "batch_size": 128,
}
transformer_configs = [
    {   
        "name": "mini_transformer",
        "epochs": 10,
        "model_kwargs": {
            "d_model": 128,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 10,
            "dropout": 0.1,
        },
    },
    {
        "name": "small_transformer",
        "epochs": 40,
        "model_kwargs": {
            "d_model": 128,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 512,
            "dropout": 0.1,
        },
    },
    {
        "name": "medium_transformer",
        "epochs": 40,
        "model_kwargs": {
            "d_model": 256,
            "nhead": 8,
            "num_layers": 3,
            "dim_feedforward": 1024,
            "dropout": 0.1,
        },
        "optimizer_args": {
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "deeper_transformer_2",
        "epochs": 90,
        "model_kwargs": {
            "d_model": 256,
            "nhead": 8,
            "num_layers": 4,
            "dim_feedforward": 1024,
            "dropout": 0.1,
        },
        "optimizer_args": {
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "deeper_transformer",
        "epochs": 50,
        "model_kwargs": {
            "d_model": 256,
            "nhead": 8,
            "num_layers": 4,
            "dim_feedforward": 1024,
            "dropout": 0.1,
        },
        "optimizer_args": {
            "lr": 5e-4,
        },
    },
]

lstm_defaults = dict(batch_size=512)
lstm_configs = [
    {   
        "name": "mini_lstm",
        "epochs": 10,
        "model_kwargs": {
            "hidden_size": 64,
            "num_layers": 2,
        },
    },
    {
        "name": "small_lstm",
        "epochs": 40,
        "model_kwargs": {
            "hidden_size": 128,
            "num_layers": 3,
        },
    },
    {
        "name": "medium_lstm",
        "epochs": 40,
        "model_kwargs": {
            "hidden_size": 512,
            "num_layers": 3,
        },
        "optimizer_args": {
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "deeper_lstm_2",
        "epochs": 90,
        "model_kwargs": {
            "hidden_size": 512,
            "num_layers": 4,
        },
        "optimizer_args": {
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "deeper_lstm",
        "epochs": 90,
        "model_kwargs": {
            "hidden_size": 512,
            "num_layers": 5,
        },
        "optimizer_args": {
            "lr": 5e-4,
        },
    },
]

autoreg_defaults = {
    "optimizer": torch.optim.Adam,
    "optimizer_args": {
        "lr": 1e-3,
        "weight_decay": 0,
    }
}

autoreg_configs = [
    {   
        "name": "mini_autoreg_lstm",
        "epochs": 10,
        "model_kwargs": {
            "hidden_size": 64,
            "num_layers": 2,
        },
    },
    {
        "name": "small_autoreg_lstm",
        "epochs": 40,
        "model_kwargs": {
            "hidden_size": 128,
            "num_layers": 3,
        },
    },
    {
        "name": "medium_autoreg_lstm",
        "epochs": 40,
        "model_kwargs": {
            "hidden_size": 512,
            "num_layers": 3,
        },
        "optimizer_args": {
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "deeper_autoreg_lstm_2",
        "epochs": 90,
        "model_kwargs": {
            "hidden_size": 512,
            "num_layers": 4,
        },
        "optimizer_args": {
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "deeper_autoreg_lstm",
        "epochs": 90,
        "model_kwargs": {
            "hidden_size": 512,
            "num_layers": 5,
        },
        "optimizer_args": {
            "lr": 5e-4,
        },
    },
]


def train_linear_model(train: torch.Tensor, val: torch.Tensor):
    model = LinearModel.train(train)
    results = model.evaluate(train, val)
    model_path = training.checkpoint_model_path('linear_model')
    torch.save(model, model_path)
    # wrap losses in lists so they look like other models' histories over epochs
    results_compat = {k: [v] for k, v in results.items()}
    return {
        'linear_model': {
            'config': {
                'name': 'linear_model',
                'model_kwargs': {},
            },
            'model': model,
            'history': results_compat,
            'checkpoint_path': model_path,
        }
    }


def isonow():
    return datetime.now().replace(microsecond=0).isoformat()

def checkpoint_exists(name):
    return training.checkpoint_model_path(name).exists()

def remove_existing(configs):
    """Returns only configs for which no checkpoint exists yet."""
    filtered = []
    for cfg in configs:
        if checkpoint_exists(cfg['name']):
            print(f"Skipping existing model checkpoint: {cfg['name']}")
        else:
            print(f"Will train model: {cfg['name']}")
            filtered.append(cfg)
    return filtered

def load_data():
    train, scaler = dataloader.load_train()
    val = dataloader.load_val(scaler or train.scaler)
    return train, val

def main():
    train, val = load_data()
    lstm_results = training.train_all(LSTMModel, remove_existing(lstm_configs), defaults=lstm_defaults, train=train, val=val)
    autoreg_results = training.train_all(Seq2SeqLSTM, remove_existing(autoreg_configs), defaults=autoreg_defaults, train=train, val=val)
    transformer_results = training.train_all(TrajectoryTransformer30to10, remove_existing(transformer_configs), defaults=transformer_defaults, train=train, val=val)
    baseline_results = train_linear_model(train, val)
    all_results = {**lstm_results, **autoreg_results, **transformer_results, **baseline_results}
    torch.save(all_results, paths.CHECKPOINTS_DIR / f"all_models_results-{isonow()}.pt")
    for result in all_results.values():
        del result['model']  # remove model from saved results for JSON serialization
        result['checkpoint_path'] = str(result['checkpoint_path'])
    with paths.CHECKPOINTS_DIR.joinpath(f"all_models_results-{isonow()}.json").open('w') as f:
        json.dump(all_results, fp=f, indent=4)

if __name__ == "__main__":
    main()
