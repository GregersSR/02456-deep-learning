#!/usr/bin/env python3
"""Runs training for several models and configurations.

Results are saved to `checkpoints/` directory.
"""

import json
import sys
import torch
import training
import dataloader
import paths
from model_lstm import LSTMModel
from model_autoregressive import Seq2SeqLSTM
from transformer_model import TrajectoryTransformer30to10
from model_baseline import LinearModel
from util import isonow
from configurations import *

FILTER_STATIONARY = False

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
    train, scaler = dataloader.load_train(filter_stationary=FILTER_STATIONARY)
    val = dataloader.load_val(scaler or train.scaler, filter_stationary=FILTER_STATIONARY)
    return train, val, scaler

def find_cfg(name):
    if "autoreg" in name:
        return Seq2SeqLSTM, [cfg for cfg in autoreg_configs if cfg['name'] == name][0]
    elif "transformer" in name:
        return TrajectoryTransformer30to10, [cfg for cfg in transformer_configs if cfg['name'] == name][0]
    elif name == "linear_model":
        return LinearModel, {}
    else:
        return LSTMModel, [cfg for cfg in lstm_configs if cfg['name'] == name][0]

def train_one(name):
    model_cls, cfg = find_cfg(name)
    now = isonow()
    train, val, scaler = load_data()
    if model_cls == LinearModel:
        results = train_linear_model(train, val)['linear_model']
    else:
        model, history = training.train_with_config(model_cls, cfg, train=train, val=val)
        results = {
            'config': cfg,
            'model': model,
            'history': history,
            'checkpoint_path': training.checkpoint_model_path(cfg['name']),
        }
    torch.save(results, paths.CHECKPOINTS_DIR / f"{name}_results-{now}.pt")
    del results['model']  # remove model from saved results for JSON serialization
    results['checkpoint_path'] = str(results['checkpoint_path'])
    with paths.CHECKPOINTS_DIR.joinpath(f"{name}_results-{now}.json").open('w') as f:
        json.dump(results, fp=f, indent=4)

def train_all():
    now = isonow()
    train, val, scaler = load_data()
    torch.save(scaler, paths.CHECKPOINTS_DIR / f"data_scaler-{now}.pt")
    lstm_results = training.train_all(LSTMModel, remove_existing(lstm_configs), train=train, val=val)
    autoreg_results = training.train_all(Seq2SeqLSTM, remove_existing(autoreg_configs), train=train, val=val)
    transformer_results = training.train_all(TrajectoryTransformer30to10, remove_existing(transformer_configs), train=train, val=val)
    baseline_results = train_linear_model(train, val)
    all_results = {**lstm_results, **autoreg_results, **transformer_results, **baseline_results}
    torch.save(all_results, paths.CHECKPOINTS_DIR / f"all_models_results-{now}.pt")
    for result in all_results.values():
        del result['model']  # remove model from saved results for JSON serialization
        result['checkpoint_path'] = str(result['checkpoint_path'])
    with paths.CHECKPOINTS_DIR.joinpath(f"all_models_results-{now}.json").open('w') as f:
        json.dump(all_results, fp=f, indent=4)

if __name__ == "__main__":
    if sys.argv[1] == "all":
        train_all()
    else:
        train_one(sys.argv[1])
