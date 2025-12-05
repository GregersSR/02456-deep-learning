import itertools
from model_lstm import LSTMModel
from model_autoregressive import Seq2SeqLSTM
from transformer_model import TrajectoryTransformer30to10
from model_baseline import LinearModel
from model_seq2seq_transformer_ed import Seq2SeqTransformerEd
from model_seq2seq_transformer_teacherforce import Seq2SeqTransformerTf

transformer_configs = [
    {   
        "name": "mini_transformer",
        "epochs": 10,
        "batch_size": 64,
        "model_kwargs": {
            "d_model": 128,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 10,
            "dropout": 0.1,
        },
        "optimizer_args": {
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "small_transformer",
        "epochs": 40,
        "batch_size": 64,
        "model_kwargs": {
            "d_model": 128,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 512,
            "dropout": 0.1,
        },
        "optimizer_args": {
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "medium_transformer",
        "epochs": 40,
        "batch_size": 64,
        "model_kwargs": {
            "d_model": 256,
            "nhead": 8,
            "num_layers": 3,
            "dim_feedforward": 1024,
            "dropout": 0.1,
        },
        "optimizer_args": {
            "lr": 5e-4,
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "deeper_transformer_2",
        "epochs": 90,
        "batch_size": 64,
        "model_kwargs": {
            "d_model": 256,
            "nhead": 8,
            "num_layers": 4,
            "dim_feedforward": 1024,
            "dropout": 0.1,
        },
        "optimizer_args": {
            "lr": 5e-4,
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "deeper_transformer",
        "epochs": 50,
        "batch_size": 64,
        "model_kwargs": {
            "d_model": 256,
            "nhead": 8,
            "num_layers": 4,
            "dim_feedforward": 1024,
            "dropout": 0.1,
        },
        "optimizer_args": {
            "lr": 5e-4,
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "deeper_transformer_3",
        "epochs": 200,
        "batch_size": 128,
        "model_kwargs": {
            "d_model": 32,
            "nhead": 8,
            "num_layers": 8,
            "dim_feedforward": 1024,
            "dropout": 0.1,
        },
        "optimizer_args": {
            "lr": 5e-4,
            "weight_decay": 1e-4,
        },
    },
]

transformer_seq2seq_tf_configs = [
    {
        "name": "mini_seq2seq_trans",
        "epochs": 1,
        "batch_size": 512,
        "model_kwargs": {
            "d_model": 8,
            "nhead": 4,
            "num_enc_layers": 1,
            "num_dec_layers": 1,
            "dim_feedforward": 1,
            "dropout": 0.1,
        },
        "optimizer_args": {
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "small_seq2seq_trans",
        "epochs": 50,
        "batch_size": 128,
        "model_kwargs": {
            "d_model": 16,
            "nhead": 4,
            "num_enc_layers": 3,
            "num_dec_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.1,
        },
        "optimizer_args": {
            "lr": 5e-4,
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "medium_seq2seq_trans",
        "epochs": 100,
        "batch_size": 128,
        "model_kwargs": {
            "d_model": 32,
            "nhead": 8,
            "num_enc_layers": 4,
            "num_dec_layers": 4,
            "dim_feedforward": 512,
            "dropout": 0.1,
        },
        "optimizer_args": {
            "lr": 5e-4,
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "deep_seq2seq_trans",
        "epochs": 100,
        "batch_size": 128,
        "model_kwargs": {
            "d_model": 64,
            "nhead": 8,
            "num_enc_layers": 5,
            "num_dec_layers": 5,
            "dim_feedforward": 1024,
            "dropout": 0.1,
        },
        "optimizer_args": {
            "lr": 5e-4,
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "twolayer_seq2seq_trans",
        "epochs": 100,
        "batch_size": 128,
        "model_kwargs": {
            "d_model": 32,
            "nhead": 4,
            "num_enc_layers": 2,
            "num_dec_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.3,
        },
        "optimizer_args": {
            "lr": 5e-4,
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "dropoutincrease_seq2seq_trans",
        "epochs": 100,
        "batch_size": 128,
        "model_kwargs": {
            "d_model": 64,
            "nhead": 8,
            "num_enc_layers": 3,
            "num_dec_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.4,
        },
        "optimizer_args": {
            "lr": 5e-4,
            "weight_decay": 1e-4,
        },
    },
]

transformer_seq2seq_notf_configs = [
    {
        "name": "ed_small_seq2seq_trans",
        "epochs": 50,
        "batch_size": 128,
        "model_kwargs": {
            "d_model": 16,
            "nhead": 4,
            "num_enc_layers": 3,
            "num_dec_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.1,
        },
        "optimizer_args": {
            "lr": 5e-4,
            "weight_decay": 1e-4,
        },
    },
]

lstm_configs = [
    {   
        "name": "mini_lstm",
        "epochs": 10,
        "batch_size": 512,
        "model_kwargs": {
            "hidden_size": 64,
            "num_layers": 2,
        },
        "optimizer_args": {
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "small_lstm",
        "epochs": 40,
        "batch_size": 512,
        "model_kwargs": {
            "hidden_size": 128,
            "num_layers": 3,
        },
        "optimizer_args": {
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "medium_lstm",
        "epochs": 40,
        "batch_size": 512,
        "model_kwargs": {
            "hidden_size": 512,
            "num_layers": 3,
        },
        "optimizer_args": {
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "deeper_lstm_2",
        "epochs": 90,
        "batch_size": 512,
        "model_kwargs": {
            "hidden_size": 512,
            "num_layers": 4,
        },
        "optimizer_args": {
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "deeper_lstm",
        "epochs": 90,
        "batch_size": 512,
        "model_kwargs": {
            "hidden_size": 512,
            "num_layers": 5,
        },
        "optimizer_args": {
            "lr": 5e-4,
            "weight_decay": 1e-2,
        },
    },
    {
        "name": "even_deeper_lstm",
        "epochs": 200,
        "batch_size": 128,
        "model_kwargs": {
            "hidden_size": 1024,
            "num_layers": 8,
        },
        "optimizer_args": {
            "lr": 5e-4,
            "weight_decay": 1e-4,
        },
    },
]

autoreg_configs = [
    {   
        "name": "mini_autoreg_lstm",
        "epochs": 10,
        "batch_size": 512,
        "model_kwargs": {
            "hidden_size": 64,
            "num_layers": 2,
        },
        "optimizer_args": {
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "small_autoreg_lstm",
        "epochs": 40,
        "batch_size": 512,
        "model_kwargs": {
            "hidden_size": 128,
            "num_layers": 3,
        },
        "optimizer_args": {
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "medium_autoreg_lstm",
        "epochs": 40,
        "batch_size": 512,
        "model_kwargs": {
            "hidden_size": 512,
            "num_layers": 3,
        },
        "optimizer_args": {
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "deeper_autoreg_lstm_2",
        "epochs": 90,
        "batch_size": 512,
        "model_kwargs": {
            "hidden_size": 512,
            "num_layers": 4,
        },
        "optimizer_args": {
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "deeper_autoreg_lstm",
        "epochs": 90,
        "batch_size": 512,
        "model_kwargs": {
            "hidden_size": 512,
            "num_layers": 5,
        },
        "optimizer_args": {
            "lr": 5e-4,
            "weight_decay": 1e-2,
        },
    },
    {
        "name": "even_deeper_autoreg_lstm",
        "epochs": 200,
        "batch_size": 128,
        "model_kwargs": {
            "hidden_size": 1024,
            "num_layers": 8,
        },
        "optimizer_args": {
            "lr": 5e-4,
            "weight_decay": 1e-4,
        },
    },
]

def find_cfg(name):
    if "autoreg" in name:
        return Seq2SeqLSTM, [cfg for cfg in autoreg_configs if cfg['name'] == name][0]
    elif "seq2seq_trans" in name:
        if name.startswith("ed_"):
            return Seq2SeqTransformerEd, [cfg for cfg in transformer_seq2seq_notf_configs if cfg['name'] == name][0]
        return Seq2SeqTransformerTf, [cfg for cfg in transformer_seq2seq_tf_configs if cfg['name'] == name][0]
    elif "transformer" in name:
        return TrajectoryTransformer30to10, [cfg for cfg in transformer_configs if cfg['name'] == name][0]
    elif name == "linear_model":
        return LinearModel, {}
    else:
        return LSTMModel, [cfg for cfg in lstm_configs if cfg['name'] == name][0]

def validate():
    valid = True
    mandatory_keys = ["name", "epochs", "batch_size", "model_kwargs", "optimizer_args"]
    mandatory_transformer_args = ["d_model", "nhead", "num_layers", "dim_feedforward", "dropout"]
    mandatory_seq2seq_trans_args = ["d_model", "nhead", "num_enc_layers", "num_dec_layers", "dim_feedforward", "dropout"]
    mandatory_lstm_args = ["hidden_size", "num_layers"]
    for cfg in transformer_configs:
        issues = []
        for key in mandatory_keys:
            if not key in cfg:
                issues.append(f"{key} missing in config.")
        for key in mandatory_transformer_args:
            if key not in cfg['model_kwargs']:
                issues.append(f"{key} missing in transformer kwargs.")
        if issues:
            valid = False
            print("\n".join(issues))
            print(cfg)
    for cfg in transformer_seq2seq_tf_configs:
        issues = []
        for key in mandatory_keys:
            if not key in cfg:
                issues.append(f"{key} missing in config.")
        for key in mandatory_seq2seq_trans_args:
            if key not in cfg['model_kwargs']:
                issues.append(f"{key} missing in transformer kwargs.")
        if issues:
            valid = False
            print("\n".join(issues))
            print(cfg)
    for config in itertools.chain(lstm_configs, autoreg_configs):
        issues = []
        for key in mandatory_keys:
            if not key in config:
                issues.append(f"{key} missing in config.")
        for key in mandatory_lstm_args:
            if key not in config['model_kwargs']:
                issues.append(f"{key} missing in lstm kwargs.")
        if issues:
            valid = False
            print("\n".join(issues))
            print(config)
    if valid:
        print("All okay.")
        return 0
    else:
        return 1
    
if __name__ == '__main__':
    exit(validate())
    