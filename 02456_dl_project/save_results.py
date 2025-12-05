import training
import torch
from torch.utils.data import DataLoader
from dataloader import load_val, load_test
import joblib
import numpy as np
from tqdm import tqdm
from paths import RESULTS_FILTERED_DIR, RESULTS_UNFILTERED_DIR
from configurations import find_cfg
import pickle
from plot_trajectory import plot_paths


def load_scaler(filtered=True):
    scaler_file = "scaler_filtered.save" if filtered else "scaler_unfiltered.save"
    scaler = joblib.load(scaler_file)
    return scaler


def load_data(validation=True, filtered=True, batch_size = 512): 
    """
    Returns validation dataloader based on whether the model is filtered or unfiltered.
    """

    if validation == True:
        ds = load_val(filter_stationary=filtered, scaler=load_scaler(filtered))
    else:
        ds = load_test(filter_stationary=filtered, scaler=load_scaler(filtered))

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False
    )

    return loader

# evaluate the model:

def best_model(best_model_name: str, best_model_data: dict, device: str, filtered = True):
    """
    Loads a trained model and set it to evaluation
    """

    results_dir = RESULTS_FILTERED_DIR if filtered else RESULTS_UNFILTERED_DIR
    best_model_path = training.checkpoint_model_path(best_model_name).name
    full_best_model_path = results_dir / best_model_path

    model_class, cfg = find_cfg(best_model_name)

    model = model_class(**best_model_data["config"]["model_kwargs"])
    model.load_state_dict(torch.load(full_best_model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model



def compute_MSE_per_sample(model, val_loader, device, output_name: str):
    """
    Computes per-sample MSE and stores the results in a pickle file
    """
    # compute per-sample MSE on validation set (notice here we do per-sample MSE, so the total loss is 20 times smaller)
    all_mse = []
    all_samples = []

    for x, y in tqdm(val_loader, desc="Computing per-sample MSE"):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_pred = model(x)
        
        mse_per_sample = torch.mean((y_pred - y)**2, dim=[1,2])
        all_mse.append(mse_per_sample.cpu().numpy())
        
        for i in range(x.shape[0]):
            all_samples.append({
                "x": x[i].cpu().numpy(),
                "y": y[i].cpu().numpy(),
                "y_pred": y_pred[i].cpu().numpy()
            })

    all_mse = np.concatenate(all_mse)

    x = np.stack([s["x"] for s in all_samples])
    y = np.stack([s["y"] for s in all_samples])
    y_pred = np.stack([s["y_pred"] for s in all_samples])

    results = {
        "all_mse": all_mse,
        "x": x,
        "y": y,
        "y_pred": y_pred
    }

    with open(f"{output_name}_results.pkl", "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n Saved {output_name}_results.pkl")

    return results


def plot_sample(x, y, y_pred, idx, title, filtered=True):
    x_sample = x[idx]
    y_sample = y[idx]
    y_pred_sample = y_pred[idx]
    return plot_paths(x_sample, y_sample, y_pred_sample, title, scaler=load_scaler(filtered))