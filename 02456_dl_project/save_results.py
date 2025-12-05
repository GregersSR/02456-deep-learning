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
from model_selection import haversine_np
import json

def select_model(json_file):
    json_file = RESULTS_FILTERED_DIR / json_file

    if not json_file.exists():
        raise FileNotFoundError(f"No JSON file found in {json_file}")
    
    with open(json_file, "r") as f:
        model_data = json.load(f)

    model_name = model_data.get("config", {}).get("name", json_file.stem)

    return model_name, model_data


def load_scaler(filtered=True):
    scaler_file = "scaler_filtered.save" if filtered else "scaler_unfiltered.save"
    scaler = joblib.load(scaler_file)
    return scaler


def load_data(validation=True, filtered=True, batch_size = 512): 
    """
    Returns validation dataloader based on whether the model is filtered or unfiltered.
    """
    scaler = load_scaler(filtered)

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

    return loader, scaler


def best_model(best_model_name, best_model_data, device, filtered = True):
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


def compute_haversine(y, y_pred, all_mse, sorted_indices, scaler):
    n_samples = len(all_mse)

    groups = {
        "Best": sorted_indices[:3],
        "Q1": sorted_indices[n_samples//4:n_samples//4+3],
        "Median": sorted_indices[n_samples//2:n_samples//2+3],
        "Q3": sorted_indices[3*n_samples//4:3*n_samples//4+3],
        "Worst": sorted_indices[-3:]
    }

    group_means_dict = {}

    print("\n=== Haversine Distance Evaluation by Groups ===")

    for group_name, indices in groups.items():
        group_means = []

        print(f"\n### {group_name} group ###")

        for idx in indices:

            y_true_scaled = y[idx]
            y_pred_scaled = y_pred[idx]


            y_true_unscaled = scaler.inverse_transform(y_true_scaled)
            y_pred_unscaled = scaler.inverse_transform(y_pred_scaled)

            # Compute Haversine
            dists_km, mean_hav_km = haversine_np(y_true_unscaled, y_pred_unscaled)

            # Save group mean
            group_means.append(mean_hav_km)

            group_means_dict[group_name] = group_means
            return groups, group_means_dict



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


def plot_samples(x, y, y_pred, all_mse, groups, group_means_dict, scaler):
    for group_name, indices in groups.items():
        group_mean_haversine = np.mean(group_means_dict[group_name])
        print(f"\nPlotting 3 samples from {group_name} group:")
        for idx in indices:
            title = f"{group_name} Sample \n (MSE={all_mse[idx]:.6f} and Haversine distance={group_mean_haversine:.4f} km.)"
            plot_paths(x[idx], y[idx], y_pred[idx], title, scaler=scaler)


if __name__ == "__main__":
    device = training.determine_device()
    filtered = True
    # metrics = ["val_mse", "val_rmse", "val_mae"]
    # best_model_name, best_score, best_model_data = rank_models(RESULTS_FILTERED_DIR, metrics[0])
    model_name, model_data = select_model("small_transformer_results-2025-11-27T10:12:31.json")
    # CHANGE validation to FALSE for the test
    loader, scaler = load_data(validation=True, filtered=filtered)
    model = best_model(model_name, model_data, device=device, filtered=True)
    results = compute_MSE_per_sample(model, loader, device, model_name)

    