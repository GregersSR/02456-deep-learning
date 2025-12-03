import json
import matplotlib.pyplot as plt
import numpy as np


# Rank the models based on metric value
def rank_models(results_dir, metric="val_mse"):
    results = []
    
    for file in results_dir.glob("*.json"):
        with open(file, "r") as f:
            data = json.load(f)
        
        name = data.get("config", {}).get("name", file.stem)
        metric_values = data.get("history", {}).get(metric, [])
        if not metric_values:
            continue

        best_value = min(metric_values)
        results.append((name, best_value, data))

    results.sort(key=lambda x: x[1])

    if metric == "val_mse":
        metric = "MSE"
    if metric == "val_rmse":
        metric = "RSME"
    if metric == "val_mae":
        metric = "MAE"

    print(f"Best validation {metric} per model (sorted): \n")
    for i, (name, value, _) in enumerate(results,1):
        print(f"{i}. {name:30s} best {metric} = {value:.6f}")
    
    # get best model
    best_model_name, best_score, best_model_data = results[0] 
    print("\nOverall best model:")
    print(f"{best_model_name} with best validation {metric} = {best_score:.6}")

    return best_model_name, best_score, best_model_data


# Plot losses
def plot_model_losses(best_model_data):
    best_model_name = best_model_data["config"]["name"]
    history = best_model_data["history"]
    epochs = range(1, len(history["val_mse"]) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history["train_mse"], label='Train MSE', marker='o', markersize=3)
    plt.plot(epochs, history["val_mse"], label='Validation MSE', marker='o', markersize=3)
    plt.plot(epochs, history["train_rmse"], label='Train RMSE', linestyle='--')
    plt.plot(epochs, history["val_rmse"], label='Validation RMSE', linestyle='--')
    plt.plot(epochs, history["train_mae"], label='Train MAE', linestyle=':')
    plt.plot(epochs, history["val_mae"], label='Validation MAE', linestyle=':')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training & Validation Losses for {best_model_name} (filtered)')
    plt.legend()
    plt.grid(True)
    plt.show()


# Haversine formula to calculate from long lats to meters
def haversine_np(true_lonlat, pred_lonlat, radius_km=6371.0):
    """
    true_lonlat, pred_lonlat: Arrays of Form (T, 2) with [lon_deg, lat_deg] in degree.
    """
    # recalc long and lat to radiants
    lon_true = np.radians(true_lonlat[:, 0])
    lat_true = np.radians(true_lonlat[:, 1])
    lon_pred = np.radians(pred_lonlat[:, 0])
    lat_pred = np.radians(pred_lonlat[:, 1])

    dlon = lon_pred - lon_true 
    dlat = lat_pred - lat_true

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_true) * np.cos(lat_pred) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    distances_km = radius_km * c
    mean_distance_km = distances_km.mean()

    return distances_km, mean_distance_km