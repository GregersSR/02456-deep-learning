import dataloader
import joblib
import paths

_, scaler = dataloader.load_train(filter_stationary=True)
joblib.dump(scaler, paths.PROJECT_DIR / 'scaler_filtered.save')
_, scaler = dataloader.load_train(filter_stationary=False)
joblib.dump(scaler, paths.PROJECT_DIR / 'scaler_unfiltered.save')
