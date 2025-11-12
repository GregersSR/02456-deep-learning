from dataloader import AisDataset
import matplotlib.pyplot as plt
import numpy as np

dataset = AisDataset.train() # change this to AisDataset.val() to use the val set

# We perform classical linear regression (least squares) on each segment individually, then compute the average MSE over all segments
X = dataset.tensors[0].reshape(len(dataset), -1).numpy()  # N x 120
Y = dataset.tensors[1].reshape(len(dataset), -1).numpy()  # N x 40
w, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None) # Solve the least-squares problem (Closed form: w = (X^T X)^(-1) X^T Y)
Y_pred = X @ w # Make predictions
mse = np.mean((Y - Y_pred)**2) # Compute the average MSE
print("Average MSE across all windows:", mse)

# Function that plots the ith window's input, true output, and predicted output
def plot_sample_prediction(x, y, y_pred, i):
    plt.figure(figsize=(8,6))
    plt.scatter(x[:, 1], x[:, 0], c='blue', s=10, label='Known Route (Input)') # first 30 steps
    plt.scatter(y[:, 1], y[:, 0], c='green', s=30, label='True Future Route')  # next 10 steps
    plt.scatter(y_pred[:, 1], y_pred[:, 0], c='red', s=30, label='Predicted Future Route', marker='x') # predicted 10 steps
    plt.title(f'Vessel Trajectory Prediction (Window {i})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.show()

# Pick a random window i to visualize
i = np.random.randint(0, len(X))
x_window = X[i].reshape(30, 4)
y_window = Y[i].reshape(10, 4)
y_pred_window = Y_pred[i].reshape(10, 4)

plot_sample_prediction(x_window, y_window, y_pred_window, i)