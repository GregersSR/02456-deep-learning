import matplotlib.pyplot as plt

def plot_paths(x_sample, y_true, y_pred, idx):
    # Past  
    x_sample = x_sample.reshape(30, 2) # reshape back
    # True future
    y_true_sample = y_true
    # Predicted future
    y_pred_sample = y_pred

    plt.figure(figsize=(6,6))
    plt.plot(x_sample[:,0], x_sample[:,1], 'bo-', label='Past')         
    plt.plot(y_true_sample[:,0], y_true_sample[:,1], 'go-', label='True')   
    plt.plot(y_pred_sample[:,0], y_pred_sample[:,1], 'ro--', label='Predicted') 
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title(f'Trajectory Sample {idx}')
    plt.legend()
    plt.show()
