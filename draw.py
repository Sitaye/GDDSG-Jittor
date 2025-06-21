import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_curve(file_path):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    plot = sns.lineplot(data=data, x='task', y='min_mse_loss', marker='o')
    
    plot.set_title('Minimum MSE Loss per Task')
    plot.set_xlabel('Task ID')
    plot.set_ylabel('Minimum MSE Loss')
    
    max_task = int(data['task'].max())
    plt.xticks(ticks=range(max_task + 1))
    
    output_path = file_path.replace('.csv', '.png')
    plt.savefig(output_path, dpi=300)
    
    print(f"Loss curve plot saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot loss curve from a CSV file.")
    parser.add_argument("file", type=str, help="Path to the loss history CSV file.")
    
    args = parser.parse_args()
    
    plot_curve(args.file)