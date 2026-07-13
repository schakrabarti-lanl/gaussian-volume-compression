import argparse
import json
import matplotlib.pyplot as plt
import os

def load_log(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_metrics(log_files, save_path="training_metrics.png", max_iter=None):
    plt.figure(figsize=(20, 16))

    metrics = ['psnr', 'loss', 'num_gaussians']
    titles = ['PSNR', 'Loss', 'Number of Gaussians']
    
    for idx, metric in enumerate(metrics, 1):
        ax = plt.subplot(2, 2, idx)
        
        for log_file in log_files:
            data = load_log(log_file)
            if max_iter is not None:
                data = [entry for entry in data if entry['iteration'] <= max_iter]
            iterations = [entry['iteration'] for entry in data]
            values = [entry[metric] for entry in data]
            label = os.path.basename(os.path.dirname(log_file))

            ax.plot(iterations, values, label=label)

        ax.set_xlabel('Iteration')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(titles[idx-1])

        if metric == 'loss':
            ax.set_yscale('log')

        ax.legend()
        ax.grid(True, which='both')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved combined plot to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training logs.')
    parser.add_argument('log_files', nargs='+', help='Paths to JSON log files.')
    parser.add_argument('--output', default='training_metrics.png', help='Output filename for combined plots.')
    parser.add_argument('--max-iter', type=int, default=None, help='Only show iterations up to this value.')

    args = parser.parse_args()

    plot_metrics(args.log_files, args.output, max_iter=args.max_iter)
