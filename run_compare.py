import argparse
import json
import matplotlib.pyplot as plt
import os

def load_log(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_metrics(log_files, save_path="training_metrics.png", sample_rate=100):
    plt.figure(figsize=(20, 16))

    metrics = ['psnr', 'psnr2', 'loss', 'num_gaussians']
    titles = ['PSNR', 'PSNR2 (excluding empty cells)', 'Loss', 'Number of Gaussians']
    
    for idx, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, idx)
        
        for log_file in log_files:
            data = load_log(log_file)
            iterations = [entry['iteration'] for entry in data]
            values     = [entry[metric]   for entry in data]
            label = os.path.basename(os.path.dirname(log_file))
            
            # compute windowed averages
            avg_iters = []
            avg_vals  = []
            for start in range(0, len(iterations), sample_rate):
                window_iters = iterations[start:start + sample_rate]
                window_vals  = values[start:start + sample_rate]
                if not window_iters:
                    continue
                avg_iters.append(sum(window_iters) / len(window_iters))
                avg_vals.append(sum(window_vals)  / len(window_vals))
            
            plt.plot(avg_iters, avg_vals, label=label)

        plt.xlabel('Iteration')
        plt.ylabel(metric.capitalize())
        plt.title(titles[idx-1])
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved averaged plot (avg over {sample_rate} iterations) to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot averaged training logs.')
    parser.add_argument('log_files', nargs='+', help='Paths to JSON log files.')
    parser.add_argument('--output',       default='training_metrics.png',
                        help='Output filename for combined plots.')
    parser.add_argument('--sample-rate', type=int, default=5,
                        help='Number of iterations per averaging window (times 20).')
    
    args = parser.parse_args()
    plot_metrics(args.log_files, args.output, sample_rate=args.sample_rate)
