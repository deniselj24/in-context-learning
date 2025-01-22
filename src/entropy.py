import argparse
import os 
import json 
import matplotlib.pyplot as plt
import glob 
import numpy as np 
import math
from tqdm import tqdm

def filter_eigenvalues(eigen_list, weight_list, threshold=None):
    filtered_eigen = []
    filtered_weight = []
    #print(np.max(weight_list))
    for eig, w in zip(eigen_list, weight_list):
        if threshold is not None:
            if eig >= threshold and w >= 1e-7:
                filtered_eigen.append(eig)
                filtered_weight.append(w)
        else:
            if w >= 1e-10:
                filtered_eigen.append(eig)
                filtered_weight.append(w)
    #print(filtered_eigen)
    return filtered_eigen, filtered_weight

def gaussian_density(t, values, sigma=1e-5**0.5):
    coeff = 1.0 / np.sqrt(2 * math.pi * sigma**2)
    val = -(values - t) ** 2
    val = val / (2.0 * sigma**2)
    val = np.exp(val)
    density = coeff * val
    return density

def interpolate(weights, values):
    left_boundary = np.mean(np.min(values, axis = 1))-1
    right_boundary= np.mean(np.max(values, axis = 1)) +1
    n_grid = 50000
    grid = np.linspace(left_boundary, right_boundary, n_grid).tolist()
    density_all = np.zeros((10, n_grid))
    weights = np.array(weights)
    values = np.array(values)

    for k in range(10):
        for idx, t in enumerate(grid):
            values_each_v_t = gaussian_density(t, values[k,:])
            density_each_v_t = np.sum(values_each_v_t * weights[k,:])
            density_all[k,idx] = density_each_v_t

    density_avg = np.nanmean(density_all, axis = 0)
    norm_fact = np.sum(density_avg)*(grid[1]- grid[0])
    density_avg /= norm_fact

    return grid, density_avg

def renormalize_weights(filtered_weight, epsilon=1e-12):
    total = sum(filtered_weight)
    if total > 0:
        renormalized_weight = [w / (total + epsilon) for w in filtered_weight]
    else:
        # Handle case where all weights are zero
        renormalized_weight = [0.0 for _ in filtered_weight]
    return renormalized_weight

def compute_spectral_entropy(values_dic, weights_dic, epsilon=1e-12):

    for name in weights_dic.keys(): 
        weights = weights_dic[name]
        values = values_dic[name]
        #print("weights", weights)
        #print("values", values)
        grid, density_avg = interpolate(weights, values)

    print("density", len(density_avg))
    print("grid", len(grid))

    filtered_eigen, filtered_weight = filter_eigenvalues(grid, density_avg)
    # filtered_eigen, renormalized_weight = grid, density_avg
    renormalized_weight = renormalize_weights(filtered_weight)
    p = np.array(renormalized_weight) #+ epsilon  # Avoid log(0)
    print(len(p), len(filtered_eigen))
    spectral_entropy = -np.sum(p * np.log(p))
    print("p min", min(p), "p max", max(p), "eigenvalue min", min(filtered_eigen), "eigenvalue max", max(filtered_eigen))
    weighted_entropy = -np.sum(p * np.log(p) * np.array(filtered_eigen))
    centroid = np.sum(np.array(renormalized_weight) * np.array(filtered_eigen))
    spread = np.sum(np.array(renormalized_weight) * (np.array(filtered_eigen) - centroid)**2)
        
    return spectral_entropy, weighted_entropy, centroid, spread


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", required=True, help="powers of 2")
    args = parser.parse_args()
    data_dir = os.path.expanduser("~/Desktop/files-backup-jan-21-hessian/files/")
    weights_file = "weights_layer.json"
    values_file = "values_layer.json"
    entropy = {}
    for i in args.n:
        n = 2 ** i
        pattern = f"gpt2-4layer-icl-diversity-last-layers-lbl-grad-acc-1-{n}-tasks*"
        subdirs = glob.glob(os.path.join(data_dir, pattern))
        print(subdirs)
        for subdir in subdirs:  
            dirname = os.path.basename(subdir)
            ckpt = int(dirname.split('ckpt_')[1])
            with open(os.path.join(subdir, weights_file), 'r') as f:
                weights = json.load(f)
            with open(os.path.join(subdir, values_file), 'r') as f:
                values = json.load(f)
            spectral_entropy, weighted_entropy, centroid, spread = compute_spectral_entropy(weights, values)   
            print(f"# of Tasks: {n}, ckpt: {ckpt}, spectral entropy: {spectral_entropy}, weighted entropy: {weighted_entropy}, centroid: {centroid}, spread: {spread}")
            for key in ["spectral_entropy", "weighted_entropy", "centroid", "spread"]:
                if key not in entropy:
                    entropy[key] = {}
                if n not in entropy[key]:
                    entropy[key][n] = {}
                entropy[key][n][ckpt] = eval(key)

    with open(os.path.join(data_dir, "entropy.json"), "w") as f:
        json.dump(entropy, f)

    # plot weighted entropy
    for i in args.n:
        n = 2 ** i
        iterations = sorted(entropy["weighted_entropy"][n].keys())
        entropies = [entropy["weighted_entropy"][n][i] for i in iterations]
        plt.plot(iterations, entropies, 
                label=f"{n} Tasks", 
                marker='o',
                markersize=4,
                linewidth=2)

    plt.xlabel("Number of training iterations")
    plt.ylabel("Weighted Entropy")
    plt.legend()
    plt.savefig(os.path.join(data_dir, f"weighted_entropy.png"))
    plt.close()

    # plot spectral entropy
    for i in args.n:
        n = 2 ** i
        iterations = sorted(entropy["spectral_entropy"][n].keys())
        entropies = [entropy["spectral_entropy"][n][i] for i in iterations]
        plt.plot(iterations, entropies, 
                label=f"{n} Tasks", 
                marker='o',
                markersize=4,
                linewidth=2)

    plt.xlabel("Number of training iterations")
    plt.ylabel("Spectral Entropy")
    plt.legend()
    plt.savefig(os.path.join(data_dir, f"spectral_entropy.png"))
    plt.close()

    # plot spread
    for i in args.n:
        n = 2 ** i
        iterations = sorted(entropy["spread"][n].keys())
        entropies = [entropy["spread"][n][i] for i in iterations]
        plt.plot(iterations, entropies, 
                label=f"{n} Tasks", 
                marker='o',
                markersize=4,
                linewidth=2)

    plt.xlabel("Number of training iterations")
    plt.ylabel("Spread")
    plt.legend()
    plt.savefig(os.path.join(data_dir, f"spread.png"))
    plt.close()

    # plot centroid
    for i in args.n:
        n = 2 ** i
        iterations = sorted(entropy["centroid"][n].keys())
        entropies = [entropy["centroid"][n][i] for i in iterations]
        plt.plot(iterations, entropies, 
                label=f"{n} Tasks", 
                marker='o',
                markersize=4,
                linewidth=2)

    plt.xlabel("Number of training iterations")
    plt.ylabel("Centroid")
    plt.legend()
    plt.savefig(os.path.join(data_dir, f"centroid.png"))
    plt.close()


