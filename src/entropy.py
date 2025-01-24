import argparse
import os 
import json 
import matplotlib.pyplot as plt
import glob 
import numpy as np 
import math
from tqdm import tqdm

def gaussian_density(t, values, sigma=1e-5**0.5):
    coeff = 1.0 / np.sqrt(2 * math.pi * sigma**2)
    val = -(values - t) ** 2
    val = val / (2.0 * sigma**2)
    val = np.exp(val)
    density = coeff * val
    return density

def get_spectral_density(weights, values, n_lanczos=10):
    #weights = np.array(weights)
    #values = np.array(values)
    density_all = np.zeros((n_lanczos, values.shape[1]))

    for k in range(n_lanczos):
        for idx, t in enumerate(values):
            values_each_v_t = gaussian_density(t, values[k,:])
            density_each_v_t = np.sum(values_each_v_t * weights[k,:])
            density_all[k,idx] = density_each_v_t
    # average across lanczos 
    # density_avg = np.nanmean(density_all, axis = 0)
    return density_all

def compute_spectral_entropy(values_dic, weights_dic, epsilon=1e-12):
    spectral_entropy = {}
    weighted_entropy = {}
    centroid = {}
    spread = {}

    for name in weights_dic.keys(): 
        weights = np.array(weights_dic[name])
        values = np.array(values_dic[name])
        #print("weights", weights)
        #print("values", values)
        density = get_spectral_density(weights, values)
        # renormalize density which is a np array of (n_lanczos=10, 100)
        total = np.sum(density, axis=1).reshape((10, 1))
        normalized_density = density / total

        # filtered_eigen, filtered_weight = filter_eigenvalues(values, density)
        # renormalized_weight = renormalize_weights(filtered_weight)
        # p = np.array(renormalized_weight) + epsilon  # Avoid log(0)
        p = np.array(normalized_density) + epsilon
        print(p.shape, np.array(values).shape)
        spectral_entropy[name] = -np.sum(np.mean(p * np.log(p), axis=0))
        #print("log", np.log(p))
        #print("product", p * np.log(p))
        #print("mean", np.mean(p * np.log(p), axis=0))
        # print("p min", min(p), "p max", max(p), "eigenvalue min", min(filtered_eigen), "eigenvalue max", max(filtered_eigen))
        weighted_entropy[name] = -np.sum(p * np.log(p) * np.array(values))
        centroid[name] = np.sum(np.array(normalized_density) * np.array(values))
        spread[name] = np.sum(np.array(normalized_density) * (np.array(values) - centroid[name])**2)

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
                # spectral_entropy, etc. is dictionary with layer name as key 
    # print(entropy)
    with open(os.path.join(data_dir, "entropy.json"), "w") as f:
        json.dump(entropy, f)

    for layer_name in entropy["weighted_entropy"][2][0].keys():
        # plot weighted entropy
        for i in args.n:
            n = 2 ** i
            iterations = sorted(entropy["weighted_entropy"][n].keys())
            print("debugging", iterations)
            entropies = [entropy["weighted_entropy"][n][j][layer_name] for j in iterations]
            plt.plot(iterations, entropies, 
                    label=f"{n} Tasks", 
                    marker='o',
                    markersize=4,
                    linewidth=2)

        plt.xlabel("Number of training iterations")
        plt.ylabel("Weighted Entropy")
        plt.legend()
        plt.savefig(os.path.join(data_dir, f"weighted_entropy_{layer_name}.png"))
        plt.close()

        # plot spectral entropy
        for i in args.n:
            n = 2 ** i
            iterations = sorted(entropy["spectral_entropy"][n].keys())
            entropies = [entropy["spectral_entropy"][n][j][layer_name] for j in iterations]
            plt.plot(iterations, entropies, 
                    label=f"{n} Tasks", 
                    marker='o',
                    markersize=4,
                    linewidth=2)

        plt.xlabel("Number of training iterations")
        plt.ylabel("Spectral Entropy")
        plt.legend()
        plt.savefig(os.path.join(data_dir, f"spectral_entropy_{layer_name}.png"))
        plt.close()

        # plot spread
        for i in args.n:
            n = 2 ** i
            iterations = sorted(entropy["spread"][n].keys())
            entropies = [entropy["spread"][n][j][layer_name] for j in iterations]
            plt.plot(iterations, entropies, 
                    label=f"{n} Tasks", 
                    marker='o',
                    markersize=4,
                    linewidth=2)

        plt.xlabel("Number of training iterations")
        plt.ylabel("Spread")
        plt.legend()
        plt.savefig(os.path.join(data_dir, f"spread_{layer_name}.png"))
        plt.close()

        # plot centroid
        for i in args.n:
            n = 2 ** i
            iterations = sorted(entropy["centroid"][n].keys())
            entropies = [entropy["centroid"][n][j][layer_name] for j in iterations]
            plt.plot(iterations, entropies, 
                    label=f"{n} Tasks", 
                    marker='o',
                    markersize=4,
                    linewidth=2)

        plt.xlabel("Number of training iterations")
        plt.ylabel("Centroid")
        plt.legend()
        plt.savefig(os.path.join(data_dir, f"centroid_{layer_name}.png"))
        plt.close()


