import yaml
import subprocess
from tqdm import tqdm
import argparse 

def update_config(config_path, new_values):
    # Read existing config
    with open(config_path, 'r') as config_stream:
        config = yaml.safe_load(config_stream)  # safe_load is preferred over load for security
    
    # Update values
    for key, value in new_values.items():
        if isinstance(key, str) and '.' in key:
            # Handle nested keys like 'training.num_tasks'
            parts = key.split('.')
            current = config
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value
        else:
            config[key] = value
    
    # Write updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

def run_experiments():
    # Parameters to sweep
    num_tasks_values = [2 ** i for i in range(1, 21)]

    config_path = 'conf/base.yaml'
    train_config_path = f'conf/linear_regression.yaml'
    for i in tqdm(range(len(num_tasks_values))):
        # Update config
        updates = {
            'training.num_tasks': num_tasks_values[i]
        }
        update_config(config_path, updates)
            
        # Run training
        print(f"\nRunning num_tasks={num_tasks_values[i]}")
        subprocess.run(['python', 'train.py', '--config', train_config_path])

def run_experiment(list_n):
    config_path = 'conf/base.yaml'
    train_config_path = f'conf/linear_regression.yaml'
    for i in tqdm(range(len(list_n))):
        # Update config
        updates = {
            'training.num_tasks': list_n[i]
        }
        update_config(config_path, updates)
            
        # Run training
        print(f"\nRunning num_tasks={list_n[i]}")
        subprocess.run(['python', 'train.py', '--config', train_config_path])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, nargs='*', required=False, default=[-1],
                       help='Number of tasks to use (should be a power of 2)')
    args = parser.parse_args()
    if -1 in args.n: 
        run_experiments()
    else: 
        run_experiment(args.n)
