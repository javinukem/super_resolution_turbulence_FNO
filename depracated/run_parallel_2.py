# run_parallel_grid.py
from autocvd import autocvd

# Set the number of GPUs you want to use
DESIRED_NUM_GPUS = 3  # Change this to the number of GPUs you want
autocvd(num_gpus=DESIRED_NUM_GPUS)

from multiprocessing import Process, Lock, Manager
from itertools import product
import yaml
from train_one_model import train_on_gpu
import copy
import torch
import sys
import os
import time
import psutil
import subprocess
from datetime import datetime
from multiprocessing import Value

num_gpus = DESIRED_NUM_GPUS  # Use the configured number instead of device_count()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

class ProgressTracker:
    def __init__(self):
        self.manager = Manager()
        self.process_status = self.manager.dict()
        self.start_time = time.time()
        self.lock = Lock()
    
    def update_progress(self, process_id, status, details=""):
        with self.lock:
            self.process_status[process_id] = {
                'status': status,
                'details': details,
                'timestamp': time.time()
            }
    
    def get_summary(self):
        with self.lock:
            status_dict = dict(self.process_status)
            
        total = len(status_dict)
        completed = sum(1 for v in status_dict.values() if v['status'] == 'completed')
        running = sum(1 for v in status_dict.values() if v['status'] == 'running')
        failed = sum(1 for v in status_dict.values() if v['status'] == 'failed')
        elapsed = time.time() - self.start_time
        
        return {
            'total': total,
            'completed': completed,
            'running': running,
            'failed': failed,
            'elapsed': elapsed
        }

def check_gpu_available(gpu_id, verbose=False):
    """Check if GPU is actually free by examining processes using it"""
    try:
        # Method 1: Check using torch if available
        if torch.cuda.is_available():
            # Try to allocate a small tensor to test if GPU is accessible
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
                # Check memory usage
                memory_allocated = torch.cuda.memory_allocated(gpu_id)
                memory_cached = torch.cuda.memory_reserved(gpu_id)
                # Consider available if very little memory is used (< 100MB)
                return memory_allocated < 100 * 1024 * 1024 and memory_cached < 100 * 1024 * 1024
            return False
    except Exception as e:
        if verbose:
            print(f"Error checking GPU {gpu_id}: {e}")
        # If we can't check, assume it's available to avoid blocking
        return True

def wait_for_gpu_free(gpu_id, max_wait=60, check_interval=2, verbose=False):
    """Wait for GPU to become available, with timeout"""
    start_time = time.time()
    
    # First check - if it's available immediately, return
    if check_gpu_available(gpu_id, verbose):
        return True
    
    if verbose:
        print(f"GPU {gpu_id} appears busy, checking availability...")
    
    while time.time() - start_time < max_wait:
        if check_gpu_available(gpu_id, verbose):
            if verbose:
                print(f"GPU {gpu_id} is now available!")
            return True
        time.sleep(check_interval)
    
    if verbose:
        print(f"GPU {gpu_id} still appears busy after {max_wait}s, proceeding anyway...")
    return False

def safe_train_on_gpu(gpu_id, config, gpu_lock):
    """Training function that ensures exclusive GPU access"""
    with gpu_lock:
        # Quick check with shorter timeout
        if not wait_for_gpu_free(gpu_id, max_wait=30):
            print(f"GPU {gpu_id} availability check timed out, proceeding...")
        
        print(f"Process {os.getpid()} acquired GPU {gpu_id}")
        
        # Call your actual training function
        train_on_gpu(gpu_id, config)
        
        print(f"Process {os.getpid()} finished with GPU {gpu_id}")

# Alternative approach using a GPU pool manager with process limits

class GPUPool:
    def __init__(self, num_gpus, max_processes_per_gpu=1, verbose=False):
        self.manager = Manager()
        self.num_gpus = num_gpus
        self.max_processes_per_gpu = max_processes_per_gpu
        self.verbose = verbose
        
        # Use a Namespace to hold shared state atomically
        self.shared = self.manager.Namespace()
        self.shared.active_processes = [0] * num_gpus
        self.shared.last_assigned_gpu = -1
        
        self.pool_lock = Lock()

    def get_gpu(self):
        """Get an available GPU ID, blocking until one has capacity"""
        while True:
            with self.pool_lock:
                num_gpus = self.num_gpus
                max_proc = self.max_processes_per_gpu
                start_gpu = (self.shared.last_assigned_gpu + 1) % num_gpus

                # Search GPUs circularly starting from next GPU after last assigned
                for i in range(num_gpus):
                    gpu_id = (start_gpu + i) % num_gpus
                    if self.shared.active_processes[gpu_id] < max_proc:
                        self.shared.active_processes[gpu_id] += 1
                        self.shared.last_assigned_gpu = gpu_id
                        if self.verbose:
                            print(f"[{os.getpid()}] Assigned GPU {gpu_id}. Running on it: {self.shared.active_processes[gpu_id]}/{max_proc}")
                        return gpu_id
            # No GPUs free, sleep and retry
            time.sleep(1)

    def release_gpu(self, gpu_id):
        with self.pool_lock:
            if self.shared.active_processes[gpu_id] > 0:
                self.shared.active_processes[gpu_id] -= 1
                if self.verbose:
                    print(f"[{os.getpid()}] Released GPU {gpu_id}. Running on it: {self.shared.active_processes[gpu_id]}/{self.max_processes_per_gpu}")

def pool_train_on_gpu(gpu_pool, config_dict, process_id, progress_tracker):
    """Training function using GPU pool with progress tracking"""
    gpu_id = gpu_pool.get_gpu()
    try:
        progress_tracker.update_progress(process_id, 'running', f'GPU {gpu_id}')
        
        # Create a deep copy of the config to avoid sharing issues
        config_copy = copy.deepcopy(config_dict)
        
        # Call your actual training function with the copied config
        train_on_gpu(gpu_id, config_copy)
        
        progress_tracker.update_progress(process_id, 'completed', f'GPU {gpu_id}')
        
    except Exception as e:
        progress_tracker.update_progress(process_id, 'failed', f'GPU {gpu_id}: {str(e)}')
        raise
    finally:
        gpu_pool.release_gpu(gpu_id)

def progress_monitor(progress_tracker, total_processes, update_interval=30):
    """Monitor and print progress every update_interval seconds"""
    while True:
        summary = progress_tracker.get_summary()
        
        if summary['total'] == 0:
            time.sleep(update_interval)
            continue
            
        elapsed_hours = summary['elapsed'] / 3600
        completion_rate = summary['completed'] / summary['total'] * 100
        
        print(f"\n{'='*60}")
        print(f"PROGRESS UPDATE - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        print(f"Total processes: {summary['total']}")
        print(f"Completed: {summary['completed']} ({completion_rate:.1f}%)")
        print(f"Running: {summary['running']}")
        print(f"Queued: {summary['total'] - summary['completed'] - summary['running']}")
        print(f"Elapsed time: {elapsed_hours:.1f} hours")
        
        if summary['completed'] > 0:
            avg_time_per_job = summary['elapsed'] / summary['completed']
            remaining_jobs = summary['total'] - summary['completed']
            eta_seconds = remaining_jobs * avg_time_per_job
            eta_hours = eta_seconds / 3600
            print(f"Estimated time remaining: {eta_hours:.1f} hours")
        
        print(f"{'='*60}\n")
        
        # Stop monitoring when all jobs are complete
        if summary['completed'] >= total_processes:
            break
            
        time.sleep(update_interval)

# Load and prepare configs
base_config = yaml.safe_load(open("config.yaml", "r"))
grid_config = base_config["grid_fno2"]

# Build parameter grid
keys = list(grid_config.keys())
values = list(grid_config.values())
grid = list(product(*values))

# Construct full configs
configs = []
for combo in grid:
    config = copy.deepcopy(base_config)
    for k, v in zip(keys, combo):
        if k == "chan":
            config["fno_2"]["n_channels"] = v
        elif k == "res":
            config["fno_2"]["n_residual_blocks"] = v
        elif k == "op":
            config["fno_2"]["n_operator_blocks"] = v
        elif k == "modes":
            config["fno_2"]["modes"] = v
        elif k == "last_layer_constraint":
            config["fno_2"]["last_layer_constraint"] = v
    configs.append(config)

if __name__ == "__main__":
    # Configuration
    MAX_PROCESSES_PER_GPU = 2  # Adjust this number as needed
    PROGRESS_UPDATE_INTERVAL = 30  # seconds between progress updates
    VERBOSE = True  # Set to True for detailed GPU assignment logs
    
    print(f"Starting grid search with {len(configs)} configurations")
    print(f"Using {num_gpus} GPUs with max {MAX_PROCESSES_PER_GPU} processes per GPU")
    print(f"Progress updates every {PROGRESS_UPDATE_INTERVAL} seconds")
    print(f"Verbose logging: {VERBOSE}")
    
    # Print a few example configs to verify they're different
    print("\nFirst few configurations:")
    for i, config in enumerate(configs[:3]):
        print(f"Config {i}: channels={config['fno_2']['n_channels']}, "
              f"modes={config['fno_2']['modes']}, "
              f"res_blocks={config['fno_2']['n_residual_blocks']}, "
              f"op_blocks={config['fno_2']['n_operator_blocks']}, "
              f"last_layer_constraint={config['fno_2']['last_layer_constraint']}")
    
    print("="*60)
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker()
    
    gpu_pool = GPUPool(num_gpus, MAX_PROCESSES_PER_GPU, verbose=VERBOSE)
    
    # Start progress monitoring in a separate process
    monitor_process = Process(target=progress_monitor, args=(progress_tracker, len(configs), PROGRESS_UPDATE_INTERVAL))
    monitor_process.start()
    
    processes = []
    for i, config in enumerate(configs):
        # Convert config to a serializable format and pass a copy
        config_copy = copy.deepcopy(config)
        p = Process(target=pool_train_on_gpu, args=(gpu_pool, config_copy, i, progress_tracker))
        p.start()
        processes.append(p)
        
        # Print config details for every process launch
        print(f"Launched process {i+1}/{len(configs)} - Config: channels={config['fno_2']['n_channels']}, "
              f"modes={config['fno_2']['modes']}, "
              f"res_blocks={config['fno_2']['n_residual_blocks']}, "
              f"op_blocks={config['fno_2']['n_operator_blocks']}, "
              f"last_layer_constraint={config['fno_2']['last_layer_constraint']}")
    
    print(f"\nAll {len(processes)} processes launched! Monitor will show progress updates.")
    
    # Wait for all training processes to complete
    for p in processes:
        p.join()
    
    # Stop the monitor
    monitor_process.terminate()
    monitor_process.join()
    
    # Final summary
    final_summary = progress_tracker.get_summary()
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total jobs: {final_summary['total']}")
    print(f"Completed: {final_summary['completed']}")
    print(f"Total time: {final_summary['elapsed']/3600:.1f} hours")
    if final_summary['completed'] > 0:
        print(f"Average time per job: {final_summary['elapsed']/final_summary['completed']:.1f} seconds")
    print("All training jobs completed!")
