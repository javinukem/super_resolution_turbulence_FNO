# run_parallel_grid.py

# Set the number of GPUs you want to use
DESIRED_NUM_GPUS = 5  # Change this to the number of GPUs you want

from multiprocessing import Process, Lock, Manager, Queue
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


def get_available_gpus(mem_threshold=500):
    """
    Finds GPUs with memory usage below a certain threshold.

    Args:
        mem_threshold (int): The maximum memory usage in MiB to be considered "free".

    Returns:
        list: A list of integer IDs of the available GPUs.
    """
    try:
        # Command to get GPU index and memory usage, sorted by index
        command = "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        
        available_gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            index, memory_used = line.split(',')
            if int(memory_used.strip()) < mem_threshold:
                available_gpus.append(int(index.strip()))
        
        return available_gpus
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error calling nvidia-smi: {e}")
        print("Could not determine GPU availability. Assuming no GPUs are free.")
        return []
    

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

# Simple and reliable GPU pool with strict process limits
class GPUPool:
    def __init__(self, available_gpu_ids, max_processes_per_gpu=1, verbose=False):
        if not available_gpu_ids:
            raise ValueError("Cannot initialize GPUPool with no available GPUs.")
            
        self.available_gpus = available_gpu_ids
        self.max_processes_per_gpu = max_processes_per_gpu
        self.verbose = verbose
        
        self.manager = Manager()
        # Use a dictionary to map physical GPU IDs to their queues
        self.gpu_queues = self.manager.dict()
        self.active_processes = self.manager.dict()
        
        for gpu_id in self.available_gpus:
            # Create a queue for each available GPU
            gpu_queue = self.manager.Queue()
            for _ in range(max_processes_per_gpu):
                gpu_queue.put(gpu_id)
            self.gpu_queues[gpu_id] = gpu_queue
            self.active_processes[gpu_id] = 0
            
        self.pool_lock = Lock()
        
        if verbose:
            print(f"Initialized GPU pool for physical GPUs: {self.available_gpus}")
            print(f"Total available slots: {len(self.available_gpus) * max_processes_per_gpu}")

    def get_gpu(self):
        """Get an available GPU ID - this will block until one has capacity"""
        if self.verbose:
            print(f"[{os.getpid()}] Requesting GPU...")
        
        while True:
            # Iterate through the specific GPUs we are allowed to use
            for gpu_id in self.available_gpus:
                try:
                    # Non-blocking attempt to get a slot from this GPU's queue
                    assigned_gpu = self.gpu_queues[gpu_id].get_nowait()
                    
                    with self.pool_lock:
                        self.active_processes[gpu_id] += 1
                    
                    if self.verbose:
                        print(f"[{os.getpid()}] Assigned physical GPU {assigned_gpu}")
                    return assigned_gpu
                except:
                    # This GPU's queue is empty, try the next one
                    continue
            
            # If all queues were empty, wait a moment and retry
            time.sleep(0.1)

    def release_gpu(self, gpu_id):
        """Release a GPU slot back to the pool"""
        # Put the GPU ID back into its specific queue
        self.gpu_queues[gpu_id].put(gpu_id)
        
        with self.pool_lock:
            if self.active_processes[gpu_id] > 0:
                self.active_processes[gpu_id] -= 1
        
        if self.verbose:
            print(f"[{os.getpid()}] Released physical GPU {gpu_id}")
    
    # The get_status method also needs a minor tweak to iterate over available_gpus
    def get_status(self):
        """Get current status of all GPUs"""
        with self.pool_lock:
            status = {}
            for gpu_id in self.available_gpus:
                queue_size = self.gpu_queues[gpu_id].qsize()
                status[gpu_id] = {
                    'active': self.active_processes[gpu_id],
                    'max': self.max_processes_per_gpu,
                    'available_slots': queue_size
                }
            
            total_active = sum(self.active_processes.values())
            total_slots = len(self.available_gpus) * self.max_processes_per_gpu
            
            return {
                'gpu_status': status,
                'total_active': total_active,
                'total_slots': total_slots,
            }

def pool_train_on_gpu(gpu_pool, config_dict, process_id, progress_tracker):
    """Training function using GPU pool with progress tracking"""
    gpu_id = None
    try:
        # Get GPU from pool (this will block until one is available)
        gpu_id = gpu_pool.get_gpu()
        progress_tracker.update_progress(process_id, 'running', f'GPU {gpu_id}')
        
        # Create a deep copy of the config to avoid sharing issues
        config_copy = copy.deepcopy(config_dict)
        
        # Call your actual training function with the copied config
        train_on_gpu(gpu_id, config_copy)
        
        progress_tracker.update_progress(process_id, 'completed', f'GPU {gpu_id}')
        
    except Exception as e:
        error_msg = f'GPU {gpu_id if gpu_id is not None else "unknown"}: {str(e)}'
        progress_tracker.update_progress(process_id, 'failed', error_msg)
        print(f"Process {process_id} failed: {error_msg}")
        raise
    finally:
        # Always release the GPU, even if there was an error
        if gpu_id is not None:
            gpu_pool.release_gpu(gpu_id)

def progress_monitor(progress_tracker, gpu_pool, total_processes, update_interval=30):
    """Monitor and print progress every update_interval seconds"""
    while True:
        summary = progress_tracker.get_summary()
        gpu_status = gpu_pool.get_status()
        
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
        print(f"Failed: {summary['failed']}")
        print(f"Queued: {summary['total'] - summary['completed'] - summary['running'] - summary['failed']}")
        print(f"Elapsed time: {elapsed_hours:.1f} hours")
        
        # GPU status
        print(f"\nGPU Status:")
        for gpu_id, status in gpu_status['gpu_status'].items():
            print(f"  GPU {gpu_id}: {status['active']}/{status['max']} processes "
                  f"({status['available_slots']} slots available)")
        print(f"Total active processes: {gpu_status['total_active']}/{gpu_status['total_slots']}")
        print(f"Total available slots: {gpu_status['total_slots']}")
        
        if summary['completed'] > 0:
            avg_time_per_job = summary['elapsed'] / summary['completed']
            remaining_jobs = summary['total'] - summary['completed'] - summary['failed']
            eta_seconds = remaining_jobs * avg_time_per_job
            eta_hours = eta_seconds / 3600
            print(f"Estimated time remaining: {eta_hours:.1f} hours")
        
        print(f"{'='*60}\n")
        
        # Stop monitoring when all jobs are complete
        if summary['completed'] + summary['failed'] >= total_processes:
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
    MAX_PROCESSES_PER_GPU = 1
    # Memory threshold (in MiB) to consider a GPU "free"
    GPU_MEMORY_THRESHOLD = 50
    VERBOSE = False
    
    # 1. Scan for GPUs that are not being used by other users
    print("Scanning for available GPUs...")
    available_gpu_ids = get_available_gpus(mem_threshold=GPU_MEMORY_THRESHOLD)
    
    if not available_gpu_ids:
        print("ERROR: No free GPUs found that meet the criteria. Exiting.")
        sys.exit(1)
        
    print(f"Found {len(available_gpu_ids)} free GPUs: {available_gpu_ids}")
    print(f"Starting grid search with {len(configs)} configurations.")
    print(f"Using GPUs {available_gpu_ids} with max {MAX_PROCESSES_PER_GPU} processes per GPU.")
    print("-" * 60)
    
    # 2. Initialize the pool with ONLY the free GPUs
    progress_tracker = ProgressTracker()
    gpu_pool = GPUPool(
        available_gpu_ids=available_gpu_ids,
        max_processes_per_gpu=MAX_PROCESSES_PER_GPU,
        verbose=VERBOSE
    )
    
    # 3. Launch processes (the rest of your code is the same)
    # Start progress monitoring in a separate process
    monitor_process = Process(target=progress_monitor, 
                              args=(progress_tracker, gpu_pool, len(configs)))
    monitor_process.start()
    
    # Launch all processes
    processes = []
    for i, config in enumerate(configs):
        p = Process(target=pool_train_on_gpu, args=(gpu_pool, config, i, progress_tracker))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
        
    monitor_process.terminate()
    monitor_process.join()

    # ... (final summary printout)    
    # Final summary
    final_summary = progress_tracker.get_summary()
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total jobs: {final_summary['total']}")
    print(f"Completed: {final_summary['completed']}")
    print(f"Failed: {final_summary['failed']}")
    print(f"Total time: {final_summary['elapsed']/3600:.1f} hours")
    if final_summary['completed'] > 0:
        print(f"Average time per job: {final_summary['elapsed']/final_summary['completed']:.1f} seconds")
    print("All training jobs completed!")