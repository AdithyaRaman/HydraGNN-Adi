from collections import defaultdict
import numpy as np
import torch
from mpi4py import MPI
from pynvml import *
import os

DEVICE_COUNT = None
DEVICE_HANDLER = None
DEVICE_UUID = None
DEVICE_NAME = None

ENERGY_COUNTERS = None
ENERGY_TRACERS = None

def initialize():
    global DEVICE_COUNT, DEVICE_HANDLER, DEVICE_UUID, DEVICE_NAME
    global ENERGY_COUNTERS, ENERGY_TRACERS
    nvmlInit()    
    DEVICE_COUNT = nvmlDeviceGetCount()
    DEVICE_HANDLER = nvmlDeviceGetHandleByIndex(0)
    DEVICE_UUID = nvmlDeviceGetUUID(DEVICE_HANDLER)
    DEVICE_NAME = nvmlDeviceGetName(DEVICE_HANDLER)
    print(f"Initialized for NVML Handler for {DEVICE_NAME}:{DEVICE_UUID}")

    ENERGY_COUNTERS = {}
    ENERGY_TRACERS = defaultdict(list)
    
def start(name, cudasync = False, sync = False):
    global ENERGY_COUNTERS
    if cudasync and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except:
            pass
        if sync:
            MPI.COMM_WORLD.Barrier()
            
    ENERGY_COUNTERS[name] = nvmlDeviceGetTotalEnergyConsumption(DEVICE_HANDLER)
    pass
    
def stop(name, cudasync = False, sync = False):
    global ENERGY_COUNTERS, ENERGY_TRACERS
    if cudasync and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except:
            pass
        if sync:
            MPI.COMM_WORLD.Barrier()

    ENERGY_COUNTERS[name] = nvmlDeviceGetTotalEnergyConsumption(DEVICE_HANDLER) - ENERGY_COUNTERS[name]
    ENERGY_TRACERS[name].append(ENERGY_COUNTERS[name])
    pass

def enable():
    pass

def disable():
    #nvmlShutdown()
    pass


def reset():
    global ENERGY_COUNTERS, ENERGY_TRACERS
    ENERGY_COUNTERS = {}
    ENERGY_TRACERS = defaultdict(list)
    

def print_device():
    print(f"Initialized for NVML Handler for {DEVICE_NAME}:{DEVICE_UUID}")
    print(f"ENERGY_TRACER:{ENERGY_TRACERS}")
    print(f"ENERGY_COUNTER:{ENERGY_COUNTERS}")

def pr_file(file_path,rank):
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
        
    with open(f"{file_path}/nvml_dump_p{rank}.csv", mode="w", encoding="utf-8") as file:
        file.write("name, ncalls, mean, total, median, std_dev, max, min\n")
        for k,v in ENERGY_TRACERS.items():
            mean_energy = np.mean(v)
            total_energy = np.sum(v)
            median_energy = np.median(v)
            stdDev = np.std(v)
            max_energy = np.max(v)
            min_energy = np.min(v)

            file.write(f"{k}, {len(v)}, {mean_energy}, {total_energy}, {median_energy}, {stdDev}, {max_energy}, {min_energy}\n")
            
    nvmlShutdown()
