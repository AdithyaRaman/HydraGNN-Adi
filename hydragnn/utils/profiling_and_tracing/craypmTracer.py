from collections import defaultdict
import numpy as np

import os

ENERGY_COUNTERS = None
ENERGY_TRACERS = None
ENERGY_CALLS = None

if os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"):
    ## Summit
    LOCAL_RANK = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
elif os.getenv("SLURM_LOCALID"):
    ## CADES
    LOCAL_RANK = int(os.environ["SLURM_LOCALID"])
else:
    LOCAL_RANK = 0

TRACK_NAME = ["forward", "backward", "dataload", "zero_grad", "get_head_indices", "opt_step","train"]

def get_craypmEnergyCounter(rank):
    path = "/sys/cray/pm_counters/"
    
    ## This should be rank//2 when running on frontier
    ## since there are 2 GCD for each GPU.

    with open(f"{path}accel{rank//2}_energy") as f:
    #with open(f"{path}accel{rank}_energy") as f:
        data = f.read()
    energyCounter = int(data.split()[0])
    return energyCounter


def initialize():
    
    global ENERGY_COUNTERS, ENERGY_TRACERS, ENERGY_CALLS

    ENERGY_COUNTERS = {}
    ENERGY_TRACERS = defaultdict(int)
    ENERGY_CALLS = defaultdict(int)
    
def start(name):

    global ENERGY_COUNTERS, ENERGY_TRACERS, ENERGY_CALLS
    if name in TRACK_NAME:
        ENERGY_COUNTERS[name] = get_craypmEnergyCounter(LOCAL_RANK)
        ENERGY_CALLS[name] +=1
    pass
    
def stop(name):

    global ENERGY_COUNTERS, ENERGY_TRACERS
    if name in TRACK_NAME:
        ENERGY_COUNTERS[name] = get_craypmEnergyCounter(LOCAL_RANK) - ENERGY_COUNTERS[name]
        ENERGY_TRACERS[name] += ENERGY_COUNTERS[name]
            
    pass

def enable():
    pass

def disable():
    #nvmlShutdown()
    pass


def reset():
    global ENERGY_COUNTERS, ENERGY_TRACERS
    ENERGY_COUNTERS = {}
    ENERGY_CALLS = defaultdict(int)
    ENERGY_TRACERS = defaultdict(int)
    

def print_device():
    pass

def pr_file(file_path,rank):
    if not os.path.isdir(file_path):
        os.makedirs(file_path, exist_ok = True)
        
    with open(f"{file_path}/cray_dump_p{rank}.csv", mode="w", encoding="utf-8") as file:
        file.write("name,ncalls,total\n")
        for k,v in ENERGY_TRACERS.items():
            file.write(f"{k},{ENERGY_CALLS[k]},{v}\n")
