from collections import defaultdict
import numpy as np
import torch
from mpi4py import MPI

try:
    from pynvml import *
    class NVMLpy:
        def __init__(self, rank):
            self.rank = rank
            nvmlInit()

            deviceCount = nvmlDeviceGetCount()
            self.d_handle = nvmlDeviceGetHandleByIndex(0)
            self.device_name = nvmlDeviceGetName(self.d_handle)
            self.energyCounters = {}
            self.energyTracer = defaultdict(list)
            print(f"Initialized NVMLpy:{rank}, \t Device Name:{self.device_name}")

        def start(self, name, cudasync=False, sync=False):
            if cudasync and torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except:
                    pass
            if sync:
                MPI.COMM_WORLD.Barrier()
                
            self.energyCounters[name] = nvmlDeviceGetTotalEnergyConsumption(self.d_handle)

        def stop(self,name, cudasync=False, sync=False):

            if cudasync and torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except:
                    pass
            if sync:
                MPI.COMM_WORLD.Barrier()
                
            self.energyCounters[name] = nvmlDeviceGetTotalEnergyConsumption(self.d_handle) - self.energyCounters[name]
            self.energyTracer[name].append(self.energyCounters[name])

        def pr_file(self,file_path):
            with open(f"{file_path}/nvml_dump_p{self.rank}.csv", mode="w", encoding="utf-8") as file:
                file.write("name, ncalls, mean, total, median, std_dev, max, min\n")
                for k,v in self.energyTracer.items():
                    mean_energy = np.mean(v)
                    total_energy = np.sum(v)
                    median_energy = np.median(v)
                    stdDev = np.std(v)
                    max_energy = np.max(v)
                    min_energy = np.min(v)

                    file.write(f"{k}, {len(v)}, {mean_energy}, {total_energy}, {median_energy}, {stdDev}, {max_energy}, {min_energy}\n")

        def disable(self):
            nvmlShutdown()

except:
    print(f"Error importing pynvml")
    pass
            
