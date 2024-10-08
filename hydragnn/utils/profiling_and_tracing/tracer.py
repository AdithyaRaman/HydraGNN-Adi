""" 
This is a tracer package to act as a wrapper to execute gptl and/or scorep.
"""

from __future__ import absolute_import
from functools import wraps
from contextlib import contextmanager

from abc import ABC, abstractmethod
import torch
from mpi4py import MPI


class Tracer(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def start(self, name):
        pass

    @abstractmethod
    def stop(self, name):
        pass

    @abstractmethod
    def enable(self):
        pass

    @abstractmethod
    def disable(self):
        pass


try:
    import gptl4py as gp

    class GPTLTracer(Tracer):
        def __init__(self, **kwargs):
            gp.initialize()

        def start(self, name):
            gp.start(name)

        def stop(self, name):
            gp.stop(name)

        def enable(self):
            gp.enable()

        def disable(self):
            gp.disable()

        def reset(self):
            gp.reset()


except:
    pass


try:
    import hydragnn.utils.profiling_and_tracing.amdTracer as amd

    class AMDTracer(Tracer):
        def __init__(self, **kwargs):
            amd.initialize()

        def start(self, name):
            amd.start(name)

        def stop(self, name):
            amd.stop(name)

        def enable(self):
            pass

        def disable(self):
            pass

        def reset(self):
            pass

except:
    print(f"Error importing AMD Tracer")
    pass


try:
    import hydragnn.utils.profiling_and_tracing.nvidiaTracer as et

    class NVIDIATracer(Tracer):
        def __init__(self, **kwargs):
            et.initialize()

        def start(self, name):
            et.start(name)

        def stop(self, name):
            et.stop(name)

        def enable(self):
            et.enable()

        def disable(self):
            et.disable()

        def reset(self):
            et.reset()
except:
    print(f"Error importing NVIDIATracer")
    pass

try:
    import hydragnn.utils.profiling_and_tracing.craypmTracer as ct

    class CRAYPMTracer(Tracer):
        def __init__(self, **kwargs):
            ct.initialize()

        def start(self, name):
            ct.start(name)

        def stop(self, name):
            ct.stop(name)

        def enable(self):
            ct.enable()

        def disable(self):
            ct.disable()

        def reset(self):
            ct.reset()
except:
    print(f"Error importing CRAYPMTracer")
    pass


try:
    import scorep.user as sp

    class SCOREPTracer(Tracer):
        def __init__(self, **kwargs):
            pass

        def start(self, name):
            sp.region_begin(name)

        def stop(self, name):
            sp.region_end(name)

        def enable(self):
            sp.enable_recording()

        def disable(self):
            sp.disable_recording()

        def reset(self):
            pass


except:
    pass

__tracer_list__ = dict()


def has(name):
    return name in __tracer_list__


def initialize(trlist=["GPTLTracer", "SCOREPTracer"], verbose=False, **kwargs):
    for trname in trlist:
        try:
            tr = globals()[trname](**kwargs)
            __tracer_list__[trname] = tr
        except Exception as e:
            if verbose:
                print("tracer loading error:", trname, e)
            pass


def start(name, cudasync=False, sync=False):
    if cudasync and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except:
            pass
    if sync:
        MPI.COMM_WORLD.Barrier()
    for tr in __tracer_list__.values():
        tr.start(name)


def stop(name, cudasync=False, sync=False):
    if cudasync and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except:
            pass
    if sync:
        MPI.COMM_WORLD.Barrier()
    for tr in __tracer_list__.values():
        tr.stop(name)


def enable():
    for tr in __tracer_list__.values():
        tr.enable()


def disable():
    for tr in __tracer_list__.values():
        tr.disable()


def reset():
    for tr in __tracer_list__.values():
        tr.reset()


def profile(x_or_func=None, *decorator_args, **decorator_kws):
    def _decorator(func):
        @wraps(func)
        def wrapper(*args, **kws):
            if "x_or_func" not in locals() or callable(x_or_func) or x_or_func is None:
                x = func.__name__
            else:
                x = x_or_func
            start(x)
            out = func(*args, **kws)
            stop(x)
            return out

        return wrapper

    return _decorator(x_or_func) if callable(x_or_func) else _decorator


@contextmanager
def timer(x):
    start(x)
    yield
    stop(x)
