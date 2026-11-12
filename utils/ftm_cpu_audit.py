"""Explicit batch20 CPU provenance guard and stdlib process-memory measurement."""
import ctypes
import os
import platform
import sys

import torch


def validate(args,records):
    enabled=getattr(args,'ftm_batch20_cpu_audit',False)
    if not enabled:
        if args.batch_size==20:raise ValueError('Batch20 requires --ftm-batch20-cpu-audit')
        return False
    if args.batch_size!=20 or str(args.device)!='cpu' or args.split!='pilot':raise ValueError('CPU batch20 audit requires batch-size20, device cpu and split pilot')
    if getattr(args,'ftm_input_profile','canonical224-v2')!='original299-v1' or args.source!='rn50_v1' or args.ftm_layer_profile!='mapped299':raise ValueError('CPU batch20 audit requires original299-v1/rn50_v1/mapped299')
    if len(records)!=20 or len({r['id'] for r in records})!=20 or any(r['split']!='pilot' for r in records):raise ValueError('CPU batch20 audit requires exactly20 unique selected pilot IDs, no incomplete group')
    return True


def configure():
    """Only explicitly requested CPU20 runs call this before model loading."""
    torch.set_num_threads(4)
    if torch.get_num_interop_threads()!=4:torch.set_num_interop_threads(4)
    if torch.get_num_threads()!=4 or torch.get_num_interop_threads()!=4:raise ValueError('CPU audit requires actual intraop4/interop4; use a fresh Python process')
    return {'kind':'single CPU batch20 source-only provenance audit, not10 pairs or20 independent samples',
        'cpu_processor':platform.processor(),'machine':platform.machine(),'logical_cpu_count':os.cpu_count(),
        'platform':platform.platform(),'python_version':sys.version,'torch_version':torch.__version__,
        'intraop_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
        'mkl_available':torch.backends.mkl.is_available(),'onednn_available':torch.backends.mkldnn.is_available(),
        'onednn_enabled':torch.backends.mkldnn.enabled,'torch_build_configuration':torch.__config__.show(),
        'environment':{key:os.environ.get(key) for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','MKL_CBWR','OMP_DYNAMIC','MKL_DYNAMIC','KMP_AFFINITY','OMP_PROC_BIND','DNNL_VERBOSE','ONEDNN_VERBOSE')},
        'configuration_note':'No MKL_CBWR, oneDNN, kernel or environment flag changes; library versions retained when present in torch build configuration. No cross-device identity claim.'}


class ProcessMemoryCounters(ctypes.Structure):
    # Windows DWORD is always32 bits; SIZE_T follows the process pointer width.
    _fields_=[('cb',ctypes.c_uint32),('PageFaultCount',ctypes.c_uint32),
        *[(name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
            'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]]


def process_peak_memory(system=None,dll_loader=None):
    system=platform.system() if system is None else system
    if system!='Windows':return {'available':False,'reason':'Windows GetProcessMemoryInfo is unavailable on '+system}
    loader=ctypes.WinDLL if dll_loader is None else dll_loader
    kernel=loader('kernel32',use_last_error=True);psapi=loader('psapi',use_last_error=True)
    current=kernel.GetCurrentProcess;current.argtypes=[];current.restype=ctypes.c_void_p
    measure=psapi.GetProcessMemoryInfo
    measure.argtypes=[ctypes.c_void_p,ctypes.POINTER(ProcessMemoryCounters),ctypes.c_uint32]
    measure.restype=ctypes.c_int32
    counters=ProcessMemoryCounters();counters.cb=ctypes.sizeof(counters)
    if not measure(current(),ctypes.byref(counters),ctypes.sizeof(counters)):
        code=ctypes.get_last_error() if hasattr(ctypes,'get_last_error') else 0
        raise OSError(code,'Windows GetProcessMemoryInfo failed; no fabricated peak RAM')
    return {'available':True,'api':'Windows GetProcessMemoryInfo/PROCESS_MEMORY_COUNTERS',
        'PeakWorkingSet64':int(counters.PeakWorkingSetSize),'units':'bytes',
        'scope':'Actual process-lifetime peak working set, including model, audit and runtime; not sum of image or batch peaks'}
