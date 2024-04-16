from time import sleep
import os, resource


def limit_logical_cpus(logical_cpus):
    """Set process CPU affinity to the first max_cpus CPUs"""
    os.sched_setaffinity(os.getpid(), logical_cpus)
    print('Using following CPUs: ', os.sched_getaffinity(os.getpid()))


def limit_memory_usage(max_memory_limit_gb):
    """Set process memory limit to max_memory_limit_gb GB"""
    max_memory_limit_bytes = max_memory_limit_gb * 1024 * 1024 * 1024  # GB to bytes
    resource.setrlimit(resource.RLIMIT_AS, (max_memory_limit_bytes, max_memory_limit_bytes)) # soft and hard limit
    print('Using following memory limit: ', resource.getrlimit(resource.RLIMIT_AS)[1] / 1024 / 1024 / 1024, 'GB')
