import os, resource


def limit_logical_cpus(logical_cpus):
    """Set process CPU affinity to the first max_cpus CPUs"""
    os.sched_setaffinity(os.getpid(), logical_cpus)
    print('Using following CPUs: ', os.sched_getaffinity(os.getpid()))


def limit_memory_usage(max_memory_limit_gb):
    """Set process memory limit to max_memory_limit_gb GB"""
    max_memory_limit_bytes = max_memory_limit_gb * 1024 * 1024 * 1024  # GB to bytes
    resource.setrlimit(resource.RLIMIT_AS, (max_memory_limit_bytes, max_memory_limit_bytes))  # soft and hard limit
    print('Using following memory limit: ', resource.getrlimit(resource.RLIMIT_AS)[1] / 1024 / 1024 / 1024, 'GB')


def limit_gpu_memory_usage(gpu_list, max_memory_limit_gb):
    """Restrict TensorFlow to only allocate max_memory_limit_gb of memory on all given GPUs"""
    import tensorflow as tf

    # set GPUs
    CUDA_VISIBLE_DEVICES = ','.join([str(gpu) for gpu in gpu_list])
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    gpus = tf.config.list_physical_devices('GPU')

    # set mem limit for each GPU
    for device in gpus:
        tf.config.set_logical_device_configuration(
            device,
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * max_memory_limit_gb)])


def multiple_gpu_distribution(func):
    """Decorator to distribute a function across multiple GPUs"""
    import tensorflow as tf

    def wrapper(*args, **kwargs):
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        # no distributed training if only one GPU is available
        if len(logical_gpus) < 2:
            return func(*args, **kwargs)
        # distribute across all GPUs with mirrored strategy
        strategy = tf.distribute.MirroredStrategy(logical_gpus)
        try:
            with strategy.scope():
                return func(*args, **kwargs)
        except Exception as e:
            print('Error during distributed training: ', e)
            return func(*args, **kwargs)

    return wrapper
