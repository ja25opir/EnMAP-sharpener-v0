from time import sleep
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


def limit_tf_gpu_usage(gpu_list, max_memory_limit_gb):
    """Restrict TensorFlow to only allocate max_memory_limit_gb of memory on the first GPU"""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    print('Available GPUs:', gpus)

    tf.config.set_logical_device_configuration(
        gpus[1],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * max_memory_limit_gb)])

    # for gpu in gpu_list:
    #     try:
    #         tf.config.set_logical_device_configuration(
    #             gpus[gpu],
    #             [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * max_memory_limit_gb)])
    #         print('Using GPU', gpu, 'with memory limit of', max_memory_limit_gb, 'GB')
    #     except RuntimeError as e:
    #         # Virtual devices must be set before GPUs have been initialized
    #         print(e)


def flexible_tf_gpu_memory_growth(gpu_list):
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpu_list:
        try:
            # memory growth needs to be the same across GPUs
            tf.config.experimental.set_memory_growth(gpus[gpu], True)
            print('Using GPU', gpu)
        except RuntimeError as e:
            # memory growth must be set before GPUs have been initialized
            print(e)
