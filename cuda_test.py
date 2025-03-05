import tensorflow as tf
#print(tf.test.is_built_with_cuda())
#print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
#print(tf.config.list_physical_devices('GPU'))

print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU Details:", tf.config.list_physical_devices('GPU'))
print(tf.sysconfig.get_build_info()["cuda_version"])
print(tf.sysconfig.get_build_info()["cudnn_version"])
print(tf.sysconfig.get_build_info())
