# Installation Guide 

## GPU support

Ideally, ADLStream should be run in a two GPU computer. 
However, it is not compulsory and ADLStream can be also run in CPU.

ADLStream uses [Tensorflow](https://www.tensorflow.org/). 
If you are interested in running ADLStream in GPU, the [tensorflow>=2.1.0](https://www.tensorflow.org/install/gpu
) GPU specifications are querired.

If you don't want to use GPU go to [Installing ADLStream](#installing-adlstream).

#### Hardware requirements <a name="hardware"></a>

  * Computer with at least 2 NVIDIA® GPU card with CUDA® Compute Capability 3.5 or higher 
  
#### Software requirements <a name="software"></a>

The following NVIDIA® software must be installed on your system:

  * NVIDIA® GPU drivers —CUDA 10.0 requires 418.x or higher.
  * CUDA® Toolkit —TensorFlow supports CUDA 10.1 (TensorFlow >= 2.1.0)
  * CUPTI ships with the CUDA Toolkit.
  * cuDNN SDK (>= 7.6)
  * (Optional) TensorRT 6.0 to improve latency and throughput for inference on some models.

## Installing ADLStream

You can install ADLStream and its dependencies from PyPI with:

```bash
pip install ADLStream
```

We strongly recommend that you install ADLStream in a dedicated virtualenv, to avoid conflicting with your system packages.

To use ADLStream:

```python 
import ADLStream
```

Check [getting started](../getting_started) for an example of use.


