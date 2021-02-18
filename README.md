# ADLStream

[![PyPI](https://img.shields.io/pypi/v/ADLStream.svg)](https://pypi.org/project/ADLStream/) 
[![Documentation Status](https://readthedocs.org/projects/adlstream/badge/?version=latest)](https://adlstream.readthedocs.io/en/latest/?badge=latest)
![CI](https://github.com/pedrolarben/ADLStream/workflows/CI/badge.svg?branch=master)
[![Downloads](https://pepy.tech/badge/adlstream)](https://pepy.tech/project/adlstream)
[![Python 3.6](https://img.shields.io/badge/Python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

##### Asynchronous dual-pipeline deep learning framework for online data stream mining. 


ADLStream is a novel asynchronous  dual-pipeline deep  learning  framework  for  data  stream  mining. 
This system has two separated layers for training and testing that work simultaneously in order to provide quick predictions and perform frequent updates of the model. 
The dual-layer architecture  allows  to  alleviate  the  computational  cost problem  of  complex  deep  learning  models,  such  as convolutional neural networks, for the data streaming context,  in  which  speed  is  essential.

<p align="center">
  <img width="100%"  src="https://raw.githubusercontent.com/pedrolarben/ADLStream/master/docs/img/ADLStream.png">
</p>

Complete documentation and API of ADLStream can be found in [adlstream.readthedocs.io](https://adlstream.readthedocs.io).


- [ADLStream](#adlstream)
        - [Asynchronous dual-pipeline deep learning framework for online data stream mining.](#asynchronous-dual-pipeline-deep-learning-framework-for-online-data-stream-mining)
  - [Installation Guide](#installation-guide)
    - [GPU support](#gpu-support)
      - [Hardware requirements](#hardware-requirements)
      - [Software requirements](#software-requirements)
    - [Installing ADLStream](#installing-adlstream)
  - [Getting Started](#getting-started)
      - [1. Create the stream](#1-create-the-stream)
      - [2. Create the stream generator.](#2-create-the-stream-generator)
      - [3. Configure the evaluation process.](#3-configure-the-evaluation-process)
      - [4. Configure model and create ADLStream](#4-configure-model-and-create-adlstream)
      - [5. Run ADLStream & Results](#5-run-adlstream--results)
  - [Research papers related](#research-papers-related)
  - [Contributing](#contributing)
  - [License](#license)
  - [Authors](#authors)

## Installation Guide

### GPU support

Ideally, ADLStream should be run in a two GPU computer. 
However, it is not compulsory and ADLStream can be also run in CPU.

ADLStream uses [Tensorflow](https://www.tensorflow.org/). 
If you are interested in running ADLStream in GPU, the [tensorflow>=2.1.0](https://www.tensorflow.org/install/gpu
) GPU specifications are querired.

If you don't want to use GPU go to [Installing ADLStream](#installing-adlstream).

#### Hardware requirements

  * Computer with at least 2 NVIDIA® GPU card with CUDA® Compute Capability 3.5 or higher 
  
#### Software requirements

The following NVIDIA® software must be installed on your system:

  * NVIDIA® GPU drivers —CUDA 10.0 requires 418.x or higher.
  * CUDA® Toolkit —TensorFlow supports CUDA 10.1 (TensorFlow >= 2.1.0)
  * CUPTI ships with the CUDA Toolkit.
  * cuDNN SDK (>= 7.6)
  * (Optional) TensorRT 6.0 to improve latency and throughput for inference on some models.

### Installing ADLStream

You can install ADLStream and its dependencies from PyPI with:

```bash
pip install ADLStream
```

We strongly recommend that you install ADLStream in a dedicated virtualenv, to avoid conflicting with your system packages.

To use ADLStream:

```python 
import ADLStream
```

## Getting Started 

These instructions explain how to use ADLStream framework with a simple example.

In this example we will use a LSTM model for time series forecasting in streaming.

#### 1. Create the stream

Fist of all we will need to create the stream.
Stream objects can be created using the classes from `ADLStream.data.stream`. We can choose different options depending on the source of our stream (from a csv file, a Kafka cluster, etc). 

In this example, we will use the `FakeStream`, which implements a sine wave.

```python
import ADLStream

stream = ADLStream.data.stream.FakeStream(
    num_features=6, stream_length=1000, stream_period=100
)
```

More precisely, this stream will return a maximun of 1000 instances. The stream sends one message every 100 milliseconds (0.1 seconds).

#### 2. Create the stream generator.

Once we have our source stream, we need to create our stream generator. 
A `StreamGenerator` is an object that will preprocess the `stream` and convert the messages into input (`x`) and target (`y`) data of the deep learning model.
There are different options to choose under `ADLStream.data` and, if needed, we can create our custom `StreamGenerator` by inheriting `BaseStreamGenerator`.

As our problem is time series forecasting, we will use the `MovingWindowStreamGenerator`, which performs the moving-window preprocessing method.

```python
stream_generator = ADLStream.data.MovingWindowStreamGenerator(
    stream=stream, past_history=12, forecasting_horizon=3, shift=1
)
```

For the example we have set the past history to 12 and the model will predict the next 3 elements.

#### 3. Configure the evaluation process.

In order to evaluate the performance of the model, we need to create a validator object. 
There exist different alternative for data-stream validation, some of the most common one can be found under `ADLStream.evaluation`. 
Furthermore, custom evaluators can be easily implemented by inheriting `BaseEvaluator`.

In this case, we are going to create a `PrequentialEvaluator` which implements the idea that more recent examples are more important using a decaying factor.

```python
evaluator = ADLStream.evaluation.PrequentialEvaluator(
    chunk_size=10,
    metric="MAE",
    fadding_factor=0.98,
    results_file="ADLStream.csv",
    dataset_name="Fake Data",
    show_plot=True,
    plot_file="test.jpg",
)
```

As can be seen, we are using the mean absolute error (MAE) metrics. Other options can be found in `ADLStream.evaluation.metrics`.
The evaluator will save the progress of the error metric in `results_file` and will also plot the progress and saved the image in `plot_file`.

#### 4. Configure model and create ADLStream

Finally we will create our `ADLStream` object specifying the model to use.

The required model arguments are the architecture, the loss and the optimizer. In addition, we can provides a dict with the model parameters to customize its architecture. 
All the available model architecture and its parameters can be found in `ADLStream.models`.

For the example we are using a deep learning model with 3 stacked LSTM layers of 16, 32 and 64 units followed by a fully connected block of two layers with 16 and 8 neurons.

```python
model_architecture = "lstm"
model_loss = "mae"
model_optimizer = "adam"
model_parameters = {
    "recurrent_units": [16, 32, 64],
    "recurrent_dropout": 0,
    "return_sequences": False,
    "dense_layers": [16, 8],
    "dense_dropout": 0,
}

adls = ADLStream.ADLStream(
    stream_generator=stream_generator,
    evaluator=evaluator,
    batch_size=60,
    num_batches_fed=20,
    model_architecture=model_architecture,
    model_loss=model_loss,
    model_optimizer=model_optimizer,
    model_parameters=model_parameters,
    log_file="ADLStream.log",
)
```

#### 5. Run ADLStream & Results

Once we came the ADLStream object created, we can initiate it by calling its `run` function.

```python
adls.run()
```

The processes will start and the progress will be plot obtaining a result similar to this one

![output-plot](https://raw.githubusercontent.com/pedrolarben/ADLStream/master/docs/img/fakedata-example.gif)

Complete API reference can be found [here](https://adlstream.readthedocs.io).

## Research papers related

Here it is the original paper that you can cite to reference ADLStream

* [Lara-Benítez, Pedro, Manuel Carranza-garcía, et al. ‘Asynchronous Dual-pipeline Deep Learning Framework for Online Data Stream Classification’. Integrated Computer-Aided Engineering. 1 Jan. 2020 : 101 – 119.](https://doi.org/10.3233/ICA-200617)

Any other study using ADLStream framework will be listed here.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). We appreciate all kinds of help.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Authors 

* **Pedro Lara-Benítez** - [LinkedIn](www.linkedin.com/in/pedrolarben) 
* **Manuel Carranza-García** - [LinkedIn](https://www.linkedin.com/in/manuelcarranzagarcia96/)
* **Jorge García-Gutiérrez** 
* **José C. Riquelme**



