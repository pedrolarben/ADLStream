# ADLStream
###### Asynchronous dual-pipeline deep learning framework for online data stream classification

1. [API Docs](#api)
  * [runADLStream](#runADLStream)
  * [runARFFProducer](#runARFFProducer)
  * [runMultiARFFProducer](#runMultiARFFProducer)
2. [Getting Started](#gs)
  * [Prerequisites](#prerequisites)
    * [Hardware requirements](#hardware)
    * [Software requirements](#software)
  * [Installing](#installing)
    * [Sample project](#sample)
3. [Running the tests](#test)
4. [Authors](#authors)
5. [Licence](#licence)
6. [Research papers related](#research)

ADLStream is a novel asynchronous  dual-pipeline deep  learning  framework  for  data  stream  classification. 
This system has two separated layers for training and testing that work simultaneously in order to provide quick predictions and perform frequent updates of the model. 
The dual-layer architecture  allows  to  alleviate  the  computational  cost problem  of  complex  deep  learning  models,  such  as convolutional neural networks, for the data streaming context,  in  which  speed  is  essential.

<p align="center">
  <img width="451" height="368" src="https://github.com/pedrolarben/datastream-minerva/blob/master/doc/images/Parallel-NN.png">
</p>

## API Docs <a name="api"></a>
This project has a main function which implement the algorithm and two auxiliary functions which can be helpful for testing the software.
Here it is a brief description of these three functions

### runADLStream <a name="runADLStream"></a>
This is the main function. It run the ADLStream algorithm. For more details about how ADLStream works, read [this article](#research).
```python
runADLStream(topic, create_model_func, two_gpu, batch_size, num_batches_fed, debug, output_path, from_beginning, time_out_ms, bootstrap_servers)
```
##### Args:
* **`topic`**: A string representing the name of the topic to consume from kafka.
* **`create_model_func`**: Function with three parametres that return a keras model already compiled.
   * Args:
      * `num_features`: Number of input units.
      * `num_classes`: Number of possible classes of the classification problem.
      * `loss_func`: Loss function needed to compile the model.
   * Returns:
      * `model`: Keras deep model
   * Example:
      ```python
        def create_cnn_model(num_features, num_classes, loss_func):
            from keras.layers import Dense, Conv1D, Flatten, MaxPool1D, Dropout
            from keras.models import Input, Model

            inp = Input(shape=(num_features, 1), name='input')
            c = Conv1D(32, 7, padding='same', activation='relu', dilation_rate=3)(inp)
            c = MaxPool1D(pool_size=2)(c)
            c = Conv1D(64, 5, padding='same', activation='relu', dilation_rate=3)(c)
            c = MaxPool1D(pool_size=2)(c)
            c = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=3)(c)
            c = MaxPool1D(pool_size=2)(c)
            c = Flatten()(c)
            c = Dense(512, activation='relu')(c)
            c = Dropout(0.2)(c)
            c = Dense(128, activation='relu')(c)
            c = Dropout(0.2)(c)
            c = Dense(num_classes, activation="softmax", name="prediction")(c)
            model = Model(inp, c)

            model.compile(loss=loss_func, optimizer='adam', metrics=['accuracy'])
            return model
      ```
* **`two_gpu`**: (Optional.) Whether the computer has two GPU available or not. It is `True` by default as it will not have any improvement on a single-gpu system.
* **`batch_size`**: (Optional.) Size of the training batch (`20` by default).
* **`num_batches_fed`**: (Optional.) Number of batches fed for every training iteraction (`40` by default). The model will be trained with a total of `num_batches_fed * batch_size` instances.
* **`debug`**: (Optional.) Whether to show debug messages or not (`True` by default).
* **`output_path`**: (Optional.) Path to folder where to save predictions and statistics. By default it is `'./ADLStreamResults/'`.
* **`from_beginning `**: (Optional.) Kafka consumer's parameter which represents whether to start consuming the stream from the beginning or not. By default it is `False`.
* **`time_out_ms`**: (Optional.) Kafka consumer's parameter which represents the number of milliseconds to block during message iteration before raising StopIteration.
* **`bootstrap_servers`**: (Optional.) String with the server address. In case there are more than one address, the different addresses are separated by a space. By default it is `'localhost:9092'`.

##### Returns:
* **`None`**. It doesn't return nothing. It finish after `time_out_ms` miliseconds without reaciving any message from the stream source. The prediction results and different statistics can be found in `output_path`.

### runARFFProducer <a name="runARFFProducer"></a>
It sends a ARFF file data to a kafka server in order to be consumed lately by ADLStream algorithm. This function is created for testing and experimental purposes.
```python
runARFFProducer(file_path, bootstrap_servers)
```
##### Args:
* **`file_path`**: String representing the path to the ARFF file. 
* **`bootstrap_servers`**: (Optional.) String with the server address. In case there are more than one address, the different addresses are separated by a space. By default it is `'localhost:9092'`.

##### Returns:
* **`topic`**: String representing the topic name created in the kafka server.


### runMultiARFFProducer <a name="runMultiARFFProducer"></a>
It sends all the ARFF files data from an specific folder to a kafka server in order to be consumed lately by ADLStream algorithm. This function is created for testing and experimental purposes.
```python
runMultiARFFProducer(dir_path, bootstrap_servers)
```
##### Args:
* **`dir_path`**: String representing the path to the folder containing the ARFF files. 
* **`bootstrap_servers`**: (Optional.) String with the server address. In case there are more than one address, the different addresses are separated by a space. By default it is `'localhost:9092'`.

##### Returns:
* **`topics`**: List of strings representing the topic names created in the kafka server (one for each ARFF file).

## Getting Started <a name="gs"></a>

These instructions will get you a copy of the project up and running on your local machine for development, testing and deployment purposes. 

### Prerequisites <a name="prerequisites"></a>

In order to run this project, it is needed the following prerequisites

#### Hardware requirements <a name="hardware"></a>

  * Computer with 2 NVIDIA® GPU card with CUDA® Compute Capability 3.5 or higher 
  
#### Software requirements <a name="software"></a>

The following NVIDIA® software must be installed on your system:

  * NVIDIA® GPU drivers —CUDA 10.0 requires 410.x or higher.
  * CUDA® Toolkit —TensorFlow supports CUDA 10.0 (TensorFlow >= 1.13.0)
  * CUPTI ships with the CUDA Toolkit.
  * cuDNN SDK (>= 7.4.1)
  * (Optional) TensorRT 5.0 to improve latency and throughput for inference on some models.




### Installing <a name="installing"></a>
First install the system dependencies
##### Ubuntu
```bash
sudo apt update
sudo apt install python3-dev python3-pip
sudo pip3 install -U virtualenv  # system-wide install
```
##### Centos
```bash
sudo yum install python3-dev python3-pip
sudo yum install -U virtualenv
```

Then we create a virtual environment where to install the libraries needed.
This project works with tensorflow < 2.0
```bash
virtualenv --system-site-packages -p python3 ./tensorflow1
source ./tensorflow1/bin/activate
pip install numpy pandas sklearn tensorflow==1.15rc2 keras kafka-python
```

in order to use ADLStream, we'll download the source code
```bash
cd ~/ # Directory of your choise
git clone https://github.com/pedrolarben/datastream-minerva.git
cd datastream-minerva
```

you can add your code to this folder but we strongly recommend you to follow the next instructions to use ADLStream in your own project.

#### Sample project <a name="sample"></a>

Copy the folder `datastream_minerva` to your project (in the instrucción well create a project from scratch)

```bash
mkdir ~/my_ADLStream_project/
cp -r datastream_minerva/ ~/my_ADLStream_project/datastream_minerva/
cd ~/my_ADLStream_project/
```
Once we have the ADLStream code in our project we can use it as a library in our script

```bash
> main.py # create a python file 
```
here there is a small piece of code to test ADLStream

```python3
from datastream_minerva.ADLStream import runADLStream, runARFFProducer

bootstrap_servers = 'localhost:9092' # Kafka server
arff_file = '/path/to/your/data/file/dataset_name.arff' 

# Create topic and send dataset instances as messages to kafka
topic = runARFFProducer(arff_file, bootstrap_servers)

# Consume topic created previously and run ADLStream with the default parameters
runADLStream(topic, output_path='~/ADLStream_example_output/', bootstrap_servers=bootstrap_servers)
```


## Running the tests <a name="test"></a>

A small script to test the ARFF producers and the algorithm  have been inplemented in `test.py`. 
In order to run the tests, execute the following instrucction. Be aware it may take significant time.
```bash
python ./test.py
```

## Authors <a name="authors"></a>

* **Pedro Lara-Benítez** - [LinkedIn](www.linkedin.com/in/pedrolarben)
* **Manuel Carranza-García** - [LinkedIn](https://www.linkedin.com/in/manuelcarranzagarcia96/)
* **Jorge García-Gutiérrez** 
* **José C. Riquelme**

## License<a name="licence"></a>

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Research papers related <a name="research"></a>

Here it is the original paper that you can cite to reference ADLStream
* [Pedro Lara-Benitez, Manuel Carranza-garcía, et al. Asynchronous dual-pipeline deep learning
framework for online data stream
classification. Integrated Computer-Aided Engineering. 2019. Under revision.]()


