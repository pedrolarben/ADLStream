import os
import glob
import time
import random
import json
import traceback
from multiprocessing import Process, Lock
from multiprocessing.managers import BaseManager
from kafka import KafkaConsumer, KafkaProducer
import sklearn.metrics as metrics
from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd
import scipy.io.arff as arff
import tensorflow as tf
from datastream_minerva.models import create_model


# Object shared among processes
class Consumer:

    def __init__(self, batch_size=10, num_batches_fed=40, topic='', results_path='./', bootstrap_servers=None,
                 from_beginning=False, debug=True, two_gpu=False, create_model_func='cnn', clf_name=None, time_out_ms=10000):
        self.batch_size = batch_size
        self.num_batches_fed = num_batches_fed
        self.topic = topic
        self.data_set_name = self.topic.replace('streams_', '')
        self.results_path = os.path.join(results_path, self.data_set_name)
        self.bootstrap_servers = bootstrap_servers
        self.from_beginning = from_beginning
        self.debug = debug
        self.two_gpu = two_gpu
        self.create_model_func = create_model_func
        self.time_out_ms = time_out_ms
        self.new_model_available = False
        self.buffer = []
        self.time_out = False
        self.finished = False
        self.history = {self.data_set_name: {
            'data': pd.DataFrame(),
            'metrics': pd.DataFrame()
        }}
        self.train_time = 0.
        self.x_training = None
        self.y_training = None
        self.batch_counter = 0
        self.count = 0
        self.initial_training_set = []
        self.num_features = None
        self.num_classes = None
        self.weights = None
        self.available = True
        self.clf_name = clf_name if clf_name is not None else 'ADLStream_{0}_{1}x{2}'.format(str(self.create_model_func),str(num_batches_fed),str(batch_size))
        self.average = 'weighted'
        self.loss_func = 'categorical_crossentropy'
        self.columns = None
        self.classes = None
        self.prequential_kappa = None
        self.first_history = 0


    def add(self, ls):
        if not self.initial_training_set:
            self.initial_training_set = [] + ls
        else:
            self.buffer = self.buffer + ls

    def next(self):
        if len(self.buffer) == 0:
            return None
        res = self.buffer[0]
        self.buffer.pop(0)
        self.count = self.count + 1
        return res

    def get_timeout_ms(self):
        return self.time_out_ms

    def is_time_out(self):
        return self.time_out

    def set_time_out(self, b=True):
        self.time_out = b

    def get_count(self):
        return self.count

    def get_batch_size(self):
        return self.batch_size

    def get_bootstrap_servers(self):
        return self.bootstrap_servers

    def get_from_begining(self):
        return self.from_beginning

    def get_debug(self):
        return self.debug

    def is_two_gpu(self):
        return self.two_gpu

    def get_topic(self):
        return self.topic

    def get_data_set_name(self):
        return self.data_set_name

    def get_history(self):
        return self.history[self.data_set_name]

    def is_first_history(self):
        if self.first_history<2:
            self.first_history += 1
            return True
        else:
            return False

    def write_history_metrics(self, k, v):
        dir_name = os.path.dirname(__file__)
        file_path = os.path.join(dir_name, self.get_results_path(), self.get_clf_name())
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = os.path.join(file_path, k + '.csv')
        if self.is_first_history():
            with open(file_path, 'w') as fd:
                fd.write(','.join([str(c) for c in v.columns]) + '\n')
        if k == 'metrics' or k == 'data':
            values = v.values if len(v.values.shape) > 1 else v.values.reshape((1, v.values.shape[0]))
            values_csv = '\n'.join([','.join([str(c) for c in r]) for r in values]) + '\n'
            with open(file_path, 'a') as fd:
                fd.write(values_csv)
        else:
            raise Exception("history key must be 'data' or 'metrics'")

    def get_results_path(self):
        return self.results_path

    def get_clf_name(self):
        return self.clf_name

    def get_average(self):
        return self.average

    def set_average(self, av):
        self.average = av

    def get_loss_function(self):
        return self.loss_func

    def set_loss_function(self, lf):
        self.loss_func = lf

    def get_columns(self):
        return self.columns

    def set_columns(self, c):
        self.columns = c

    def get_classes(self):
        return self.classes

    def set_classes(self, c):
        self.classes = c

    def get_buffer_len(self):
        return len(self.buffer)

    def get_initial_training_set(self):
        return self.initial_training_set

    def set_num_features(self, nf):
        self.num_features = nf

    def get_num_features(self):
        return self.num_features

    def get_num_classes(self):
        return len(self.get_classes())

    def set_weights(self, w):
        self.weights = w

    def get_weights(self):
        return self.weights

    def set_new_model_available(self, b=True):
        self.new_model_available = b

    def is_new_model_available(self):
        return self.new_model_available

    def is_finished(self):
        return self.finished

    def set_finished(self):
        self.finished = True

    def is_available(self):
        return self.available

    def set_available(self, b):
        self.available = b

    def add_train_time(self, new_train_time):
        self.train_time += new_train_time

    def get_train_time(self):
        res = self.train_time
        self.train_time = 0
        return res

    def add_training_data(self, x, y):
        if self.x_training is None or self.y_training is None:
            self.x_training = x
            self.y_training = y
        else:
            self.x_training = np.vstack((self.x_training, x))
            self.y_training = np.vstack((self.y_training, y))
            self.batch_counter += 1

    def get_training_data(self):
        x, y = np.copy(self.x_training), np.copy(self.y_training)

        num_instances_to_train = self.batch_size * self.num_batches_fed
        if len(x) > num_instances_to_train:
            self.x_training = self.x_training[-num_instances_to_train:]
            self.y_training = self.y_training[-num_instances_to_train:]
            indices = list(range(len(x)))
            k, m = divmod(len(indices), 4)
            w = [1/8]*(k+1 if m > 0 else k) + [1/4]*(k+1 if m > 1 else k) + [1/2]*(k+1 if m > 2 else k) + [10]*k
            w = [e/sum(w) for e in w]
            indices = sorted(np.random.choice(indices, num_instances_to_train, False, w))
            x = np.take(x, indices, axis=0)
            y = np.take(y, indices, axis=0)

        return x, y

    def create_model(self, num_features, num_classes, loss_func):
        return create_model(self.create_model_func, num_features, num_classes, loss_func)
        # try:
        #     model = self.create_model_func[0](num_features, num_classes, loss_func)
        #     return model
        # except Exception:
        #     raise ValueError('Create-model function is not well defined')

    def get_create_model_func(self):
        return self.create_model_func

    def update_prequential_kappa(self, new_kappa):
        if self.prequential_kappa is None:
            self.prequential_kappa = new_kappa
        else:
            fading_factor = 0.98
            self.prequential_kappa = (new_kappa + fading_factor * self.prequential_kappa) / (1 + fading_factor)
        return self.prequential_kappa



# Get next message from Kafka
def read_message(consumer):
    message = consumer.next()
    return message


# Save Metrics
def save_history(consumer, x, y, y_pred, test_time, train_time):
    # print(y_pred)
    # print(y)
    y_pred = np.argmax(y_pred, axis=1)
    y = np.argmax(y, axis=1)
    # print(y_pred)
    # print(y)

    x = np.array(x)
    if len(x.shape) == 3:
        x = x.reshape(x.shape[0], -1)

    records = pd.DataFrame(np.hstack((x, y.reshape(-1, 1), y_pred.reshape(-1, 1))))
    records.columns = consumer.get_columns()
    consumer.write_history_metrics('data', records)

    prec, recall, fbeta, support = metrics.precision_recall_fscore_support(y, y_pred, average=consumer.get_average())
    accuracy = metrics.accuracy_score(y, y_pred)
    f1_score = metrics.f1_score(y, y_pred, average='weighted')
    conf_matrix = metrics.confusion_matrix(y, y_pred)

    try:
        tn, fp, fn, tp = conf_matrix.ravel()
    except ValueError as ve:
        tn, fp, fn, tp = conf_matrix[0][0], 0, 0, 0

    metrics_record = pd.DataFrame({
        'total': [len(consumer.get_history()['data'])],
        'tn': [tn],
        'fp': [fp],
        'fn': [fn],
        'tp': [tp],
        'precision': [prec],
        'recall': [recall],
        'f1': [f1_score],
        'fbeta': [fbeta],
        'accuracy': [accuracy],
        'train_time': [train_time],
        'test_time': [test_time]
    })
    consumer.write_history_metrics('metrics', metrics_record)


# Create results files
def write_results_file(consumer):
    dir_name = os.path.dirname(__file__)
    file_path = os.path.join(dir_name, consumer.get_results_path(), 'ADLStream_' + consumer.get_clf_name())
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    # history_data = consumer.get_history()['data']
    # print(history_data)
    # history_data.columns = consumer.get_columns()
    # dir_name = os.path.dirname(__file__)



def train_model(x_train, y_train, consumer, model, index=-1, device=None):
    # Train
    # if consumer.get_debug():
    print('P' + str(index) + ' (' + device + '):  Training with the last {} last instances'.format(len(x_train)))
    train_time_start = time.time()
    model.fit(x_train, y_train, consumer.get_batch_size(), epochs=1, verbose=0)
    train_time_end = time.time()
    train_time = train_time_end - train_time_start

    # Save model weights
    consumer.set_weights(model.get_weights())

    # Notify new model available
    consumer.set_new_model_available()

    return train_time


def classify(model, input_data):
    test_time_start = time.time()
    y_pred = model.predict(input_data, verbose=0)
    test_time_end = time.time()
    test_time = test_time_end - test_time_start

    return y_pred, test_time


# DNN training process
def dnn_train(index, consumer, lock_training_data):
    # # Specify GPU to use
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) >= 2:
        device = gpus[0]
        tf.config.experimental.set_memory_growth(device, True)
        tf.config.experimental.set_visible_devices(device, 'GPU')
        device_name = device[0]
    else:
        device_name = '/GPU:0'
        print("ERROR - TRAINING PROCESS: ADLStream needs at least 2 GPUs.")
        print("ERROR - TRAINING PROCESS: {0} GPUs found: {1}".format(len(gpus), gpus))

    # Wait until first window available.
    while not consumer.get_initial_training_set():
        if consumer.is_time_out():
            return
        pass

    # Create and train model for the first time.
    initial_training_set = consumer.get_initial_training_set()
    window_y, window_x = [], []

    for record in initial_training_set:
        consumer.set_classes(record.pop('classes', None))
        window_y.append(record.pop('class', None))
        window_x.append(list(record.values()))

        if not consumer.get_columns():
            aux_columns = list(record.keys())
            aux_columns.extend(['class', 'prediction'])
            consumer.set_columns(aux_columns)

    if consumer.get_num_classes() == 2:
        consumer.set_loss_function('binary_crossentropy')
        consumer.set_average('binary')

    x = np.expand_dims(np.array(window_x), axis=-1)
    y = label_binarize(window_y, consumer.get_classes())
    if consumer.get_num_classes()==2:
        y = np.eye(2)[y.flatten()]
    with lock_training_data:
        consumer.add_training_data(x, y)
    consumer.set_num_features(x.shape[1])

    # create model, train it and save the weights.

    # model = consumer.create_model(consumer.get_num_features(), consumer.get_num_classes(), consumer.get_loss_function())
    model = create_model(consumer.get_create_model_func(), consumer.get_num_features(), consumer.get_num_classes(), consumer.get_loss_function())

    with lock_training_data:
        x_train, y_train = consumer.get_training_data()
    train_model(x_train, y_train, consumer, model, index, device_name)

    # Main loop
    while not consumer.is_finished():
        with lock_training_data:
            x_train, y_train = consumer.get_training_data()
        train_time = train_model(x_train, y_train, consumer, model, index, device_name)
        consumer.add_train_time(train_time)


# DNN classifying process
def dnn_classify(index, consumer, lock_messages, lock_training_data):
    #
    # Specify GPU to use
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) >= 2:
        device = gpus[1]
        tf.config.experimental.set_memory_growth(device, True)
        tf.config.experimental.set_visible_devices(device, 'GPU')
        device_name = device[0]
    else:
        device_name = "/CPU:0"
        print("ERROR - CLASSIFYING PROCESS: ADLStream needs at least 2 GPUs.")
        print("ERROR - CLASSIFYING PROCESS: {0} GPUs found: {1}".format(len(gpus), gpus))
        # consumer.set_finished()


    window_y, window_x = [], []
    batch_size = consumer.get_batch_size()
    count_messages = 0
    y_pred_history = None
    x_pred_history = None
    y_history = None
    test_time = 0.

    # Wait until model is created and trained for the first time.
    while not consumer.is_new_model_available():
        if consumer.is_time_out():
            return
        pass

    # create model and load weights
    # model = consumer.create_model(consumer.get_num_features(), consumer.get_num_classes(),
    #                               consumer.get_loss_function())
    model = create_model(consumer.get_create_model_func(), consumer.get_num_features(), consumer.get_num_classes(),
                         consumer.get_loss_function())

    # model = create_cnn_model(consumer.get_num_features(), consumer.get_num_classes(), consumer.get_loss_function())

    model.set_weights(consumer.get_weights())

    # Main loop
    while True:
        # Read messages
        with lock_messages:
            record = read_message(consumer)
        # if no message received...
        if record is None:
            # if timeout = true --> break
            if consumer.is_time_out():
                consumer.set_finished()
                if consumer.get_debug():
                    print("P" + str(index) + ": Finish")
                if y_pred_history is not None:
                    save_history(consumer, x_pred_history, y_history, y_pred_history, test_time, 0.)
                # TODO: WHY????? write_results_file(consumer)
                break
            # ToDo: what if time_out is not True (wait? continue asking?)
            if not consumer.is_time_out():
                if consumer.get_debug():
                    print("P" + str(index) + ": ... waiting feeder")
                time.sleep(0.2)

        else:
            _ = record.pop('classes', None)
            record_class = record.pop('class', None)
            record_values = list(record.values())

            window_y.append(record_class)
            window_x.append(record_values)

            y = label_binarize([window_y[-1]], consumer.get_classes())
            if consumer.get_num_classes() == 2:
                y = np.eye(2)[y.flatten()]

            x_pred = np.expand_dims(np.array([list(record_values)]), axis=-1)
            # Update model if new model available
            if consumer.is_new_model_available():
                model.set_weights(consumer.get_weights())
                consumer.set_new_model_available(False)
            # Classify
            y_pred, test_time_aux = classify(model, x_pred)
            test_time += test_time_aux

            if y_pred_history is None:
                y_pred_history = np.array(y_pred)
            else:
                y_pred_history = np.append(y_pred_history, y_pred, axis=0)

            if x_pred_history is None:
                x_pred_history = np.array(x_pred)
            else:
                x_pred_history = np.append(x_pred_history, x_pred, axis=0)

            if y_history is None:
                y_history = np.array(y)
            else:
                y_history = np.append(y_history, y, axis=0)

            # if window completed
            if len(window_x) % batch_size == 0:
                count_messages += batch_size
                if consumer.get_debug():
                    print('P' + str(index) + ' (' + device_name + '):  {} instances classified'.format(
                        count_messages))
                # Format window data
                x = np.expand_dims(np.array(window_x), axis=-1)
                window_y = list(np.array(window_y))
                y = label_binarize(window_y, consumer.get_classes())
                if consumer.get_num_classes() == 2:
                    y = np.eye(2)[y.flatten()]

                with lock_training_data:
                    consumer.add_training_data(x, y)
                window_y, window_x = [], []

                # Save metrics
                save_history(consumer, x_pred_history, y_history, y_pred_history, test_time, consumer.get_train_time())
                y_pred_history = None
                x_pred_history = None
                y_history = None
                test_time = 0.


# Buffer feeder
def buffer_feeder(consumer, lock):
    batch_size = consumer.get_batch_size()
    kafka_consumer = KafkaConsumer(consumer.get_topic(),
                                   group_id='DSClassification_Consumer_par_{}'.format(random.randrange(999999)),
                                   bootstrap_servers=consumer.get_bootstrap_servers(),
                                   auto_offset_reset='earliest' if consumer.get_from_begining() else 'latest',
                                   consumer_timeout_ms=consumer.get_timeout_ms(),
                                   value_deserializer=lambda m: json.loads(m.decode('ascii')))
    new_batch = []
    i = 0
    while True:
        try:
            message = next(kafka_consumer).value
            i += 1
            new_batch.append(message)
        except StopIteration:
            if consumer.get_debug():
                print("BF: finish" + " - " + str(i) + " messages")
            if new_batch:
                with lock:
                    consumer.add(new_batch)
            consumer.set_time_out()
            break
        except Exception:
            print('Exception getting new messages from kafka')
            print(traceback.format_exc())

        if len(new_batch) == batch_size:
            with lock:
                consumer.add(new_batch)
                len_buffer = consumer.get_buffer_len()
            new_batch = []
            # Control buffer size
            if len_buffer > 100 * batch_size:
                time.sleep(len_buffer / (100 * batch_size))


def runADLStream(topic, create_model_func='cnn', two_gpu=True, batch_size=10, num_batches_fed=40, debug=True, output_path='./ADLStreamResults/', from_beginning=True, time_out_ms=10000, bootstrap_servers='localhost:9092', clf_name=None):
    bootstrap_servers = bootstrap_servers.split(' ')
    print("Consuming from bootstrap_servers: " + str(bootstrap_servers))
    print('Topic:', topic)
    print('Two GPU:', two_gpu)
    print('Batch size:', batch_size)
    print('Number of batches fed:', num_batches_fed)
    print('Output path:', output_path)
    print('time out (ms)', time_out_ms)

    BaseManager.register('Consumer', Consumer)
    manager = BaseManager()
    manager.start()
    consumer = manager.Consumer(batch_size=batch_size, num_batches_fed=num_batches_fed, results_path=output_path,
                                topic=topic, clf_name=clf_name,
                                bootstrap_servers=bootstrap_servers, from_beginning=from_beginning, debug=debug,
                                two_gpu=two_gpu, create_model_func=create_model_func, time_out_ms=time_out_ms)

    lock_messages = Lock()
    lock_training_data = Lock()

    pb = Process(target=buffer_feeder, args=[consumer, lock_messages])
    p0 = Process(target=dnn_train, args=[0, consumer, lock_training_data])
    p1 = Process(target=dnn_classify, args=[1, consumer, lock_messages, lock_training_data])
    pb.start()
    p0.start()
    p1.start()
    pb.join()
    p0.join()
    p1.join()


def runMultiARFFProducer(dir_path, bootstrap_servers='localhost:9092'):
    bootstrap_servers = bootstrap_servers.split(' ')

    topics = []

    arff_files = glob.glob(os.path.join( dir_path, '**/*.arff'), recursive=True)
    for file_path in arff_files:
        producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                                 client_id='ARFF_Producer_{}'.format(random.randrange(999999)),
                                 value_serializer=lambda m: json.dumps(m).encode('ascii'))
        topic = 'streams_' + os.path.basename(file_path).replace('.arff', '')
        topics.append(topic)
        print('Creating topic for {}'.format(topic))
        data, meta = arff.loadarff(file_path)
        attributes = [x for x in meta]
        df = pd.DataFrame(data)
        class_name = 'class' if 'class' in df.columns else 'target'
        classes = np.unique(df[class_name]).astype(np.int).tolist()
        min_classes = min(classes)

        for _, entry in enumerate(data):
            record = {'classes': classes}
            for _, attr in enumerate(attributes):
                value = entry[attr]
                if type(value) == np.bytes_:
                    value = int(value.decode('UTF-8'))
                if attr == class_name:
                    attr = 'class'
                record[attr] = value
            producer.send(topic, record)
        producer.close()

    return topics


def decode_class(c):
    is_dataset = type(c) == type(pd.Series())
    if is_dataset:
        decoded = c.str.decode("utf-8") if type(c[0]) == type(b'') else c.astype('int').astype('str')
    else:
        decoded = c.decode("utf-8") if isinstance(c, (bytes, np.bytes_)) else str(int(c))

    if is_dataset:
        if 'class' in decoded[0]:
            decoded = decoded.str.split('class', expand=True)[1]
        elif 'level' in decoded[0]:
            decoded = decoded.str.split('class', expand=True)[1]
        elif 'group' in decoded[0]:
            decoded = decoded.str.split('group', expand=True)[1]
            decoded[decoded == 'A'] = 0
            decoded[decoded == 'B'] = 1
    else:
        if 'class' in decoded:
            decoded = decoded.split('class')[1]
        elif 'group' in decoded:
            decoded = decoded.split('group')[1]
            decoded = 0 if decoded == 'A' else decoded
            decoded = 1 if decoded == 'B' else decoded

    return decoded



def runARFFProducer(file_path, bootstrap_servers='localhost:9092'):
    bootstrap_servers = bootstrap_servers.split(' ')

    producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                             client_id='ARFF_Producer_{}'.format(random.randrange(999999)),
                             value_serializer=lambda m: json.dumps(m).encode('ascii'))
    topic = 'streams_' + os.path.basename(file_path).replace('.arff', '')
    print('Creating topic for {}'.format(topic))
    data, meta = arff.loadarff(file_path)
    attributes = [x for x in meta]
    df = pd.DataFrame(data)
    class_name = 'class' if 'class' in df.columns else 'target'
    classes = np.unique(decode_class(df[class_name])).astype(np.int)
    min_classes = min(classes)
    classes = classes - min_classes
    classes = [int(c) for c in classes]

    for _, entry in enumerate(data):
        record = {'classes': classes}
        for _, attr in enumerate(attributes):
            value = entry[attr]
            if attr == class_name:
                attr = 'class'
                value = int(decode_class(value)) - int(min_classes)

            elif type(value) == np.bytes_:
                value = value.decode('UTF-8')

                if meta[attr][0] == 'nominal':
                    # print(meta[attr], value)
                    value = meta[attr][1].index(value)

            record[attr] = value

        producer.send(topic, record)
    producer.close()

    return topic

