import os
import argparse
import time
import random
import json
import traceback
from multiprocessing import Process, Lock
from multiprocessing.managers import BaseManager
from kafka import KafkaConsumer
import sklearn.metrics as metrics
import numpy as np
import pandas as pd


# Object shared among processes
class Consumer:

    def __init__(self, batch_size=10, num_batches_fed=40, topic='', results_path='./', bootstrap_servers=None,
                 from_beginning=False, debug=True):
        self.batch_size = batch_size
        self.num_batches_fed = num_batches_fed
        self.topic = topic
        self.data_set_name = self.topic.replace('streams_', '')
        self.results_path = os.path.join(results_path, self.data_set_name)
        self.bootstrap_servers = bootstrap_servers
        self.from_beginning = from_beginning
        self.debug = debug
        self.buffer = []
        self.time_out = False
        self.history = {self.data_set_name: {
            'data': pd.DataFrame(),
            'metrics': pd.DataFrame()
        }}
        self.x_training = None
        self.y_training = None
        self.batch_counter = 0
        self.count = 0
        self.initial_training_set = []
        self.num_features = None
        self.num_classes = None
        self.weights = None
        self.available = True
        self.clf_name = '3_Dilated_Conv'
        self.average = 'weighted'
        self.loss_func = 'categorical_crossentropy'
        self.columns = None
        self.classes = None

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

    # def get_num_batches_fed(self):
    #     return self.num_batches_fed

    def get_topic(self):
        return self.topic

    def get_data_set_name(self):
        return self.data_set_name

    def get_history(self):
        return self.history[self.data_set_name]

    def append_history(self, k, v):
        if k == 'metrics':
            self.history[self.data_set_name]['metrics'] = self.history[self.data_set_name]['metrics'].append(v,
                                                                                                             ignore_index=True)
        elif k == 'data':
            self.history[self.data_set_name][k] = pd.concat((self.history[self.data_set_name][k], v), ignore_index=True)
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

    def is_available(self):
        return self.available

    def set_available(self, b):
        self.available = b

    def add_training_data(self, x, y):
        # ToDo: control training data size (streams are potentially unlimited, memory is finite)
        if self.x_training is None or self.y_training is None:
            self.x_training = x
            self.y_training = y
        else:
            self.x_training = np.vstack((self.x_training, x))
            self.y_training = np.vstack((self.y_training, y))
            self.batch_counter += 1
            if self.batch_counter >= self.num_batches_fed-1:
                self.x_training = self.x_training[self.batch_size:]
                self.y_training = self.y_training[self.batch_size:]
                self.batch_counter = self.num_batches_fed-1

    def get_training_data(self):
        return np.copy(self.x_training), np.copy(self.y_training)


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
    consumer.append_history('data', records)

    prec, recall, fbeta, support = metrics.precision_recall_fscore_support(y, y_pred, average=consumer.get_average())
    accuracy = metrics.accuracy_score(y, y_pred)
    f1_score = metrics.f1_score(y, y_pred, average='weighted')
    conf_matrix = metrics.confusion_matrix(y, y_pred)

    try:
        tn, fp, fn, tp = conf_matrix.ravel()
    except ValueError as ve:
        tn, fp, fn, tp = conf_matrix[0][0], 0, 0, 0

    metrics_record = pd.Series({
        'total': len(consumer.get_history()['data']),
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'precision': prec,
        'recall': recall,
        'f1': f1_score,
        'fbeta': fbeta,
        'accuracy': accuracy,
        'train_time': train_time,
        'test_time': test_time
    })
    # history[clf_name]['metrics'] = history[clf_name]['metrics'].append(metrics_record, ignore_index=True)
    consumer.append_history('metrics', metrics_record)


# Create results files
def write_results_file(consumer):
    dir_name = os.path.dirname(__file__)
    file_path = os.path.join(dir_name, consumer.get_results_path(), 'keras_parallel_' + consumer.get_clf_name())
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    history_data = consumer.get_history()['data']
    history_data.columns = consumer.get_columns()
    history_data.to_csv(os.path.join(file_path, 'data.csv'), index=False)
    consumer.get_history()['metrics'].to_csv(os.path.join(file_path, 'metrics.csv'),
                                             columns=('total', 'tp', 'tn', 'fp', 'fn', 'precision',
                                                      'recall', 'f1', 'fbeta',
                                                      'accuracy', 'train_time', 'test_time'),
                                             index=False)


def create_cnn_model(num_features, num_classes, loss_func):
    from keras.layers import Dense, Conv1D, Flatten, Dropout
    from keras.models import Input, Model

    inp = Input(shape=(num_features, 1), name='input')
    c = Conv1D(32, 7, padding='same', activation='relu', dilation_rate=3)(inp)
    c = Conv1D(64, 5, padding='same', activation='relu', dilation_rate=3)(c)
    c = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=3)(c)
    c = Flatten()(c)
    c = Dense(512, activation='relu')(c)
    c = Dropout(0.2)(c)
    c = Dense(128, activation='relu')(c)
    c = Dropout(0.2)(c)
    c = Dense(num_classes, activation="softmax", name="prediction")(c)
    model = Model(inp, c)

    model.compile(loss=loss_func, optimizer='adam', metrics=['accuracy'])
    return model


def train_model(consumer, model, index=-1):
    # Add new nata to the training data
    x_train, y_train = consumer.get_training_data()

    # Load weights
    weights = consumer.get_weights()
    if weights:
        model.set_weights(weights)

    # Train
    train_time_start = time.time()
    if consumer.get_debug():
        print('P' + str(index) + ': Training with the last {} last instances'.format(len(x_train)))
    model.fit(x_train, y_train, consumer.get_batch_size(), epochs=1, verbose=0)
    train_time_end = time.time()
    train_time = train_time_end - train_time_start

    # Save model weights
    consumer.set_weights(model.get_weights())

    return train_time


def classify(model, in_put):
    test_time_start = time.time()
    y_pred = model.predict(in_put, verbose=0)
    test_time_end = time.time()
    test_time = test_time_end - test_time_start

    return y_pred, test_time


# DNN 0
def DNN(index, consumer, lock_messages, lock_train):
    import tensorflow as tf
    from keras import backend as K
    from keras.utils import to_categorical

    session_conf = tf.ConfigProto(log_device_placement=True)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    K.set_session(sess)

    window_y, window_x = [], []
    model = None
    batch_size = consumer.get_batch_size()
    # num_batches_fed = consumer.get_num_batches_fed()
    batch_counter = 1
    y_pred_history = None
    x_pred_history = None
    y_history = None
    train_time, test_time = 0., 0.
    count_messages = 0
    waiting_to_train = False

    # Wait until first window available.
    while not consumer.get_initial_training_set():
        if consumer.is_time_out():
            return
        pass

    # Create and train model for the first time.
    with lock_train:

        initial_training_set = consumer.get_initial_training_set()

        # If first one training
        if consumer.get_weights() is None:

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
            y = to_categorical(window_y, len(consumer.get_classes()))

            consumer.add_training_data(x, y)
            consumer.set_num_features(x.shape[1])

            # create model, train it and save the weights.
            model = create_cnn_model(consumer.get_num_features(), consumer.get_num_classes(),
                                     consumer.get_loss_function())
            train_model(consumer, model, index)

            batch_counter += 1

        # If It has been already trained by other process ...
        else:
            # create model and load weights
            model = create_cnn_model(consumer.get_num_features(), consumer.get_num_classes(),
                                     consumer.get_loss_function())
            model.set_weights(consumer.get_weights())

    # Main loop
    while True:
        with lock_messages:
            count_messages += 1
            record = read_message(consumer)
        # if no message received...
        if record is None:
            # if timeout = true --> break
            if consumer.is_time_out():
                if consumer.get_debug():
                    print("P" + str(index) + ": Finish - " + str(count_messages) + " messages")
                if y_pred_history is not None:
                    save_history(consumer, x_pred_history, y_history, y_pred_history, test_time, 0.)
                write_results_file(consumer)
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

            y = to_categorical([window_y[-1]], consumer.get_num_classes())

            # Classify
            x_pred = np.expand_dims(np.array([list(record_values)]), axis=-1)
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
                # Format window data
                x = np.expand_dims(np.array(window_x), axis=-1)
                window_y = list(np.array(window_y))
                y = to_categorical(window_y, consumer.get_num_classes())
                consumer.add_training_data(x, y)
                train_time = 0.
                window_y, window_x = [], []
                waiting_to_train = True

            if waiting_to_train:
                available = lock_train.acquire(False)
                if available:
                    train_time = train_model(consumer, model, index)
                    lock_train.release()
                    waiting_to_train = False

            # Save metrics
            if len(y_pred_history) % batch_size == 0:
                save_history(consumer, x_pred_history, y_history, y_pred_history, test_time, train_time)
                y_pred_history = None
                x_pred_history = None
                y_history = None
                test_time = 0.
                train_time = 0.


# Buffer feeder
def buffer_feeder(consumer, lock):
    batch_size = consumer.get_batch_size()
    kafka_consumer = KafkaConsumer(consumer.get_topic(),
                                   group_id='DSClassification_Consumer_par_{}'.format(random.randrange(999999)),
                                   bootstrap_servers=consumer.get_bootstrap_servers(),
                                   auto_offset_reset='earliest' if consumer.get_from_begining() else 'latest',
                                   consumer_timeout_ms=5000,
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


def run(args):
    bootstrap_servers = args.bootstrap_servers.split(' ')
    topic = args.topic
    from_beginning = args.from_beginning
    batch_size = args.batch_size
    num_batches_fed = args.num_batches_fed
    output_path = args.output_path
    debug = args.debug

    BaseManager.register('Consumer', Consumer)
    manager = BaseManager()
    manager.start()
    consumer = manager.Consumer(batch_size=batch_size, num_batches_fed=num_batches_fed, results_path=output_path, topic=topic,
                                bootstrap_servers=bootstrap_servers, from_beginning=from_beginning, debug=debug)
    lock_messages = Lock()
    lock_train = Lock()

    pb = Process(target=buffer_feeder, args=[consumer, lock_messages])
    p0 = Process(target=DNN, args=[0, consumer, lock_messages, lock_train])
    p1 = Process(target=DNN, args=[1, consumer, lock_messages, lock_train])
    pb.start()
    p0.start()
    p1.start()
    pb.join()
    p0.join()
    p1.join()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_path",
                        help="Directory path where results will be stored.",
                        default='../../DSClassificationResults/DSClassificationResults_keras')

    parser.add_argument("--from_beginning",
                        help="Whether read messages from the beginning",
                        default=True,
                        action='store_true')

    parser.add_argument("--bootstrap_servers",
                        help="Bootstrap servers for Kafka producer",
                        default='localhost:9092')

    parser.add_argument("--topic",
                        help="Kafka topic name",
                        default='streams_breast')

    parser.add_argument("--batch_size",
                        help="Chunk size",
                        default=10)

    parser.add_argument("--num_batches_fed",
                        help="Number of batches fed to the training process",
                        default=40)

    parser.add_argument("--debug",
                        help="Whether print some log messages",
                        default=True)

    args = parser.parse_args()
    run(args)
