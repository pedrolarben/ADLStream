from kafka import KafkaConsumer
import json
import argparse
import random
import numpy as np
import pandas as pd
import time
import os
import sklearn.metrics as metrics
import traceback
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Conv1D, Flatten, MaxPool1D, Dropout
from keras.models import Input, Model
from keras.utils import to_categorical

# SEED = 1337
# np.random.seed(SEED)
# random.seed(SEED)

# Avoid full memory allocation on GPU
session_conf = tf.ConfigProto(log_device_placement=True)
session_conf.gpu_options.allow_growth = True
# tf.set_random_seed(SEED)
sess = tf.InteractiveSession(config=session_conf)
K.set_session(sess)

def get_cnn_model(num_features, num_classes):
    input = Input(shape=(num_features, 1), name='input')
    x = Conv1D(32, 7, padding='same', activation='relu', dilation_rate=3)(input)
    # x = MaxPool1D()(x)
    x = Conv1D(64, 5, padding='same', activation='relu', dilation_rate=3)(x)
    # x = MaxPool1D()(x)
    x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=3)(x)
    # x = MaxPool1D()(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes, activation="softmax", name="prediction")(x)
    model = Model(input, x)
    return model


def run(args):
    try:
        bootstrap_servers = args.bootstrap_servers.split(' ')
        topic = args.topic
        dataset_name = topic.replace('streams_', '')
        from_beginning = args.from_beginning
        batch_size = args.batch_size
        num_batches_fed = args.num_batches_fed
        results_path = os.path.join(args.output_path, dataset_name)

        consumer = KafkaConsumer(topic,
                                 group_id='DSClassification_Consumer_{}'.format(random.randrange(999999)),
                                 bootstrap_servers=bootstrap_servers,
                                 auto_offset_reset='earliest' if from_beginning else 'latest',
                                 consumer_timeout_ms=5000,
                                 value_deserializer=lambda m: json.loads(m.decode('ascii')))

        X_data = []
        y_data = []
        history = {}
        classes = None
        columns = None
        is_first_batch = True
        X_history = None
        y_history = None
        loss_func = 'categorical_crossentropy'
        clf = None
        batch_counter = 1
        average = 'weighted'
        clf_name = '3_Dilated_Conv'

        for message in consumer:
            record = message.value
            classes = record.pop('classes', None)
            y_data.append(record.pop('class', None))
            X_data.append(list(record.values()))
            if not columns:
                columns = list(record.keys())
                columns.extend(['class', 'prediction'])
            if len(X_data) % batch_size == 0:
                X = np.expand_dims(np.array(X_data), axis=-1)
                y = to_categorical(y_data, len(classes))
                if is_first_batch:
                    clf = get_cnn_model(X.shape[1], len(classes))
                    if not clf_name in history:
                        history[clf_name] = {
                            'data': pd.DataFrame(),
                            'metrics': pd.DataFrame()
                        }
                    if len(classes) == 2:
                        loss_func = 'binary_crossentropy'
                        average = 'binary'
                    clf.compile(loss=loss_func, optimizer='adam', metrics=['accuracy'])
                    X_history = X
                    y_history = y
                    clf.fit(X_history, y_history, batch_size, epochs=1, verbose=0)
                    is_first_batch = False
                    batch_counter += 1
                else:
                    test_time_start = time.time()
                    y_pred = clf.predict(X, verbose=0)
                    test_time_end = time.time()
                    test_time = test_time_end - test_time_start

                    if batch_counter < num_batches_fed:
                        X_history = np.vstack((X_history, X))
                        y_history = np.vstack((y_history, y))
                        batch_counter += 1

                    train_time_start = time.time()
                    print('Entrenando con las {} últimas instancias'.format(len(X_history)))
                    clf.fit(X_history, y_history, batch_size, epochs=1, verbose=0)
                    train_time_end = time.time()
                    train_time = train_time_end - train_time_start

                    if batch_counter >= num_batches_fed:
                        X_history = np.vstack((X_history, X))
                        y_history = np.vstack((y_history, y))
                        X_history = X_history[batch_size:]
                        y_history = y_history[batch_size:]
                        batch_counter = num_batches_fed

                    y_pred = np.argmax(y_pred, axis=1)
                    y = np.argmax(y, axis=1)

                    X = np.array(X)
                    if len(X.shape) == 3:
                        X = X.reshape(X.shape[0], -1)

                    records = pd.DataFrame(np.hstack((X, y.reshape(-1, 1), y_pred.reshape(-1, 1))))
                    history[clf_name]['data'] = pd.concat((history[clf_name]['data'], records), ignore_index=True)

                    prec, recall, fbeta, support = metrics.precision_recall_fscore_support(y, y_pred, average=average)
                    accuracy = metrics.accuracy_score(y, y_pred)
                    f1_score = metrics.f1_score(y, y_pred, average='weighted')
                    conf_matrix = metrics.confusion_matrix(y, y_pred)

                    try:
                        tn, fp, fn, tp = conf_matrix.ravel()
                    except ValueError as ve:
                        tn, fp, fn, tp = conf_matrix[0][0], 0, 0, 0

                    metrics_record = pd.Series({
                        'total': len(history[clf_name]['data']),
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
                    history[clf_name]['metrics'] = history[clf_name]['metrics'].append(metrics_record,
                                                                                       ignore_index=True)
                    X_data = []
                    y_data = []

        if X_data and y_data:
            X = np.expand_dims(np.array(X_data), axis=-1)
            y = to_categorical(y_data, len(classes))

            test_time_start = time.time()
            y_pred = clf.predict(X, verbose=0)
            test_time_end = time.time()
            test_time = test_time_end - test_time_start

            if batch_counter < num_batches_fed:
                X_history = np.vstack((X_history, X))
                y_history = np.vstack((y_history, y))
                batch_counter += 1

            train_time_start = time.time()
            print('Entrenando con los {} últimas instancias'.format(len(X_history)))
            clf.fit(X_history, y_history, batch_size, epochs=1, verbose=0)
            train_time_end = time.time()
            train_time = train_time_end - train_time_start

            if batch_counter >= num_batches_fed:
                X_history = np.vstack((X_history, X))
                y_history = np.vstack((y_history, y))
                X_history = X_history[batch_size:]
                y_history = y_history[batch_size:]
                batch_counter = num_batches_fed

            y_pred = np.argmax(y_pred, axis=1)
            y = np.argmax(y, axis=1)

            X = np.array(X)
            if len(X.shape) == 3:
                X = X.reshape(X.shape[0], -1)

            records = pd.DataFrame(np.hstack((X, y.reshape(-1, 1), y_pred.reshape(-1, 1))))
            history[clf_name]['data'] = pd.concat((history[clf_name]['data'], records), ignore_index=True)

            prec, recall, fbeta, support = metrics.precision_recall_fscore_support(y, y_pred, average=average)
            accuracy = metrics.accuracy_score(y, y_pred)
            f1_score = metrics.f1_score(y, y_pred, average='weighted')
            conf_matrix = metrics.confusion_matrix(y, y_pred)

            try:
                tn, fp, fn, tp = conf_matrix.ravel()
            except ValueError as ve:
                tn, fp, fn, tp = conf_matrix[0][0], 0, 0, 0

            metrics_record = pd.Series({
                'total': len(history[clf_name]['data']),
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
            history[clf_name]['metrics'] = history[clf_name]['metrics'].append(metrics_record, ignore_index=True)

        output_path = os.path.join(results_path, 'keras_' + clf_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        history[clf_name]['data'].columns = columns
        history[clf_name]['data'].to_csv(os.path.join(output_path, 'data.csv'),index=False)
        history[clf_name]['metrics'].to_csv(os.path.join(output_path, 'metrics.csv'),
                                            columns=('total', 'tp', 'tn', 'fp', 'fn', 'precision',
                                                     'recall', 'f1', 'fbeta',
                                                     'accuracy', 'train_time', 'test_time'),
                                            index=False)
    except Exception as e:
        print('Exception classifying stream: {} with {}'.format(topic, clf_name))
        print(traceback.format_exc())


if __name__ == "__main__":
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

    args = parser.parse_args()
    run(args)
