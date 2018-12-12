from kafka import KafkaConsumer
import json
import argparse
import random
import pandas as pd
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Conv1D, Flatten, MaxPool1D
from keras.regularizers import l2
from keras.layers.recurrent import LSTM
import numpy as np
import os
import time


def generateModels():
    model_fc = Sequential()
    model_fc.add(Dense(32, activation='relu', input_dim=2))
    model_fc.add(Dense(1, activation='sigmoid'))

    input = Input(shape=(2, 1), name="input")
    x = Dense(16, activation="relu", name="fc1")(input)
    x = LSTM(16, return_sequences=True, name='lstm_1')(x)
    x = Flatten()(x)
    x = Dense(1, activation="sigmoid", activity_regularizer=l2(0.0015), name="prediction")(x)
    model_1_unidir_lstm = Model(input, x)

    input = Input(shape=(2, 1), name="input")
    x = Dense(16, activation="relu", name="fc1")(input)
    x = LSTM(16, return_sequences=True, name='lstm_1')(x)
    x = LSTM(16, return_sequences=True, name='lstm_2')(x)
    x = Flatten()(x)
    x = Dense(1, activation="sigmoid", activity_regularizer=l2(0.0015), name="prediction")(x)
    model_2_unidir_lstm = Model(input, x)

    models = [
        {'model_name': 'fully-connected', 'model': model_fc},
        {'model_name': '1-Unidir-LSTM', 'model': model_1_unidir_lstm},
        {'model_name': '2-Unidir-LSTM', 'model': model_2_unidir_lstm}
    ]
    return models


def run(args):
    bootstrap_servers = args.bootstrap_servers
    topic = args.topic
    from_beginning = args.from_beginning
    results_path = args.results_path
    csv_file = args.csv_file
    model_name = args.model_name
    model = args.model

    consumer = KafkaConsumer(topic,
                             group_id='NAB_Consumer_{}'.format(random.randrange(999999)),
                             bootstrap_servers=bootstrap_servers,
                             auto_offset_reset='earliest' if from_beginning else 'latest',
                             value_deserializer=lambda m: json.loads(m.decode('ascii')),
                             consumer_timeout_ms=10000)

    # print(model.summary())

    loss = 'binary_crossentropy'
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

    num_samples = 0
    num_samples_correct = 0

    nab_rows = []
    for message in consumer:
        record = message.value
        if record['dataset'] == 'NAB_' + dataset and record['csv_file'] == csv_file:
            timestamp = record['timestamp']
            readable_timestamp = record['readable_timestamp']
            value = record['value']
            label = record['is_anomaly']

            X = np.expand_dims([timestamp, value], 0)
            if model_name != 'fully-connected':
                X = np.expand_dims(X, axis=2)
            y = np.expand_dims(label, 0)
            test_start_time = time.time()
            pred = model.predict_on_batch(X)
            test_end_time = time.time()
            test_duration = test_end_time - test_start_time
            if loss == 'binary_crossentropy':
                pred = 1.0 if pred > 0.5 else 0.0
                if pred == y:
                    num_samples_correct += 1
            train_start_time = time.time()
            model.train_on_batch(X, y)
            train_end_time = time.time()
            train_duration = train_end_time - train_start_time
            num_samples += 1
            acc = 100 * num_samples_correct / num_samples
            nab_rows.append([readable_timestamp, value, int(pred), label, acc, train_duration, test_duration])

    if nab_rows:
        output_path = os.path.join(results_path, 'Own_{}'.format(model_name), dataset)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        df = pd.DataFrame(nab_rows, columns=('timestamp', 'value', 'anomaly_score', 'label',
                                             'accuracy', 'training_time_s', 'test_time_s'))
        df.to_csv(os.path.join(output_path, 'Own_{}_{}'.format(model_name, csv_file)), index=False)

        acc = 100 * num_samples_correct / num_samples
        with open(os.path.join(results_path, 'accuracy_results.csv'), 'a') as results_file:
            results_file.write('{},{},Own_{},{}\n'.format(dataset, csv_file, model_name, acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nab_data_path",
                        help="NAB datasets root directory",
                        default='C:/Users/Alvaro/NAB/data')

    parser.add_argument("--from_beginning",
                        help="Whether read messages from the beginning",
                        default=False,
                        action='store_true')

    parser.add_argument("--bootstrap_servers",
                        help="Bootstrap servers for Kafka producer",
                        default=['localhost:9092'])

    parser.add_argument("--topic",
                        help="Kafka topic name",
                        default='streams_nab')

    parser.add_argument("--results_path",
                        help="Directory path where results will be saved",
                        default='C:/nab_results')

    args = parser.parse_args()

    nab_home = args.nab_data_path
    datasets = [x for x in os.listdir(nab_home) if os.path.isdir(os.path.join(nab_home, x))]
    models = generateModels()
    for dataset in datasets:
        csv_files = os.listdir(os.path.join(nab_home, dataset))
        for csv_file in csv_files:
            args.dataset = dataset
            args.csv_file = csv_file
            for model in models:
                args.model_name, args.model = model['model_name'], model['model']
                run(args)
