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
import datetime


def generateModel(model_name):
    if model_name == '1-Unidir-LSTM':
        input = Input(shape=(2, 1), name="input")
        x = Dense(16, activation="relu", name="fc1")(input)
        x = LSTM(16, return_sequences=True, name='lstm_1')(x)
        x = Flatten()(x)
        x = Dense(1, activation="sigmoid", activity_regularizer=l2(0.0015), name="prediction")(x)
        model_1_unidir_lstm = Model(input, x)
        return model_1_unidir_lstm
    elif model_name == '2-Unidir-LSTM':
        input = Input(shape=(2, 1), name="input")
        x = Dense(16, activation="relu", name="fc1")(input)
        x = LSTM(16, return_sequences=True, name='lstm_1')(x)
        x = LSTM(16, return_sequences=True, name='lstm_2')(x)
        x = Flatten()(x)
        x = Dense(1, activation="sigmoid", activity_regularizer=l2(0.0015), name="prediction")(x)
        model_2_unidir_lstm = Model(input, x)
        return model_2_unidir_lstm
    else:
        model_fc = Sequential()
        model_fc.add(Dense(32, activation='relu', input_dim=2))
        model_fc.add(Dense(1, activation='sigmoid'))
        return model_fc


def run(args):
    bootstrap_servers = args.bootstrap_servers
    topic = args.topic
    from_beginning = args.from_beginning
    results_path = args.results_path
    model_name = args.classifier_name
    model = generateModel(model_name)
    dataset_name = args.dataset_name

    columns = ('timestamp', 'readable_timestamp', 'value', 'anomaly_score', 'label',
               'producer_timestamp', 'producer_readable_timestamp',
               'consumer_timestamp', 'consumer_readable_timestamp',
               'output_timestamp', 'output_readable_timestamp',
               'producer_delay')

    consumer = KafkaConsumer(topic,
                             group_id='NAB_Consumer_{}'.format(random.randrange(999999)),
                             bootstrap_servers=bootstrap_servers,
                             auto_offset_reset='earliest' if from_beginning else 'latest',
                             value_deserializer=lambda m: json.loads(m.decode('ascii')),
                             consumer_timeout_ms=20000)

    # print(model.summary())

    loss = 'binary_crossentropy'
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

    last_csv_file = None
    nab_rows = []
    for message in consumer:
        consumer_timestamp = int(time.time())
        consumer_readable_timestamp = datetime.datetime.fromtimestamp(consumer_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        record = message.value
        if record['dataset'] == 'NAB_' + dataset_name:
            timestamp = record['timestamp']
            readable_timestamp = record['readable_timestamp']
            value = record['value']
            label = record['is_anomaly']
            csv_file = record['csv_file']
            producer_delay = record['producer_delay']
            producer_timestamp = record['producer_timestamp']
            producer_readable_timestamp = record['producer_readable_timestamp']

            if last_csv_file is None:
                last_csv_file = csv_file

            if last_csv_file != csv_file and nab_rows:
                output_path = os.path.join(results_path, 'Own_{}'.format(model_name), dataset_name)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                df = pd.DataFrame(nab_rows, columns=columns)
                df.to_csv(os.path.join(output_path, 'Own_{}_{}'.format(model_name, csv_file)), index=False)
                nab_rows = []
                last_csv_file = csv_file

            X = np.expand_dims([timestamp, value], 0)
            if model_name != 'fully-connected':
                X = np.expand_dims(X, axis=2)
            y = np.expand_dims(label, 0)
            pred = model.predict_on_batch(X)
            model.train_on_batch(X, y)

            output_timestamp = int(time.time())
            output_readable_timestamp = datetime.datetime.fromtimestamp(output_timestamp).strftime('%Y-%m-%d %H:%M:%S')

            nab_rows.append([timestamp, readable_timestamp, value, int(pred), label,
                             producer_timestamp, producer_readable_timestamp,
                             consumer_timestamp, consumer_readable_timestamp,
                             output_timestamp, output_readable_timestamp,
                             producer_delay])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--bootstrap_servers",
                        help="Bootstrap servers for Kafka producer",
                        default=['localhost:9092'])

    parser.add_argument("--topic",
                        help="Kafka topic name",
                        default='streams_nab')

    parser.add_argument("--classifier_name",
                        help="fully-connected | 1-Unidir-LSTM | 2-Unidir-LSTM",
                        default='fully-connected')

    parser.add_argument("--dataset_name",
                        help="NAB dataset name",
                        default='realKnownCause')

    parser.add_argument("--from_beginning",
                        help="Whether read messages from the beginning",
                        default=False,
                        action='store_true')

    parser.add_argument("--results_path",
                        help="Directory path where results will be saved",
                        default='C:/py_nab_results')

    args = parser.parse_args()
    run(args)
