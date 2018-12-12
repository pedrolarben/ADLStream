from kafka import KafkaProducer
import json
import argparse
import random
import scipy.io.arff as arff
import numpy as np
import pandas as pd


def run(args):
    file_path = args.file_path
    bootstrap_servers = args.bootstrap_servers
    topic = args.topic

    producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                             client_id='ARFF_Producer_{}'.format(random.randrange(999999)),
                             value_serializer=lambda m: json.dumps(m).encode('ascii'))

    data, meta = arff.loadarff(file_path)
    attributes = [x for x in meta]
    df = pd.DataFrame(data)
    classes = np.unique(df['class']).astype(np.int).tolist()

    for _, entry in enumerate(data):
        record = {'classes': classes}
        for _, attr in enumerate(attributes):
            value = entry[attr]
            if type(value) == np.bytes_:
                value = int(value.decode('UTF-8'))
            record[attr] = value
        producer.send(topic, record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_path",
                        help="ARFF file path")

    parser.add_argument("--bootstrap_servers",
                        help="Bootstrap servers for Kafka producer",
                        default=['localhost:9092'])

    parser.add_argument("--topic",
                        help="Kafka topic name",
                        default='streams_arff')

    args = parser.parse_args()
    run(args)
