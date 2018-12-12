import pandas as pd
import os
from kafka import KafkaProducer
import json
import argparse


def run(args):
    yahoo_home_path = args.root_path
    dataset = args.dataset
    bootstrap_servers = args.bootstrap_servers
    topic = args.topic

    data_dir = os.path.join(yahoo_home_path, dataset)
    csv_files = os.listdir(data_dir)

    producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                             client_id='Yahoo_Anomalies_Producer',
                             value_serializer=lambda m: json.dumps(m).encode('ascii'))

    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(data_dir, csv_file))
        for idx in df.index:
            record = df.loc[idx].to_dict()
            record['dataset'] = 'Yahoo_'+dataset
            record['csv_file'] = csv_file
            producer.send(topic, record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path",
                        help="Yahoo anomalies root directory",
                        default='C:/Users/Alvaro/Downloads/ydata-labeled-time-series-anomalies-v1_0')

    parser.add_argument("--dataset",
                        help="A1Benchmark | A2Benchmark | A3Benchmark | A4Benchmark",
                        default='A1Benchmark')

    parser.add_argument("--bootstrap_servers",
                        help="Bootstrap servers for Kafka producer",
                        default=['localhost:9092'])

    parser.add_argument("--topic",
                        help="Kafka topic name",
                        default='streams_yahoo')

    args = parser.parse_args()
    run(args)
