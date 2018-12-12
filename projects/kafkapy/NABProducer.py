import pandas as pd
import os
from kafka import KafkaProducer
import json
import argparse
import random
import datetime
import time

def run(args):
    nab_home_path = args.nab_path
    dataset = args.dataset
    bootstrap_servers = args.bootstrap_servers
    topic = args.topic
    delay = int(args.delay)

    producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                             client_id='NAB_Producer_{}'.format(random.randrange(999999)),
                             value_serializer=lambda m: json.dumps(m).encode('ascii'))

    data_dir = os.path.join(nab_home_path, 'data', dataset)
    csv_files = os.listdir(data_dir)

    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(data_dir, csv_file))
        windows_json = json.load(open(os.path.join(nab_home_path, 'labels/combined_windows.json')))
        windows = windows_json[dataset + '/' + csv_file]

        for idx, row in df.iterrows():
            timestamp = row['timestamp']
            anomaly = 0
            for t1, t2 in windows:
                if t1 <= timestamp <= t2:
                    anomaly = 1
            readable_timestamp = timestamp
            timestamp = int(datetime.datetime.strptime(readable_timestamp, "%Y-%m-%d %H:%M:%S").timestamp())
            sent_timestamp = int(time.time())
            sent_readable_timestamp = datetime.datetime.fromtimestamp(sent_timestamp).strftime('%Y-%m-%d %H:%M:%S')
            record = {
                'dataset': 'NAB_' + dataset,
                'csv_file': csv_file,
                'timestamp': timestamp,
                'readable_timestamp': readable_timestamp,
                'value': row['value'],
                'is_anomaly': anomaly,
                'producer_delay': delay,
                'producer_timestamp': sent_timestamp,
                'producer_readable_timestamp': sent_readable_timestamp
            }
            producer.send(topic, record)
            time.sleep(delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nab_path",
                        help="NAB root directory",
                        default='C:/Users/Alvaro/NAB')

    parser.add_argument("--dataset",
                        help="One of NAB Data Corpus: https://github.com/numenta/NAB/tree/master/data",
                        default='realTweets')

    parser.add_argument("--bootstrap_servers",
                        help="Bootstrap servers for Kafka producer",
                        default=['localhost:9092'])

    parser.add_argument("--topic",
                        help="Kafka topic name",
                        default='streams_nab')
    parser.add_argument("--delay",
                        help="Time delay between sending entries (seconds)",
                        default=0)

    args = parser.parse_args()
    run(args)
