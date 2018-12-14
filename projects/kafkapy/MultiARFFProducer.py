from kafka import KafkaProducer
import json
import argparse
import random
import scipy.io.arff as arff
import numpy as np
import pandas as pd
import glob
import os


def run(args):
    dir_path = args.dir_path
    bootstrap_servers = args.bootstrap_servers.split(' ')

    dir_name = os.path.dirname(__file__)
    arff_files = glob.glob(os.path.join(dir_name, dir_path, '**/*.arff'), recursive=True)
    for file_path in arff_files:
        producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                                 client_id='ARFF_Producer_{}'.format(random.randrange(999999)),
                                 value_serializer=lambda m: json.dumps(m).encode('ascii'))
        topic = 'streams_' + os.path.basename(file_path).replace('.arff', '')
        print('Creating topic for {}'.format(topic))
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
        producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir_path",
                        help="Dir path where ARFF files are stored",
                        default='../../datasets_arff')

    parser.add_argument("--bootstrap_servers",
                        help="Bootstrap servers for Kafka producer",
                        default='localhost:9092')

    args = parser.parse_args()
    run(args)
