from kafka import KafkaProducer
import json
import argparse
import random
import scipy.io.arff as arff
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm

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
        class_name = 'class' if 'class' in df.columns else 'target'

        classes = np.unique(decode_class(df[class_name])).astype(np.int)
        print(classes)
        min_classes = min(classes)
        classes = classes - min_classes
        classes = [int(c) for c in classes]

        for _, entry in tqdm(enumerate(data)):
            record = {'classes': list(classes)}
            for _, attr in enumerate(attributes):
                value = entry[attr]
                if attr == class_name:
                    attr = 'class'
                    value = int(decode_class(value)) - int(min_classes)

                elif type(value) == np.bytes_:
                    value = value.decode('UTF-8')

                    if meta[attr][0] == 'nominal':
                        #print(meta[attr], value)
                        value = meta[attr][1].index(value)

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
