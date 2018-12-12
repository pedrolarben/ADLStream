from kafka import KafkaConsumer
import json
import argparse
import random

def run(args):
    dataset = args.dataset
    bootstrap_servers = args.bootstrap_servers
    topic = args.topic
    from_beginning = args.from_beginning

    consumer = KafkaConsumer(topic,
                             group_id='NAB_Consumer_{}'.format(random.randrange(999999)),
                             bootstrap_servers=bootstrap_servers,
                             auto_offset_reset='earliest' if from_beginning else 'latest',
                             value_deserializer=lambda m: json.loads(m.decode('ascii')))

    for message in consumer:
        record = message.value
        if record['dataset'] == 'NAB_' + dataset:
            print(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        help="One of NAB Data Corpus: https://github.com/numenta/NAB/tree/master/data",
                        default='realTweets')

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

    args = parser.parse_args()
    run(args)
