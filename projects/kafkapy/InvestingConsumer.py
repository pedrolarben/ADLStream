from kafka import KafkaConsumer
import json
import argparse
import random

def run(args):
    bootstrap_servers = args.bootstrap_servers
    topic = args.topic
    from_beginning = args.from_beginning

    consumer = KafkaConsumer(topic,
                             group_id='Investing_Consumer_{}'.format(random.randrange(999999)),
                             bootstrap_servers=bootstrap_servers,
                             auto_offset_reset='earliest' if from_beginning else 'latest',
                             value_deserializer=lambda m: json.loads(m.decode('ascii')))

    for message in consumer:
        record = message.value
        print(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--from_beginning",
                        help="Whether read messages from the beginning",
                        default=False,
                        action='store_true')

    parser.add_argument("--bootstrap_servers",
                        help="Bootstrap servers for Kafka producer",
                        default=['localhost:9092'])

    parser.add_argument("--topic",
                        help="Kafka topic name",
                        default='streams_investingdotcom')

    args = parser.parse_args()
    run(args)
