from kafka import KafkaConsumer
import random
import json

bootstrap_server = ['localhost:9092']
topic = 'streams_breast'

kafka_consumer = KafkaConsumer(topic,
                               group_id='test_consumer_{}'.format(random.randrange(999999)),
                               bootstrap_servers=bootstrap_server,
                               auto_offset_reset='earliest',
                               consumer_timeout_ms=5000,
                               value_deserializer=lambda m: json.loads(m.decode('ascii')))

i = 0
messages = []
for message in kafka_consumer:
    i += 1
    if i > 10:
        break
    messages.append(message)

print(str(len(messages)) + " messages received:")
print(messages)
