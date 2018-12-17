from kafka import KafkaProducer
import random
import json

bootstrap_server = 'localhost:9092'
topic = 'test'
number_messages = 10

producer = KafkaProducer(bootstrap_servers=bootstrap_server,
                         client_id='test_producer_{}'.format(random.randrange(999999)),
                         value_serializer=lambda m: json.dumps(m).encode('ascii'))


for i in range(number_messages):
    message = "This is the message number " + str(i)
    producer.send(topic, message)

producer.close()

print(str(number_messages) + " messages sent")
