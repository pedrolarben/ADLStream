import numpy as np
import json
from kafka import KafkaConsumer
from keras.models import Sequential
from keras.layers import Dense

consumer = KafkaConsumer('streams-input',
                         group_id='MOAConsumerPy',
                         bootstrap_servers=['localhost:9092'],
                         auto_offset_reset='earliest',
                         value_deserializer=lambda m: json.loads(m.decode('ascii')))

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=10))
model.add(Dense(1, activation='sigmoid'))

loss = 'binary_crossentropy'
model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

num_samples = 0
num_samples_correct = 0

for message in consumer:
    values = np.array(message.value['instanceData']['attributeValues'], np.float64)
    X = np.expand_dims(values[:-1], 0)
    y = np.expand_dims(values[-1], 0)
    pred = model.predict_on_batch(X)
    if loss == 'binary_crossentropy':
        pred = 1.0 if pred > 0.5 else 0.0
        if pred == y:
            num_samples_correct += 1
    model.train_on_batch(X, y)
    num_samples += 1
    acc = 100 * num_samples_correct / num_samples
    print('{} instances processed with {}% accuracy'.format(num_samples, acc))