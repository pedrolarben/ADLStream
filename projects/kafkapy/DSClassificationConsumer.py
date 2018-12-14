from kafka import KafkaConsumer
import json
import argparse
import random
import numpy as np
import pandas as pd
import time
import os
from sklearn.exceptions import NotFittedError
import sklearn.metrics as metrics
import traceback

from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron, SGDClassifier, PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier

np.random.seed(1337)

classifiers = [
    {'name': 'BernoulliNB', 'clf': BernoulliNB()},
    {'name': 'Perceptron', 'clf': Perceptron(penalty='l2')},
    {'name': 'SGDClassifier', 'clf': SGDClassifier()},
    {'name': 'PassiveAggressiveClassifier', 'clf': PassiveAggressiveClassifier()},
    {'name': 'MLPClassifier', 'clf': MLPClassifier()}
]


def test_train_eval(history, X_data, y_data, classes):
    for clf_info in classifiers:
        clf_name = clf_info['name']
        clf = clf_info['clf']
        if not clf_name in history:
            history[clf_name] = {
                'data': pd.DataFrame(),
                'metrics': pd.DataFrame()
            }
        try:
            test_time_start = time.clock()
            y_pred = clf.predict(X_data)
            test_time_end = time.clock()
            test_time = test_time_end - test_time_start

            records = pd.DataFrame(np.hstack(
                (np.array(X_data), np.array(y_data).reshape(-1, 1), y_pred.reshape(-1, 1))))
            history[clf_name]['data'] = pd.concat((history[clf_name]['data'], records), ignore_index=True)

            train_time_start = time.clock()
            clf.partial_fit(X_data, y_data)
            train_time_end = time.clock()
            train_time = train_time_end - train_time_start

            average = 'binary' if len(classes) == 2 else 'weighted'
            prec, recall, fbeta, support = metrics.precision_recall_fscore_support(y_data, y_pred,
                                                                                   average=average)
            accuracy = metrics.accuracy_score(y_data, y_pred)
            f1_score = metrics.f1_score(y_data, y_pred, average='weighted')
            conf_matrix = metrics.confusion_matrix(y_data, y_pred)

            try:
                tn, fp, fn, tp = conf_matrix.ravel()
            except ValueError as ve:
                tn, fp, fn, tp = conf_matrix[0][0], 0, 0, 0

            metrics_record = pd.Series({
                'total': len(history[clf_name]['data']),
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'tp': tp,
                'precision': prec,
                'recall': recall,
                'f1': f1_score,
                'fbeta': fbeta,
                'accuracy': accuracy,
                'train_time': train_time,
                'test_time': test_time
            })
            history[clf_name]['metrics'] = history[clf_name]['metrics'].append(metrics_record, ignore_index=True)

        except NotFittedError as e:
            clf.partial_fit(X_data, y_data, classes)


def run(args):
    bootstrap_servers = args.bootstrap_servers.split(' ')
    topic = args.topic
    dataset_name = topic.replace('streams_', '')
    from_beginning = args.from_beginning
    batch_size = args.batch_size
    results_path = os.path.join(args.output_path, dataset_name)

    consumer = KafkaConsumer(topic,
                             group_id='DSClassification_Consumer_{}'.format(random.randrange(999999)),
                             bootstrap_servers=bootstrap_servers,
                             auto_offset_reset='earliest' if from_beginning else 'latest',
                             consumer_timeout_ms=5000,
                             value_deserializer=lambda m: json.loads(m.decode('ascii')))

    try:
        X_data = []
        y_data = []
        history = {}
        classes = None
        columns = None
        for message in consumer:
            record = message.value
            classes = record.pop('classes', None)
            y_data.append(record.pop('class', None))
            X_data.append(list(record.values()))
            if not columns:
                columns = list(record.keys())
                columns.extend(['class', 'prediction'])
            if len(X_data) % batch_size == 0:
                test_train_eval(history, X_data, y_data, classes)
                X_data = []
                y_data = []

        if X_data and y_data:
            test_train_eval(history, X_data, y_data, classes)

        for clf_info in classifiers:
            clf_name = clf_info['name']
            output_path = os.path.join(results_path, 'sklearn_' + clf_name)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            history[clf_name]['data'].columns = columns
            history[clf_name]['data'].to_csv(os.path.join(output_path, 'data.csv'),
                                             index=False)
            history[clf_name]['metrics'].to_csv(os.path.join(output_path, 'metrics.csv'),
                                                columns=('total', 'tp', 'tn', 'fp', 'fn', 'precision',
                                                         'recall', 'f1', 'fbeta',
                                                         'accuracy', 'train_time', 'test_time'),
                                                index=False)
    except Exception as e:
        print('Exception classifying stream: {} with {}'.format(topic, clf_name))
        print(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_path",
                        help="Directory path where results will be stored.",
                        default='../../DSClassificationResults/DSClassificationResults_keras')

    parser.add_argument("--from_beginning",
                        help="Whether read messages from the beginning",
                        default=False,
                        action='store_true')

    parser.add_argument("--bootstrap_servers",
                        help="Bootstrap servers for Kafka producer",
                        default='localhost:9092')

    parser.add_argument("--topic",
                        help="Kafka topic name",
                        default='streams_classification')

    parser.add_argument("--batch_size",
                        help="Chunk size",
                        default=10)

    args = parser.parse_args()
    run(args)
