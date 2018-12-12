import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
import itertools
import argparse


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def drop_rows(df, percentage, from_start=True):
    total_rows = len(df)
    num_rows_to_drop = int(total_rows * percentage / 100)
    if from_start:
        df.drop(df.index[[range(num_rows_to_drop)]], inplace=True)
    else:
        df.drop(df.index[[range(-1, -num_rows_to_drop, -1)]], inplace=True)


def eval_results_delay(results_path, csv_out_path, percentage_to_drop=0, from_start=True, fix_timestamp=False):
    csv_files = glob.glob(os.path.join(results_path, '*/*/**.csv'))
    rows = []
    for csv_filepath in csv_files:
        print('Processing {}'.format(csv_filepath))
        csv_filename = os.path.basename(csv_filepath)
        dataset_name = os.path.basename(os.path.abspath(os.path.join(csv_filepath, os.pardir)))
        classifier_name = os.path.basename(os.path.abspath(os.path.join(os.path.abspath(os.path.join(csv_filepath, os.pardir)), os.pardir)))
        csv_filename = csv_filename.replace(classifier_name+'_', '')
            
        df = pd.read_csv(csv_filepath)

        if fix_timestamp:
            df['output_timestamp'] = [int(x / 1000) for x in df['output_timestamp']]
            df['consumer_timestamp'] = [int(x / 1000) for x in df['consumer_timestamp']]
        df['delay'] = df['output_timestamp'] - df['producer_timestamp']
        df = df[['timestamp', 'value', 'anomaly_score', 'label', 'producer_timestamp', 'consumer_timestamp',
                 'output_timestamp', 'delay']]
        df.columns = ['timestamp', 'value', 'anomaly_score', 'label', 'prod_time', 'cons_time', 'out_time', 'delay']

        if percentage_to_drop > 0:
            drop_rows(df, percentage_to_drop, from_start)
        y_true, y_pred = df['label'], df['anomaly_score']

        total = len(df)

        num_classes = len(y_true.unique())

        average = 'binary'
        if num_classes == 1:
            average = 'weighted'

        prec, recall, fbeta, support = metrics.precision_recall_fscore_support(y_true, y_pred, average=average)
        accuracy = metrics.accuracy_score(y_true, y_pred)
        avg_prec = None
        if num_classes == 2:
            avg_prec = metrics.average_precision_score(y_true, y_pred)
        f1_score = metrics.f1_score(y_true, y_pred, average='weighted')

        conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    #     plt.figure()
    #     plot_confusion_matrix(conf_matrix, classes=('0', '1'))

        if len(y_true.unique()) == 2:
            tn, fp, fn, tp = conf_matrix.ravel()
        else:
            tn, fp, fn, tp = conf_matrix[0][0], 0, 0, 0

        acc_delay_millis = df['delay'].sum()

        row = {
            'dataset': dataset_name,
            'time_serie': csv_filename,
            'classifier': classifier_name,
            'total': total,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'precision': prec,
            'recall': recall,
            'f1': f1_score,
            'fbeta': fbeta,
            'accuracy': accuracy,
            'acc_delay_millis': acc_delay_millis
        }
        rows.append(row)
    df_rows = pd.DataFrame(rows)
    df_rows = df_rows[
        ['dataset', 'time_serie', 'classifier', 'total', 'tp', 'tn', 'fp', 'fn', 'precision', 'recall', 'f1', 'fbeta',
         'accuracy', 'acc_delay_millis']
    ]
    out_filename = os.path.basename(csv_out_path)
    out_path = os.path.abspath(os.path.join(csv_out_path, os.pardir))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if percentage_to_drop > 0:
        out_filename = '{}_perc_drop_{}_{}'.format(percentage_to_drop, 'from_start' if from_start else 'at_the_end', out_filename)
    df_rows.to_csv(os.path.join(out_path, out_filename), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_path",
                        help="Directory path where results are stored.",
                        default='C:/Users/Alvaro/NAB_results/MOA')

    parser.add_argument("--csv_out_path",
                        help="CSV path where evaluation results will be written.",
                        default='C:/temp/eval_results.csv')

    parser.add_argument("--ptd",
                        help="Percentage of rows that will be dropped from the dataframe.",
                        default=0)

    parser.add_argument("--from_start",
                        help='Whether drop dataframe rows from the start or at the end.',
                        default=False,
                        action='store_true')

    parser.add_argument("--fix_timestamp",
                        help='Fix timestamp values of MOA results',
                        default=False,
                        action='store_true')

    args = parser.parse_args()
    ptd = int(args.ptd)
    eval_results_delay(args.results_path, args.csv_out_path, ptd, args.from_start, args.fix_timestamp)
