import os
import glob
import pandas as pd
import sklearn.metrics as metrics
import argparse


def eval_results(results_path, csv_out_path):
    csv_files = glob.glob(os.path.join(results_path, '*/*/data.csv'))
    rows = []
    for csv_filepath in csv_files:
        clf = os.path.basename(os.path.abspath(os.path.join(csv_filepath, os.pardir)))
        dataset = os.path.basename(
            os.path.abspath(os.path.join(os.path.abspath(os.path.join(csv_filepath, os.pardir)), os.pardir)))
        df = pd.read_csv(csv_filepath)
        y_true, y_pred = df['class'], df['prediction']
        total = len(df)
        num_classes = len(y_true.unique())
        average = 'binary' if num_classes == 2 else 'weighted'
        prec, recall, fbeta, support = metrics.precision_recall_fscore_support(y_true, y_pred, average=average)
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1_score = metrics.f1_score(y_true, y_pred, average='weighted')
        conf_matrix = metrics.confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = None, None, None, None
        if num_classes == 2:
            try:
                tn, fp, fn, tp = conf_matrix.ravel()
            except ValueError as ve:
                tn, fp, fn, tp = conf_matrix[0][0], 0, 0, 0

        df_metrics = pd.read_csv(os.path.join(results_path, dataset, clf, 'metrics.csv'))
        train_time_s = df_metrics['train_time'].mean()
        test_time_s = df_metrics['test_time'].mean()

        row = {
            'dataset': dataset,
            'classifier': clf,
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
            'train_time_s': train_time_s,
            'test_time_s': test_time_s
        }
        rows.append(row)

    df_rows = pd.DataFrame(rows)
    df_rows = df_rows[
        ['dataset', 'classifier', 'total', 'tp', 'tn', 'fp', 'fn', 'precision', 'recall', 'f1', 'fbeta', 'accuracy',
         'train_time_s', 'test_time_s']]
    df_rows.to_csv(csv_out_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_path",
                        help="Directory path where results are stored.",
                        default='/home/aarcos/DSClassificationResults')

    parser.add_argument("--csv_out_path",
                        help="CSV path where evaluation results will be written.",
                        default='/home/aarcos/DSClassificationResults/benchmark_results.csv')

    args = parser.parse_args()
    eval_results(args.results_path, args.csv_out_path)
