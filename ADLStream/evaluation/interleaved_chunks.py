"""Implements a interleaved chunks evaluator."""

from ADLStream.evaluation import metrics, BaseEvaluator


class InterleavedChunkEvaluator(BaseEvaluator):
    """Interleave chunks Evaluator.

    THis evaluator incrementally updates the accuracy by evaluating chunks of data
    sequentially.


    Arguments:
        chunck_size (int): Number of instances per chunk.
            the particular case of chunk_size = 1, represents prequential the
            interleaved train-then-test approach.
        metric (str): loss function.
            Possible options can be found in ADLStream.evaluation.metrics.
        results_file (str, optional): Name of the csv file where to write results.
            If None, no csv file is created.
            Defaults to "ADLStream.csv".
        dataset_name (str, optional): Name of the data to validate.
            Defaults to None.
        show_plot (bool, optional): Whether to plot the evolution of the metric.
            Defaults to True.
        plot_file (str, optional): Name of the plot image file.
            If None, no image is saved.
            Defaults to None.
    """

    def __init__(
        self,
        chunk_size,
        metric,
        results_file="ADLStream.csv",
        dataset_name=None,
        show_plot=True,
        plot_file=None,
        **kwargs
    ):
        self.chunk_size = chunk_size
        self.metric = metric
        super().__init__(
            results_file=results_file,
            dataset_name=dataset_name,
            show_plot=show_plot,
            plot_file=plot_file,
            ylabel=self.metric,
            **kwargs
        )

    def compute_metric(self):
        new_metric = metrics.evaluate(
            self.metric,
            self.y_eval[: self.chunk_size],
            self.o_eval[: self.chunk_size],
        )

        return new_metric

    def evaluate(self):
        new_results = []
        instances = []
        instances_index = len(self.metric_history)
        # Chunks loop
        while (
            len(self.y_eval) >= self.chunk_size and len(self.o_eval) >= self.chunk_size
        ):
            # Get metric
            new_metric = self.compute_metric()

            # Save metric
            self.metric_history.append(new_metric)
            new_results.append(new_metric)

            # Remove eval data
            self.y_eval = self.y_eval[self.chunk_size :]
            self.o_eval = self.o_eval[self.chunk_size :]
            self.x_eval = self.x_eval[self.chunk_size :]

            # Add number of instances evaluated
            instances_index += 1
            instances.append(self.chunk_size * instances_index)

        return new_results, instances
