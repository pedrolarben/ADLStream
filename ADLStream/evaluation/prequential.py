"""Implements a prequential evaluator."""

from ADLStream.evaluation import InterleavedChunkEvaluator


class PrequentialEvaluator(InterleavedChunkEvaluator):
    """Prequential Evaluator.

    This evaluator implements the idea that more recent examples are more important. It
    uses a decaying factor. It is based on the inteleaved chunks evaluator, which
    incrementally updates the accuracy by evaluating chunks of data sequentially.

    The fading factor is implemented as follow:

        ```
        S = loss + fading_factor * S_prev
        N = 1 + fading_factor * N_prev
        preq_loss = S/N
        ```

    Arguments:
        chunck_size (int): Number of instances per chunk.
            the particular case of chunk_size = 1, represents prequential the
            interleaved train-then-test approach.
        metric (str): loss function.
            Possible options can be found in ADLStream.evaluation.metrics.
        fadding_factor (float, optional): Fadding factor.
            Defaults to 0.98.
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
        fadding_factor=0.98,
        results_file="ADLStream.csv",
        dataset_name=None,
        show_plot=True,
        plot_file=None,
        **kwargs
    ):
        self.fadding_factor = fadding_factor
        self.prev_metric = None
        self.prev_n = None
        super().__init__(
            chunk_size,
            metric,
            results_file=results_file,
            dataset_name=dataset_name,
            show_plot=show_plot,
            plot_file=plot_file,
            **kwargs
        )

    def compute_metric(self):
        new_metric = super().compute_metric()

        if self.prev_metric is None:
            self.prev_metric = new_metric
            self.prev_n = 1
            return new_metric

        new_metric = new_metric + self.fadding_factor * self.prev_metric
        n = 1 + self.fadding_factor * self.prev_n

        self.prev_metric = new_metric
        self.prev_n = n

        return new_metric / n
