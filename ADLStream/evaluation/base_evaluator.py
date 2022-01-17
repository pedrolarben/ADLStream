"""Implements an abstract object representing an evaluator."""

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from datetime import datetime

from ADLStream.utils.plot_utils import EvaluationVisualizer


class BaseEvaluator(ABC):
    """Abstract base evaluator

    This is the base class for implementing a custom evaluator.

    Every `Evaluator` must have the properties below and implement `evaluate` with the
    signature `(new_results, instances) = evaluate()`. The `evaluate` function should
    contain the logic to:
        - Get validation metrics from validation data (`self.y_eval`, `self.o_eval`
          and `self.x_eval`).
        - Save metrics in `self.metric_history`.
        - Remove already evaluated data (`y_eval`, `o_eval` and `x_eval`) to keep memory
          free.
        - Return new computed accuracy and count of number of instances evaluated.

    Examples:
    ```python
        class MinimalEvaluator(BaseEvaluator):

            def __init__(self, metric='kappa', **kwargs):
                self.metric = metric
                super().__init__(**kwargs)

            def evaluate(self):
                new_results = []
                instances = []
                current_instance = len(self.metric_history)

                while self.y_eval and self.o_eval:
                    # Get metric
                    new_metric = metrics.evaluate(
                        self.metric,
                        self.y_eval[0]
                        self.o_eval[0]
                    )

                    # Save metric
                    self.metric_history.append(new_metric)

                    # Remove evaluated data
                    self.y_eval = self.y_eval[1:]
                    self.o_eval = self.o_eval[1:]
                    self.x_eval = self.x_eval[1:]

                    # Add number of instances evaluated
                    current_instance += 1
                    instances.append(current_instance)

                retun new_results, instances
    ```

    Arguments:
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
        ylabel (str, optional): y-axis label of the evolution plot.
            Defaults to "".
    """

    def __init__(
        self,
        results_file=None,
        predictions_file=None,
        dataset_name=None,
        show_plot=True,
        plot_file=None,
        ylabel="",
    ):
        self.results_file = results_file
        self.dataset_name = dataset_name
        self.predictions_file = predictions_file
        self.show_plot = show_plot
        self.plot_file = plot_file
        self.ylabel = ylabel

        self.x_eval = []
        self.y_eval = []
        self.o_eval = []
        self.metric_history = []

        self._create_results_file()

        if self.show_plot or self.plot_file is not None:
            self.visualizer = EvaluationVisualizer(self.dataset_name, self.ylabel)

    def _create_results_file(self):
        if self.results_file is not None:
            with open(self.results_file, "w") as f:
                f.write("timestamp,instances,metric\n")

    def start(self):
        self.visualizer.start()
        if self.predictions_file:
            self.predictions_file = open(self.predictions_file, "a")
        if self.results_file:
            self.results_file = open(self.results_file, "a")

    def end(self):
        self.predictions_file.close()
        self.results_file.close()

    @abstractmethod
    def evaluate(self):
        """Function that contains the main logic of the evaluator.
        In a generic scheme, this function should:
            - Get validation metrics from validation data (`self.y_eval`, `self.o_eval`
              and `self.x_eval`).
            - Save metrics in `self.metric_history`.
            - Remove already evaluated data (`y_eval`, `o_eval` and `x_eval`) to keep
              memory free.
            - Return new computed metrics and count of number of instances evaluated.

        Raises:
            NotImplementedError: This is an abstract method which should be implemented.

        Returns:
            new_metrics (list)
            instances(list)
        """
        raise NotImplementedError("Abstract method")

    def write_results(self, new_results, instances):
        if self.results_file is not None:
            for i, value in enumerate(new_results):
                self.results_file.write(
                    "{},{},{}\n".format(
                        str(datetime.now()),
                        instances[i],
                        value,
                    )
                )

    def write_predictions(self, preds):
        if self.predictions_file is not None:
            for _, prediction in enumerate(preds):
                self.predictions_file.write(f"{','.join(map(str, prediction))}\n")

    def update_plot(self, new_results, instances):
        if self.show_plot or self.plot_file is not None:
            self.visualizer.append_data(instances, new_results)

    def update_predictions(self, context):
        """Gets new predictions from ADLStream context

        Args:
            context (ADLStreamContext)
        """
        x, y, o = context.get_predictions()
        self.x_eval += x
        self.y_eval += y
        self.o_eval += o
        self.write_predictions(o)

    def run(self, context):
        """Run evaluator

        This function update predictions from context, evaluate them and update result
        file and result plot.

        Args:
            context (ADLStreamContext)
        """
        self.start()
        while not context.is_finished():
            self.update_predictions(context)
            new_results, instances = self.evaluate()
            if new_results:
                self.write_results(new_results, instances)
                self.update_plot(new_results, instances)

        if self.plot_file:
            self.visualizer.savefig(self.plot_file)
        if self.show_plot:
            self.visualizer.show()
        self.end()
