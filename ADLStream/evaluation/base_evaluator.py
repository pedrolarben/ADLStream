"""Implements an abstract object representing an evaluator."""

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from datetime import datetime


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
        xlabel (str, optional): x-axis label of the evolution plot.
            Defaults to "".
    """

    def __init__(
        self,
        results_file="ADLStream.csv",
        dataset_name=None,
        show_plot=True,
        plot_file=None,
        xlabel="",
    ):
        self.results_file = results_file
        self.dataset_name = dataset_name
        self.show_plot = show_plot
        self.plot_file = plot_file
        self.xlabel = xlabel

        self.x_eval = []
        self.y_eval = []
        self.o_eval = []
        self.metric_history = []

        self._create_results_file()

        self.fig = None
        self.ax = None
        self.line = None
        self.xlim = (0, 1)
        self.ylim = (0, 0.00001)
        self.xdata = []
        self.ydata = []

        self._initialize_plot()

    def _create_results_file(self):
        if self.results_file is not None:
            with open(self.results_file, "w") as f:
                f.write("timestamp,instances,metric\n")

    def _initialize_plot(self):
        if self.show_plot or self.plot_file is not None:
            fig, ax = plt.subplots()
            (line,) = ax.plot([], [], lw=2, label=self.xlabel)

            ax.grid()
            ax.set_title("ADLStream - {}".format(self.dataset_name))
            ax.set_ylabel(self.xlabel)
            ax.set_xlabel("Instances")

            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.legend()

            self.fig = fig
            self.ax = ax
            self.line = line

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
            with open(self.results_file, "a") as f:
                for i, value in enumerate(new_results):
                    f.write(
                        "{},{},{}\n".format(str(datetime.now()), instances[i], value,)
                    )

    def update_plot(self, new_results, instances):
        if self.show_plot or self.plot_file is not None:
            self.ydata += new_results
            self.xdata += instances

            self.xlim = (
                self.xlim[0],
                self.xlim[1] if self.xdata[-1] < self.xlim[1] else self.xdata[-1] + 1,
            )
            self.ylim = (
                self.ylim[0]
                if min(new_results) >= self.ylim[0]
                else min(new_results) - (min(new_results)) * 0.1,
                self.ylim[1]
                if max(new_results) <= self.ylim[1]
                else max(new_results) + max(new_results) * 0.1,
            )
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(self.ylim)

            self.line.set_data(self.xdata, self.ydata)
            self.line.set_label("{} ({:.4f})".format(self.xlabel, self.ydata[-1]))
            self.ax.legend(labels=["{} ({:.4f})".format(self.xlabel, self.ydata[-1])])
            plt.pause(0.0001)

    def update_predictions(self, context):
        """Gets new predictions from ADLStream context

        Args:
            context (ADLStreamContext)
        """
        x, y, o = context.get_predictions()
        self.x_eval += x
        self.y_eval += y
        self.o_eval += o

    def run(self, context):
        """Run evaluator

        This function update predictions from context, evaluate them and update result
        file and result plot.

        Args:
            context (ADLStreamContext)
        """
        while not context.is_finished():
            self.update_predictions(context)
            new_results, instances = self.evaluate()
            if new_results:
                self.write_results(new_results, instances)
                self.update_plot(new_results, instances)

        if self.plot_file is not None:
            self.fig.savefig(self.plot_file)
        if self.show_plot:
            plt.show()
