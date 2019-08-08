import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

IPYTHON = True

try:
    from IPython.display import clear_output, display, HTML
except:
    IPYTHON = False


def isnotebook():
    try:
        from google import colab

        return True
    except:
        pass
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook, Spyder or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


class ProgressBar:
    def __init__(
        self,
        num_epochs: int,
        num_batches: int,
        init_epoch: int = 0,
        main_bar_description: str = "Training",
        epoch_child_bar_decription: str = "Epoch",
        destroy_on_completed: bool = False,
        keys_to_plot: List[str] = None,
        log_plot: bool = False,
    ):
        """
        PyTorch training progress bar.
        For usage in Jupyter lab type in your poetry shell:
         - jupyter labextension install @jupyter-widgets/jupyterlab-manager
        Known issues:
        - Jupyter lab adds small div of 12px when trying to replacing bar (known bug on ipywidgets)
        - Jupyter could give an (known) error for too high data rate in output. The common solution is launch notebook with:
            jupyter notebook(lab) --NotebookApp.iopub_data_rate_limit=10000000
        :param num_epochs: Total number of epochs
        :param num_batches: Number of training batches (e.g. len(trainloader))
        :param init_epoch: Initial epoch, for restoring training (default 0)
        :param main_bar_description: Description of main training bar
        :param epoch_child_bar_decription: Description of epoch progress bar
        :param destroy_on_completed: If True new epoch bar replace the old, otherwise add new bar
        :param keys_to_plot: keys of metrics to plot (works only in notebook mode and when destroy_on_completed=True)
        """
        self.num_batches = num_batches
        self.epoch_bar_description = main_bar_description
        self.batch_bar_description = epoch_child_bar_decription
        self.leave = not destroy_on_completed
        self.is_notebook = isnotebook()
        self.log_plot = log_plot
        if self.is_notebook:
            self.epoch_bar = tqdm.tqdm_notebook(
                desc=self.epoch_bar_description,
                total=num_epochs,
                leave=True,
                unit="ep",
                initial=init_epoch,
            )
        else:
            self.epoch_bar = tqdm.tqdm(
                desc=self.epoch_bar_description,
                total=num_epochs,
                leave=True,
                unit="ep",
                initial=init_epoch,
            )
        self.batch_bar = None
        self.show_plot = (
            destroy_on_completed
            and (keys_to_plot is not None)
            and self.is_notebook
            and IPYTHON
        )
        self.fig = None
        self.ax = None
        self.init_epoch = init_epoch
        self.epoch = init_epoch
        self.keys_to_plot = keys_to_plot
        self.dict_plot = {}
        if self.show_plot:
            for key in keys_to_plot:
                self.dict_plot[key] = []

    def start_epoch(self, epoch: int):
        """
        Initialize progress bar for current epoch
        :param epoch: epoch number
        :return:
        """
        self.epoch = epoch
        if self.is_notebook:
            self.batch_bar = tqdm.tqdm_notebook(
                desc=f"{self.batch_bar_description} {epoch}",
                total=self.num_batches,
                leave=self.leave,
            )
        else:
            self.batch_bar = tqdm.tqdm(
                desc=f"{self.batch_bar_description} {epoch}",
                total=self.num_batches,
                leave=self.leave,
            )

    def end_epoch(self, metrics: Dict[str, float] = None):
        """
        Update global epoch progress/metrics
        :param metrics: dictionary of metrics
        :return:
        """
        if metrics is None:
            metrics = {}

        self.batch_bar.set_postfix(metrics)
        self.batch_bar.miniters = 0
        self.batch_bar.mininterval = 0
        self.batch_bar.update(self.num_batches - self.batch_bar.n)
        self.batch_bar.close()

        self.epoch_bar.set_postfix(metrics)
        self.epoch_bar.update(1)

        if self.show_plot:
            for key in self.keys_to_plot:
                if key in metrics:
                    self.dict_plot[key].append(metrics[key])
                else:
                    print(
                        f"WARNING: Expected keys not given as metric {key} (plot disabled)"
                    )
                    self.show_plot = False
                    if self.ax is not None:
                        plt.close(self.ax.figure)
                    break
            if self.show_plot:
                if self.fig is None:
                    self.fig, self.ax = plt.subplots(1)
                    self.myfig = display(self.fig, display_id=True)
                self.ax.clear()
                for key in self.keys_to_plot:
                    if self.log_plot:
                        self.ax.semilogy(
                            range(self.init_epoch, self.epoch + 1), self.dict_plot[key]
                        )
                    else:
                        self.ax.plot(
                            range(self.init_epoch, self.epoch + 1), self.dict_plot[key]
                        )
                self.ax.legend(self.keys_to_plot)
                self.myfig.update(self.ax.figure)

    def training_step(self, train_metrics: Dict[str, float] = None):
        """
        Update training batch progress/metrics
        :param train_metrics:
        :return:
        """
        if train_metrics is None:
            train_metrics = {}
        self.batch_bar.set_postfix(train_metrics)
        self.batch_bar.update(1)

    def close(self):
        """
        Close bar
        :return:
        """
        self.epoch_bar.close()
        if self.show_plot:
            plt.close(self.ax.figure)
