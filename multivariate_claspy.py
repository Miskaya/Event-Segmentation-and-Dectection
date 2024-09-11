import claspy
import pickle
import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from claspy.segmentation import BinaryClaSPSegmentation
from claspy.tests.evaluation import f_measure, covering

from cp_detector import CPDetector


class MultivariateClaspy(CPDetector):
    """
    Class for the Multivariate Claspy segmentation method.

    Args:
        data_frame: The data to be analyzed.
        cfg: The configuration file to be used.
        num_steps: The number of steps to be analyzed.
        chunk_size: The size of the chunks to be used.
    """
    def __init__(self, data_frame: pd.DataFrame, cfg: str, num_steps: int=-1, chunk_size: int=25000):
        super().__init__(data_frame, num_steps)
    
        self.cfg = self.load_cfg(cfg)
        self.chunk_size = chunk_size
    
    def load_cfg(self, filename: str):
        """
        Loads the configuration file.

        Args:
            filename: The name of the configuration file.

        Returns:
            The configuration file.
        """
        with open(filename, 'r') as stream:
            cfg = yaml.safe_load(stream)
        return cfg
    
    def fit_all(self):
        """
        Finds change points in the data.
        """
        self.segmentations = {}
        predictions = []
        nchunks = len(self.data_no_labels) // self.chunk_size + 1
        for field in self.data_fields:
            if self.data_no_labels[field].nunique()!=1:
                for i in range(nchunks):
                    if i < nchunks - 1:
                        chunk = self.data_no_labels.iloc[i*self.chunk_size: (i+1)*self.chunk_size].reset_index(drop=True)
                    else:
                        chunk = self.data_no_labels.iloc[i*self.chunk_size:].reset_index(drop=True)
                    clasp = BinaryClaSPSegmentation(distance="euclidean_distance", n_jobs=7).fit(chunk[field].to_numpy())
                    self.segmentations[field] = clasp
                    prediction = clasp.predict() + i * self.chunk_size
                    predictions.append(prediction)
        self.cp_pred = np.concatenate(predictions)
        self.fitted = True

    def plot_all(self):
        """
        Plots the segmentations.
        """
        assert self.fitted, "Model has not been fitted yet"

        for field, clasp in self.segmentations.items():
            clasp.plot(gt_cps=self.gt_cp, heading=field)

    def plot_histogram(self, bins: int=75):
        """
        Plots the histogram of the change points.
        """
        assert self.fitted, "Model has not been fitted yet"

        counts, bins = np.histogram(self.cp_pred, bins=bins)
        plt.stairs(counts, bins)

    def plot_cp(self, fig_size: tuple=(20, 10), font_size: int=26):
        """
        Plots the change points.

        Args:
            fig_size: The size of the figure.
            font_size: The size of the font.

        Returns:
            The plot.
        """
        assert self.fitted, "Model has not been fitted yet"
        assert self.filtered_cp_pred is not None, "Must filter cp"
    
        fig, (ax1) = plt.subplots(1, sharex=True, gridspec_kw={"hspace": .05}, figsize=fig_size)

        for field in self.data_fields:
            ax1.plot(self.data[field].to_numpy())

        ax1.set_xlabel("split point", fontsize=font_size)

        for tick in ax1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(font_size)

        for tick in ax1.yaxis.get_major_ticks():
            tick.label1.set_fontsize(font_size)

        for idx, true_cp in enumerate(self.gt_cp):
            ax1.axvline(x=true_cp, linewidth=2, color="r", label=f"True Change Point" if idx == 0 else None)

        for idx, found_cp in enumerate(self.filtered_cp_pred):
            ax1.axvline(x=found_cp, linewidth=2, color="g", label="Predicted Change Point" if idx == 0 else None)

        ax1.legend(prop={"size": font_size})

        return ax1
    
    def kernel_cp_combine(self, kernel_size: int=51, peak_height: int=3, peak_distance: int=500, plot: bool=False):
        """
        Combines the change points using a kernel. Account for when there are a lot
        of change points nearby each other. Then, the peaks are detected and used as
        the final change points.

        Args:
            kernel_size: The size of the kernel.
            peak_height: The height of the peak.
            peak_distance: The distance between the peaks.
            plot: Whether to plot the result.
        
        Returns:
            The filtered change points.
        """
        
        assert self.fitted, "Model has not been fitted yet"

        count = np.zeros((self.num_steps))
        for p in self.cp_pred:
            i = int(p)
            count[i] = count[i] + 1

        kernel = np.ones(kernel_size)
        signal = np.convolve(count, kernel, mode='full')

        if plot:
            plt.plot(signal)

        self.filtered_cp_pred, _ = find_peaks(signal, height=peak_height, distance=peak_distance)
    
    def get_f1score(self, margin: int=25, return_PR: bool=True):
        """
        Computes the F1 score of the model.

        Args:
            margin: The margin to be used around the change point.
            return_PR: Whether to return the precision and recall.

        Returns:
            The F1 score, the precision, and the recall.
        """
        annotations = {1: self.gt_cp}
        return f_measure(annotations, self.filtered_cp_pred, margin=margin, return_PR=return_PR)

    def get_covering(self):
        """
        Gets the covering score of the model.

        Returns:
            The covering score.
        """
        annotations = {1: self.gt_cp}
        return covering(annotations, self.filtered_cp_pred, self.num_steps)
        
        