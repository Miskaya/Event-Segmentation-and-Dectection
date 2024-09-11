import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CPDetector:
    """
    Parent class for change point detection.

    Args:
        data_frame: The data to be analyzed.
        num_steps: The number of steps to be analyzed.
    """
    def __init__(self, data_frame: pd.DataFrame, num_steps:int=-1):
        self.data = data_frame
        self.data_no_labels = data_frame.iloc[:, 2:len(data_frame.columns)-3]
        self.num_steps = num_steps if num_steps > 0 else len(data_frame)
        self.gt_cp = self.get_change_points()
        self.data_fields = self.get_data_fields()
        self.segmentations = {}
        self.fitted = False
        self.filtered_cp_pred = None

    def get_change_points(self):
        """
        Extracts the change points from the data.

        Returns:
            The change points.
        """
        last_label = self.data["Activity"][0]
        cp = []
        for idx, row in self.data.iterrows():
            label = row["Activity"]
            if label != last_label:
                cp.append(idx)
                last_label = label
        return np.array(cp)
    
    def plot_cp(self):
        """
        Plots the change points.
        """
        for field in self.data_fields:
            ax = self.data[field].plot(linewidth=0.5)
            ax.set_title(field)
            for cp in self.filtered_cp_pred:
                ax.axvline(x=cp, color='r', linestyle='--', linewidth=0.5)
            for cp in self.gt_cp:
                ax.axvline(x=cp, color='g', linestyle='--', linewidth=0.5)
            plt.show()
            plt.close()
    
    def get_data_fields(self):
        """
        Gets the data fields, exclude indices and labels.

        Returns:
            The data fields.
        """
        return self.data.columns[2:len(self.data.columns)-3]

    def check_match(self, cp, margin: int=25):
        """
        Checks if the predicted change point is within the margin
        from a ground truth change point.

        Args:
            cp: The ground truth change point.
            margin: The margin to be used around the change point.
        
        Returns:
            True if the change point is within the margin, False otherwise.
        """
        for p in self.filtered_cp_pred:
            if p > cp - margin and p < cp + margin:
                return True
        return False

    def get_chunk_sizes(self):
        """
        Gets the sizes of the chunks between the change points.

        Returns:
            The sizes of the chunks
        """
        cps = self.filtered_cp_pred
        chunks = [cps[0]]
        for i, cp in enumerate(cps[0:-1]):
            chunks.append(cps[i+1]-cps[i])
        chunks.append(len(self.data) - cps[-1])
        return chunks

    def get_cp_indices(self):
        """
        Gets the indices of the change points relative to the start of the
        dataset, relative to the start of the subject, and by timestamp for
        both the ground truth and predicted change points.

        Returns:
            The indices of the change points.
        """
        cp_idx = self.filtered_cp_pred.copy()
        if len(self.data) in cp_idx:
            cp_idx.remove(len(self.data))
        pred_time_idx = self.data["time"][cp_idx].tolist()
        pred_total_idx = self.data["index"][cp_idx].tolist()
        gt_total_idx = self.data["index"][self.gt_cp].tolist()
        gt_time_idx = self.data["time"][self.gt_cp].tolist()
        return [cp_idx, pred_total_idx, pred_time_idx, self.gt_cp, gt_total_idx, gt_time_idx]
    
    def get_missed_change_points(self, margin: int=25):
        """
        Gets the missed change points, their indices, and
        their edge (from which to which label).

        Args:
            margin: The margin to be used around the change point.

        Returns:
            The missed change points, their indices, and their edges.
        """
        missed_cps = []
        missed_idxs = []
        missed_edges = []
        for i, cp in enumerate(self.gt_cp):
            if not self.check_match(cp, margin=margin) and cp:
                missed_cps.append(cp)
                missed_idxs.append(i)
                missed_edges.append((self.data["Activity"][cp-1], self.data["Activity"][cp]))
        return missed_cps, missed_idxs, missed_edges
