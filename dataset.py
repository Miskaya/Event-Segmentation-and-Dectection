import pickle

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class SubjectDataset:
    def __init__(self, dataframe: pd.DataFrame, index: int):
        self.data = dataframe.reset_index(drop=False)
        self.labels = self.get_unique_labels()
        self.index = index
        self.labelwise_datasets = self.split_dataset_by_label()
    
    def get_unique_labels(self):
        """Returns the unique labels in the dataset"""
        return self.data["Activity"].unique()

    def split_dataset_by_label(self):
        labelwise_datasets = {}
        for label in self.labels:
            labelwise_datasets[label] = self.data[self.data["Activity"] == label]
        return labelwise_datasets

    def get_permuted_dataset(self, permutation: list):
        reordered_data = []
        for label in permutation:
            if label in self.labelwise_datasets:
                reordered_data.append(self.labelwise_datasets[label])
        return pd.concat(reordered_data).reset_index(drop=True)

    def get_change_points(self):
        """Returns the change points in the dataset."""
        last_label = self.data["Activity"][0]
        cp = []
        for idx, row in self.data.iterrows():
            label = row["Activity"]
            if label != last_label:
                cp.append(idx)
                last_label = label
        return np.array(cp)

    def get_chunk_sizes(self):
        """Returns the sizes of the segments in the dataset."""
        cps = self.get_change_points()
        chunks = [cps[0]]
        for i, cp in enumerate(cps[0:-1]):
            chunks.append(cps[i+1]-cps[i])
        chunks.append(len(self.data) - cps[-1])
        return chunks

    def plot(self, start=0, end=-1):
        """Plots each field of the dataset."""
        data = self.data.iloc[start:end]
        data_fields = data.columns[1:-3]
        cps = self.get_change_points()
        for field in data_fields:
            ax = data[field].plot(linewidth=0.5)
            ax.set_title(field)
            plt.show()
            plt.close()


class Dataset:
    """
    Class for handling the dataset.

    Args:
        data_file (str): The path to the data file.
    """
    def __init__(self, data_file: str):
        self.data = self.load_data(data_file).reset_index(drop=False)
        self.subjects = self.split_by_subject()
    
    def get_available_labels(self):
        """
        Gets the available labels in the dataset.

        Returns:
            list: The available labels.
        """
        return self.data["Activity"].unique()

    def chunk_size_histogram(self):
        """Plots a histogram of the chunk sizes in the dataset."""
        all_chunks = []
        for i, subject in self.subjects.items():
            all_chunks = all_chunks + subject.get_chunk_sizes()
        counts, bins = np.histogram(all_chunks, bins=25, range=(0, 10000), density=True)
        plt.stairs(counts, bins)
        plt.savefig("chunk_hist" + ".png")
        plt.clf()

    def plot_all_edges(self):
        """
        Plots a histogram of edges for all subjects.
        
        Returns:
            list: The edges.
            list: The counts of the edges.
        """
        cp_col = self.data["ChangePoint"]
        cps = cp_col[cp_col].index.values
        missed_edges = []
        for cp in cps:
            missed_edges.append((self.data["Activity"][cp-1], self.data["Activity"][cp]))
        edges = list(set(missed_edges))
        counts = np.zeros(len(edges))
        edge_str = []

        edges.sort()

        for i, edge in enumerate(edges):
            edge_str.append(str(edge))
            for medge in missed_edges:
                if str(edge) == str(medge):
                    counts[i] = counts[i] + 1
        norm = [float(i)/sum(counts) for i in counts]
        plt.bar(edge_str, norm)
        plt.bar(edge_str, norm)
        plt.suptitle('Edges')
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(24.5, 5.5)
        plt.savefig("all_edges" + ".png")
        plt.clf()
        return edges, counts

    def split_by_subject(self):
        """
        Splits the dataset by subject.
        
        Returns:
            dict: A dictionary of datasets for each subject.
        """
        subject_dict = {}
        subjects = self.data["Subject"].unique()
        for subject in subjects:
            subject_data = self.data[self.data["Subject"] == subject]
            subject_dataset = SubjectDataset(subject_data, subject)
            subject_dict[subject] = subject_dataset
        return subject_dict

    def load_data(self, filename: str):
        """
        Loads the dataset from a pickle file.
        
        Args:
            filename (str): The path to the file.

        Returns:
            pd.DataFrame: The dataset
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def get_subject(self, subject: int, permutation: list=[], with_sl: bool=True):
        """
        Returns the dataset of a subject with the given permutation.

        Args:
            subject (int): The subject number.
            permutation (list): The permutation of the labels.
            with_sl (bool): If True, the dataset will include the subject and label columns.

        Returns:
            pd.DataFrame: The dataset of the subject.
        """
        
        if len(permutation) > 0:
            data = self.subjects[subject].get_permuted_dataset(permutation)
        else:
            data = self.subjects[subject].data
        if with_sl:
            return data
        else:
            return data.iloc[:, 2:-3]
        
    def plot_subject_data(self, subject: int, start: int=0, end: int=-1):
        """
        Plots the data of a subject.

        Args:
            subject (int): The subject number.
            start (int): The start index of the plot.
            end (int): The end index of the plot.
        """
        self.subjects[subject].plot(start=start, end=end)
        
