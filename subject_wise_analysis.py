import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from multivariate_claspy import MultivariateClaspy
from ruptures_segmentation import RupturesCpDetector
from changeforest_segmentation import ChangeForestCpDetector
from dataset import Dataset
import sys
np.random.seed(4)

def do_ruptures_analysis(data, permutation: np.array, search: str="kernel", model: str="linear", min_size: int=400, pen: int=1000):
    """
    Performs the ruptures analysis on the given data and permutation.
    
    Args:
        data: The data to be analyzed. 
        permutation: The permutation of the labels.
        search: The search method to be used.
        model: The model to be used.
        min_size: The minimum size of the segments.
        pen: The penalty to be used.

    Returns:
        The f1 score, the missed edges, and the change points.
    """
    ruptures = RupturesCpDetector(data, search=search, model=model, pen=pen, min_size=min_size)
    ruptures.fit_all()
    cs = np.array(ruptures.get_chunk_sizes())
    score = ruptures.get_f1score(margin=50)
    missed_edges = get_missed_edges(permutation, ruptures)
    cps = ruptures.get_cp_indices()
    return score, missed_edges, cps

def do_changeforest_analysis(data: pd.DataFrame, permutation: np.array, search: str="bs",
                             model: str="random_forest", min_size: int=400, gain: float=1e7):
    """
    Performs the change forest analysis on the given data and permutation.

    Args:
        data: The data to be analyzed.
        permutation: The permutation of the labels.
        search: The search method to be used.
        model: The model to be used.
        min_size: The minimum size of the segments.

    Returns:
        The f1 score, the missed edges, and the change points.
    """
    cf = ChangeForestCpDetector(data, method=model, search=search, min_size=min_size, gain=gain)
    cf.fit_all()
    score = cf.get_f1score(margin=50)
    missed_edges = get_missed_edges(permutation, cf)
    cps = cf.get_cp_indices()
    return score, missed_edges, cps

def do_claspy_analysis(subject_data: pd.DataFrame, permutation: np.array, cfg: str, margin: int):
    """
    Performs the multivariate claspy analysis on the given data and permutation.

    Args:
        subject_data: The data to be analyzed.
        permutation: The permutation of the labels.
        cfg: The configuration file.
        margin: The margin to be used.
    
    Returns:
        The f1 score, the missed edges, and the change points. 
    """
    mv_claspy = MultivariateClaspy(subject_data, cfg)
    mv_claspy.fit_all()
    mv_claspy.kernel_cp_combine(kernel_size=101, peak_height=3, plot=True, peak_distance=400)
    score = mv_claspy.get_f1score(margin=margin)
    
    missed_edges = get_missed_edges(permutation, mv_claspy)
    cps = mv_claspy.get_cp_indices()
    return score, missed_edges, cps

def single_perm_subjectwise_analysis(data: pd.DataFrame, permutation: np.array, cfg: str=None,
                                     margin=125, method: str="claspy", outfile: str="", search: str="kernel",
                                     model: str="linear", pen: int=1000, min_size: int=400):
    """
    Performs the subject-wise analysis on the given data and permutation.

    Args:
        data: The data to be analyzed.
        permutation: The permutation of the labels.
        cfg: The configuration file.
        margin: The margin to be used.
        method: The method to be used.
        outfile: The output file.
        search: The search method to be used.
        model: The model to be used.
        pen: The penalty to be used.
        min_size: The minimum size of the segments.

    Returns:
        The dataframe containing the results of the analysis.
    """
    rows = []
    all_missed_edges = []
    columns = ["permutation", "subject", "f1_score", "precision",
               "recall", "missed_edges", "predicted_relative_index",
               "predicted_total_index", "predicted_time_index",
               "gt_relative_index", "gt_total_index", "gt_time_index"]

    edges, counts = data.plot_all_edges()
    data.chunk_size_histogram()
    cps = []
    for subject in list(data.subjects.keys()):
        subject_data = data.get_subject(subject)
        if method == "ruptures":
            score, missed_edges, cp_idxs = do_ruptures_analysis(subject_data, permutation, search=search, model=model, pen=pen, min_size=min_size)
        elif method == "claspy":
            score, missed_edges, cp_idxs = do_claspy_analysis(subject_data, permutation, cfg, margin)
        elif method == "change_forest":
            score, missed_edges, cp_idxs = do_changeforest_analysis(subject_data, permutation, search=search, model=model, min_size=min_size, gain=pen)
        else:
            raise ValueError("Invalid method")
        
        print("Subject: {}, F1 Score: {}, Precision: {}, Recall: {}".format(subject, score[0], score[1], score[2]))
        sys.stdout.flush()
        cps.append(cp_idxs)
        all_missed_edges.extend(missed_edges)
        row = [""] + [subject] + list(score) + [missed_edges] + cp_idxs
        rows.append(row)

    plot_missed_edges(set(all_missed_edges), all_missed_edges, permutation, outfile)
    df = pd.DataFrame(rows, columns=columns)
    mean = df.mean()
    mean_row = ["", "mean", mean["f1_score"], mean["precision"], mean["recall"], "", "", "", "", "", "", ""]
    mean_df = pd.DataFrame([mean_row], columns=columns)
    df = pd.concat([df, mean_df])
    return df

def get_edges(permutation: np.array):
    """ Returns the edges of the permutation. """
    edges = []
    for i in range(len(permutation) - 1):
        edges.append((permutation[i], permutation[i+1]))
    return edges

def get_missed_edges(permutation: np.array, model: object):
    """ Returns the missed edges of the model. """
    edges = get_edges(permutation)
    _, idxs, missed_edges = model.get_missed_change_points(margin=125)
    return missed_edges

def plot_missed_edges(edges: list, missed_edges: list, permutation: np.array, outfile: str):
    """
    Plots the missed edges.

    Args:
        edges: The edges.
        missed_edges: The missed edges.
        permutation: The permutation.
        outfile: The output file.
    """
    counts = np.zeros(len(edges))
    edge_str = []
    edges = list(edges)
    edges.sort()
    for i, edge in enumerate(edges):
        edge_str.append(str(edge))
        for medge in missed_edges:
            if str(edge) == str(medge):
                counts[i] = counts[i] + 1
    norm = [float(i)/sum(counts) for i in counts]
    plt.bar(edge_str, norm)
    plt.suptitle('Missed Edges')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(24.5, 5.5)
    plt.savefig(outfile + "_missed_edges" + ".png")
    plt.clf()

def do_subjectwise_analysis(fname: str, cfg: str, num_permutations: int, outfile: str,
                            method: str="change_forest", search: str="bs",
                            model: str="change_in_mean", pen: float=10000000, min_size: int=400):
    
    """
    Performs the subject-wise change point detection on the given data.

    Args:
        fname: The file name of the data.
        cfg: The configuration file.
        num_permutations: The number of permutations.
        outfile: The output file.
        method: The method to be used.
        search: The search method to be used.
        model: The model to be used.
        pen: The penalty to be used.
        min_size: The minimum size of the segments
    """ 
    data = Dataset(fname)
    permutation = data.get_available_labels()
    all_dfs = []
    df = single_perm_subjectwise_analysis(data, [], cfg, method=method, outfile=outfile, search=search, model=model, pen=pen, min_size=min_size)
    all_dfs.append(df)
    for i in range(num_permutations):
        np.random.shuffle(permutation)
        df = single_perm_subjectwise_analysis(data, permutation, cfg, method=method, outfile=outfile, search=search, model=model, pen=pen, min_size=min_size)
        all_dfs.append(df)
        df.to_pickle(outfile + ".pkl")
    all_dfs = pd.concat(all_dfs)
    all_dfs.to_pickle(outfile + ".pkl")
    all_dfs.to_csv(outfile + ".csv")


if __name__ == "__main__":
    fname = "../Pre_processed_OutSense.pkl"
    cfg = "./config/config.yaml"

    do_subjectwise_analysis(fname, cfg, 0, "claspy", method="claspy")