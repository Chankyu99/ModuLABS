"""
mlfinlab
"""

#Imports
import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from statsmodels.regression.linear_model import OLS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from libs.feature_engineering.codependence import get_dependence_matrix, get_distance_matrix
from typing import Union


def _improve_clusters(corr_mat: pd.DataFrame, clusters: dict, top_clusters: dict) -> Union[
        pd.DataFrame, dict, pd.Series]:
    """
    Improve number clusters using silh scores
    :param corr_mat: (pd.DataFrame) Correlation matrix
    :param clusters: (dict) Clusters elements
    :param top_clusters: (dict) Improved clusters elements
    :return: (tuple) [ordered correlation matrix, clusters, silh scores]
    """
    clusters_new, new_idx = {}, []
    for i in clusters.keys():
        clusters_new[len(clusters_new.keys())] = list(clusters[i])

    for i in top_clusters.keys():
        clusters_new[len(clusters_new.keys())] = list(top_clusters[i])

    map(new_idx.extend, clusters_new.values())
    corr_new = corr_mat.loc[new_idx, new_idx]

    dist = ((1 - corr_mat.fillna(0)) / 2.0) ** 0.5

    kmeans_labels = np.zeros(len(dist.columns))
    for i in clusters_new:
        idxs = [dist.index.get_loc(k) for k in clusters_new[i]]
        kmeans_labels[idxs] = i

    silh_scores_new = pd.Series(silhouette_samples(dist, kmeans_labels), index=dist.index)
    return corr_new, clusters_new, silh_scores_new


def _cluster_kmeans_base(corr_mat: pd.DataFrame, max_num_clusters: int = 10, repeat: int = 10) -> Union[
        pd.DataFrame, dict, pd.Series]:
    """
    Initial clustering step using KMeans.
    :param corr_mat: (pd.DataFrame) Correlation matrix
    :param max_num_clusters: (int) Maximum number of clusters to search for.
    :param repeat: (int) Number of clustering algorithm repetitions.
    :return: (tuple) [ordered correlation matrix, clusters, silh scores]
    """

    # Distance matrix
    distance = ((1 - corr_mat.fillna(0)) / 2.0) ** 0.5
    silh = pd.Series(dtype='float64')

    # Get optimal num clusters
    for _ in range(repeat):
        for num_clusters in range(2, max_num_clusters + 1):
            kmeans_ = KMeans(n_clusters=num_clusters, n_init=1)
            kmeans_ = kmeans_.fit(distance)
            silh_ = silhouette_samples(distance, kmeans_.labels_)
            stat = (silh_.mean() / silh_.std(), silh.mean() / silh.std())

            if np.isnan(stat[1]) or stat[0] > stat[1]:
                silh = silh_
                kmeans = kmeans_

    # Number of clusters equals to length(kmeans labels)
    new_idx = np.argsort(kmeans.labels_)

    # Reorder rows
    corr1 = corr_mat.iloc[new_idx]
    # Reorder columns
    corr1 = corr1.iloc[:, new_idx]

    # Cluster members
    clusters = {i: corr_mat.columns[np.where(kmeans.labels_ == i)[0]].tolist() for i in
                np.unique(kmeans.labels_)}
    silh = pd.Series(silh, index=distance.index)

    return corr1, clusters, silh


def _check_improve_clusters(new_tstat_mean: float, mean_redo_tstat: float, old_cluster: tuple,
                            new_cluster: tuple) -> tuple:
    """
    Checks cluster improvement condition based on t-statistic.
    :param new_tstat_mean: (float) T-statistics
    :param mean_redo_tstat: (float) Average t-statistcs for cluster improvement
    :param old_cluster: (tuple) Old cluster correlation matrix, optimized clusters, silh scores
    :param new_cluster: (tuple) New cluster correlation matrix, optimized clusters, silh scores
    :return: (tuple) Cluster
    """

    if new_tstat_mean > mean_redo_tstat:
        return old_cluster
    return new_cluster


def cluster_kmeans_top(corr_mat: pd.DataFrame, repeat: int = 10) -> Union[pd.DataFrame, dict, pd.Series, bool]:
    """
    Improve the initial clustering by leaving clusters with high scores unchanged and modifying clusters with
    below average scores.
    :param corr_mat: (pd.DataFrame) Correlation matrix
    :param repeat: (int) Number of clustering algorithm repetitions.
    :return: (tuple) [correlation matrix, optimized clusters, silh scores, boolean to rerun ONC]
    """
    # pylint: disable=no-else-return

    max_num_clusters = min(corr_mat.drop_duplicates().shape[0], corr_mat.drop_duplicates().shape[1]) - 1
    corr1, clusters, silh = _cluster_kmeans_base(corr_mat, max_num_clusters=max_num_clusters, repeat=repeat)

    # Get cluster quality scores
    cluster_quality = {i: float('Inf') if np.std(silh[clusters[i]]) == 0 else np.mean(silh[clusters[i]]) /
                          np.std(silh[clusters[i]]) for i in clusters.keys()}
    avg_quality = np.mean(list(cluster_quality.values()))
    redo_clusters = [i for i in cluster_quality.keys() if cluster_quality[i] < avg_quality]

    if len(redo_clusters) <= 2:
        # If 2 or less clusters have a quality rating less than the average then stop.
        return corr1, clusters, silh
    else:
        keys_redo = []
        for i in redo_clusters:
            keys_redo.extend(clusters[i])

        corr_tmp = corr_mat.loc[keys_redo, keys_redo]
        mean_redo_tstat = np.mean([cluster_quality[i] for i in redo_clusters])
        _, top_clusters, _ = cluster_kmeans_top(corr_tmp, repeat=repeat)

        # Make new clusters (improved)
        corr_new, clusters_new, silh_new = _improve_clusters(corr_mat,
                                                             {i: clusters[i] for i in clusters.keys() if
                                                              i not in redo_clusters},
                                                             top_clusters)
        new_tstat_mean = np.mean(
            [np.mean(silh_new[clusters_new[i]]) / np.std(silh_new[clusters_new[i]]) for i in clusters_new])

        return _check_improve_clusters(new_tstat_mean, mean_redo_tstat, (corr1, clusters, silh),
                                       (corr_new, clusters_new, silh_new))


def get_onc_clusters(corr_mat: pd.DataFrame, repeat: int = 10) -> Union[pd.DataFrame, dict, pd.Series]:
    """
    Optimal Number of Clusters (ONC) algorithm described in the following paper:
    `Marcos Lopez de Prado, Michael J. Lewis, Detection of False Investment Strategies Using Unsupervised
    Learning Methods, 2015 <https://papers.ssrn.com/sol3/abstract_id=3167017>`_;
    The code is based on the code provided by the authors of the paper.
    The algorithm searches for the optimal number of clusters using the correlation matrix of elements as an input.
    The correlation matrix is transformed to a matrix of distances, the K-Means algorithm is applied multiple times
    with a different number of clusters to use. The results are evaluated on the t-statistics of the silhouette scores.
    The output of the algorithm is the reordered correlation matrix (clustered elements are placed close to each other),
    optimal clustering, and silhouette scores.
    :param corr_mat: (pd.DataFrame) Correlation matrix of features
    :param repeat: (int) Number of clustering algorithm repetitions
    :return: (tuple) [correlation matrix, optimized clusters, silh scores]
    """

    return cluster_kmeans_top(corr_mat, repeat)


# pylint: disable=invalid-name
def get_feature_clusters(X: pd.DataFrame, dependence_metric: str, distance_metric: str = None,
                         linkage_method: str = None, n_clusters: int = None, critical_threshold: float = 0.0) -> list:
    """
    Machine Learning for Asset Managers
    Snippet 6.5.2.1 , page 85. Step 1: Features Clustering
    Gets clustered features subsets from the given set of features.
    :param X: (pd.DataFrame) Dataframe of features.
    :param dependence_metric: (str) Method to be use for generating dependence_matrix, either 'linear' or
                              'information_variation' or 'mutual_information' or 'distance_correlation'.
    :param distance_metric: (str) The distance operator to be used for generating the distance matrix. The methods that
                            can be applied are: 'angular', 'squared_angular', 'absolute_angular'. Set it to None if the
                            feature are to be generated as it is by the ONC algorithm.
    :param linkage_method: (str) Method of linkage to be used for clustering. Methods include: 'single', 'ward',
                           'complete', 'average', 'weighted', and 'centroid'. Set it to None if the feature are to
                           be generated as it is by the ONC algorithm.
    :param n_clusters: (int) Number of clusters to form. Must be less the total number of features. If None then it
                       returns optimal number of clusters decided by the ONC Algorithm.
    :param critical_threshold: (float) Threshold for determining low silhouette score in the dataset. It can any real number
                                in [-1,+1], default is 0 which means any feature that has a silhouette score below 0 will be
                                indentified as having low silhouette and hence requied transformation will be appiled to for
                                for correction of the same.
    :return: (list) Feature subsets.
    """
    # Checking if dataset contains features low silhouette
    X = _check_for_low_silhouette_scores(X, critical_threshold)

    # Get the dependence matrix
    if dependence_metric != 'linear':
        dep_matrix = get_dependence_matrix(X, dependence_method=dependence_metric)
    else:
        dep_matrix = X.corr()

    if n_clusters is None and (distance_metric is None or linkage_method is None):
        return list(get_onc_clusters(dep_matrix.fillna(0))[1].values())  # Get optimal number of clusters
    if distance_metric is not None and (linkage_method is not None and n_clusters is None):
        n_clusters = len(get_onc_clusters(dep_matrix.fillna(0))[1])
    if n_clusters >= len(X.columns):  # Check if number of clusters exceeds number of features
        raise ValueError('Number of clusters must be less than the number of features')

    # Apply distance operator on the dependence matrix
    dist_matrix = get_distance_matrix(dep_matrix, distance_metric=distance_metric)

    # Get the linkage
    link = linkage(squareform(dist_matrix), method=linkage_method)
    clusters = fcluster(link, t=n_clusters, criterion='maxclust')
    clustered_subsets = [[f for c, f in zip(clusters, X.columns) if c == ci] for ci in range(1, n_clusters + 1)]

    return clustered_subsets


def _cluster_transformation(X: pd.DataFrame, clusters: dict, feats_to_transform: list) -> pd.DataFrame:
    """
    Machine Learning for Asset Managers
    Snippet 6.5.2.1 , page 85. Step 1: Features Clustering (last paragraph)
    Transforms a dataset to reduce the multicollinearity of the system by replacing the original feature with
    the residual from regression.
    :param X: (pd.DataFrame) Dataframe of features.
    :param clusters: (dict) Clusters generated by ONC algorithm.
    :param feats_to_transform: (list) Features that have low silhouette score and to be transformed.
    :return: (pd.DataFrame) Transformed features.
    """
    for feat in feats_to_transform:
        for i, j in clusters.items():

            if feat in j:  # Selecting the cluster that contains the feature
                exog = sm.add_constant(X.drop(j, axis=1)).values
                endog = X[feat].values
                ols = OLS(endog, exog).fit()

                if ols.df_model < (exog.shape[1]-1):
                    # Degree of freedom is low
                    new_exog = _combine_features(X, clusters, i)
                    # Run the regression again on the new exog
                    ols = OLS(endog, new_exog.reshape(exog.shape[0], -1)).fit()
                    X[feat] = ols.resid
                else:
                    X[feat] = ols.resid

    return X


def _combine_features(X, clusters, exclude_key) -> np.array:
    """
    Combines features of each cluster linearly by following a minimum variance weighting scheme.
    The Minimum Variance weights are calculated without constraints, other than the weights sum to one.
    :param X: (pd.DataFrame) Dataframe of features.
    :param clusters: (dict) Clusters generated by ONC algorithm.
    :param exclude_key: (int) Key of the cluster which is to be excluded.
    :return: (np.array) Combined features for each cluster.
    """

    new_exog = []
    for i, cluster in clusters.items():

        if i != exclude_key:
            subset = X[cluster]
            cov_matx = subset.cov()  # Covariance matrix of the cluster
            eye_vec = np.array(cov_matx.shape[1]*[1], float)
            try:
                numerator = np.dot(np.linalg.inv(cov_matx), eye_vec)
                denominator = np.dot(eye_vec, numerator)
                # Minimum variance weighting
                wghts = numerator/denominator
            except np.linalg.LinAlgError:
                # A singular matrix so giving each component equal weight
                wghts = np.ones(subset.shape[1]) * (1/subset.shape[1])
            new_exog.append(((subset*wghts).sum(1)).values)

    return np.array(new_exog)


def _check_for_low_silhouette_scores(X: pd.DataFrame, critical_threshold: float = 0.0) -> pd.DataFrame:
    """
    Machine Learning for Asset Managers
    Snippet 6.5.2.1 , page 85. Step 1: Features Clustering (last paragraph)
    Checks where the dataset contains features low silhouette due one feature being a combination of
    multiple features across clusters. This is a problem, because ONC cannot assign one feature to multiple
    clusters and it needs a transformation.
    :param X: (pd.DataFrame) Dataframe of features.
    :param critical_threshold: (float) Threshold for determining low silhouette score.
    :return: (pd.DataFrame) Dataframe of features.
    """
    _, clstrs, silh = get_onc_clusters(X.corr())
    low_silh_feat = silh[silh < critical_threshold].index
    if len(low_silh_feat) > 0:
        print(f'{len(low_silh_feat)} feature/s found with low silhouette score {low_silh_feat}. Returning the transformed dataset')

        # Returning the transformed dataset
        return _cluster_transformation(X, clstrs, low_silh_feat)

    print('No feature/s found with low silhouette score. All features belongs to its respective clusters')

    return X