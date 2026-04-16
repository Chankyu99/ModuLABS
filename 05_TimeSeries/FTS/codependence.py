"""
mlfinlab
"""

import numpy as np
import pandas as pd
import scipy.stats as ss

from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import mutual_info_score
from scipy.stats import spearmanr


# pylint: disable=invalid-name


def angular_distance(x: np.array, y: np.array) -> float:
    """
    Returns angular distance between two vectors. Angular distance is a slight modification of Pearson correlation which
    satisfies metric conditions.
    Formula used for calculation:
    Ang_Distance = (1/2 * (1 - Corr))^(1/2)
    Read Cornell lecture notes for more information about angular distance:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes.
    :param x: (np.array/pd.Series) X vector.
    :param y: (np.array/pd.Series) Y vector.
    :return: (float) Angular distance.
    """

    corr_coef = np.corrcoef(x, y)[0][1]
    return np.sqrt(0.5 * (1 - corr_coef))


def absolute_angular_distance(x: np.array, y: np.array) -> float:
    """
    Returns absolute angular distance between two vectors. It is a modification of angular distance where the absolute
    value of the Pearson correlation coefficient is used.
    Formula used for calculation:
    Abs_Ang_Distance = (1/2 * (1 - abs(Corr)))^(1/2)
    Read Cornell lecture notes for more information about absolute angular distance:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes.
    :param x: (np.array/pd.Series) X vector.
    :param y: (np.array/pd.Series) Y vector.
    :return: (float) Absolute angular distance.
    """

    corr_coef = np.corrcoef(x, y)[0][1]
    return np.sqrt(0.5 * (1 - abs(corr_coef)))


def squared_angular_distance(x: np.array, y: np.array) -> float:
    """
    Returns squared angular distance between two vectors. It is a modification of angular distance where the square of
    Pearson correlation coefficient is used.
    Formula used for calculation:
    Squared_Ang_Distance = (1/2 * (1 - (Corr)^2))^(1/2)
    Read Cornell lecture notes for more information about squared angular distance:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes.
    :param x: (np.array/pd.Series) X vector.
    :param y: (np.array/pd.Series) Y vector.
    :return: (float) Squared angular distance.
    """

    corr_coef = np.corrcoef(x, y)[0][1]
    return np.sqrt(0.5 * (1 - corr_coef ** 2))


def distance_correlation(x: np.array, y: np.array) -> float:
    """
    Returns distance correlation between two vectors. Distance correlation captures both linear and non-linear
    dependencies.
    Formula used for calculation:
    Distance_Corr[X, Y] = dCov[X, Y] / (dCov[X, X] * dCov[Y, Y])^(1/2)
    dCov[X, Y] is the average Hadamard product of the doubly-centered Euclidean distance matrices of X, Y.
    Read Cornell lecture notes for more information about distance correlation:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes.
    :param x: (np.array/pd.Series) X vector.
    :param y: (np.array/pd.Series) Y vector.
    :return: (float) Distance correlation coefficient.
    """

    x = x[:, None]
    y = y[:, None]

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    a = squareform(pdist(x))
    b = squareform(pdist(y))

    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    d_cov_xx = (A * A).sum() / (x.shape[0] ** 2)
    d_cov_xy = (A * B).sum() / (x.shape[0] ** 2)
    d_cov_yy = (B * B).sum() / (x.shape[0] ** 2)

    coef = np.sqrt(d_cov_xy) / np.sqrt(np.sqrt(d_cov_xx) * np.sqrt(d_cov_yy))

    return coef

"""
Implementation of distance using the Generic Non-Parametric Representation approach from "Some contributions to the
clustering of financial time series and applications to credit default swaps" by Gautier Marti
https://www.researchgate.net/publication/322714557
"""

# pylint: disable=invalid-name

def spearmans_rho(x: np.array, y: np.array) -> float:
    """
    Calculates a statistical estimate of Spearman's rho - a copula-based dependence measure.
    Formula for calculation:
    rho = 1 - (6)/(T*(T^2-1)) * Sum((X_t-Y_t)^2)
    It is more robust to noise and can be defined if the variables have an infinite second moment.
    This statistic is described in more detail in the work by Gautier Marti
    https://www.researchgate.net/publication/322714557 (p.54)
    This method is a wrapper for the scipy spearmanr function. For more details about the function and its parameters,
    please visit scipy documentation
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.spearmanr.html
    :param x: (np.array/pd.Series) X vector
    :param y: (np.array/pd.Series) Y vector (same number of observations as X)
    :return: (float) Spearman's rho statistical estimate
    """

    # Coefficient calculationS
    rho, _ = spearmanr(x, y)

    return rho

def gpr_distance(x: np.array, y: np.array, theta: float) -> float:
    """
    Calculates the distance between two Gaussians under the Generic Parametric Representation (GPR) approach.
    According to the original work https://www.researchgate.net/publication/322714557 (p.70):
    "This is a fast and good proxy for distance d_theta when the first two moments ... predominate". But it's not
    a good metric for heavy-tailed distributions.
    Parameter theta defines what type of information dependency is being tested:
    - for theta = 0 the distribution information is tested
    - for theta = 1 the dependence information is tested
    - for theta = 0.5 a mix of both information types is tested
    With theta in [0, 1] the distance lies in range [0, 1] and is a metric. (See original work for proof, p.71)
    :param x: (np.array/pd.Series) X vector.
    :param y: (np.array/pd.Series) Y vector (same number of observations as X).
    :param theta: (float) Type of information being tested. Falls in range [0, 1].
    :return: (float) Distance under GPR approach.
    """

    # Calculating the GPR distance
    distance = theta * (1 - spearmans_rho(x, y)) / 2 + \
               (1 - theta) * (1 - ((2 * x.std() * y.std()) / (x.std()**2 + y.std()**2))**(1/2) *
                              np.exp(- (1 / 4) * (x.mean() - y.mean())**2 / (x.std()**2 + y.std()**2)))

    return distance**(1/2)

def gnpr_distance(x: np.array, y: np.array, theta: float, bandwidth: float = 0.01) -> float:
    """
    Calculates the empirical distance between two random variables under the Generic Non-Parametric Representation
    (GNPR) approach.
    Formula for the distance is taken from https://www.researchgate.net/publication/322714557 (p.72).
    Parameter theta defines what type of information dependency is being tested:
    - for theta = 0 the distribution information is tested
    - for theta = 1 the dependence information is tested
    - for theta = 0.5 a mix of both information types is tested
    With theta in [0, 1] the distance lies in the range [0, 1] and is a metric. (See original work for proof, p.71)
    :param x: (np.array/pd.Series) X vector.
    :param y: (np.array/pd.Series) Y vector (same number of observations as X).
    :param theta: (float) Type of information being tested. Falls in range [0, 1].
    :param bandwidth: (float) Bandwidth to use for splitting the X and Y vector observations. (0.01 by default)
    :return: (float) Distance under GNPR approach.
    """

    # Number of observations
    num_obs = x.shape[0]

    # Calculating the d_1 distance
    dist_1 = 3 / (num_obs * (num_obs**2 - 1)) * (np.power(x - y, 2).sum())

    # Creating the proper bins
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())

    # Creating a grid and histograms
    bins = np.arange(min_val, max_val + bandwidth, bandwidth)
    hist_x = np.histogram(x, bins)[0] / num_obs
    hist_y = np.histogram(y, bins)[0] / num_obs

    # Calculating the d_0 distance
    dist_0 = np.power(hist_x**(1/2) - hist_y**(1/2), 2).sum() / 2

    # Calculating the GNPR distance
    distance = theta * dist_1 + (1 - theta) * dist_0

    return distance**(1/2)

"""
Implementations of mutual information (I) and variation of information (VI) codependence measures from Cornell
lecture slides: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes
"""



# pylint: disable=invalid-name

def get_optimal_number_of_bins(num_obs: int, corr_coef: float = None) -> int:
    """
    Calculates optimal number of bins for discretization based on number of observations
    and correlation coefficient (univariate case).
    Algorithms used in this function were originally proposed in the works of Hacine-Gharbi et al. (2012)
    and Hacine-Gharbi and Ravier (2018). They are described in the Cornell lecture notes:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes (p.26)
    :param num_obs: (int) Number of observations.
    :param corr_coef: (float) Correlation coefficient, used to estimate the number of bins for univariate case.
    :return: (int) Optimal number of bins.
    """

    # Univariate case
    if corr_coef is None or abs(corr_coef - 1) <= 1e-4:
        z = (8 + 324 * num_obs + 12 * (36 * num_obs + 729 * num_obs ** 2) ** .5) ** (1 / 3.)
        bins = round(z / 6. + 2. / (3 * z) + 1. / 3)

    # Bivariate case
    else:
        bins = round(2 ** -.5 * (1 + (1 + 24 * num_obs / (1. - corr_coef ** 2)) ** .5) ** .5)
    return int(bins)


def get_mutual_info(x: np.array, y: np.array, n_bins: int = None, normalize: bool = False) -> float:
    """
    Returns mutual information (I) between two vectors.
    This function uses the discretization with the optimal bins algorithm proposed in the works of
    Hacine-Gharbi et al. (2012) and Hacine-Gharbi and Ravier (2018).
    Read Cornell lecture notes for more information about the mutual information:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes.
    :param x: (np.array) X vector.
    :param y: (np.array) Y vector.
    :param n_bins: (int) Number of bins for discretization, if None the optimal number will be calculated.
                         (None by default)
    :param normalize: (bool) Flag used to normalize the result to [0, 1]. (False by default)
    :return: (float) Mutual information score.
    """

    if n_bins is None:
        corr_coef = np.corrcoef(x, y)[0][1]
        n_bins = get_optimal_number_of_bins(x.shape[0], corr_coef=corr_coef)

    contingency = np.histogram2d(x, y, n_bins)[0]
    mutual_info = mutual_info_score(None, None, contingency=contingency)  # Mutual information
    if normalize is True:
        marginal_x = ss.entropy(np.histogram(x, n_bins)[0])  # Marginal for x
        marginal_y = ss.entropy(np.histogram(y, n_bins)[0])  # Marginal for y
        mutual_info /= min(marginal_x, marginal_y)
    return mutual_info


def variation_of_information_score(x: np.array, y: np.array, n_bins: int = None, normalize: bool = False) -> float:
    """
    Returns variantion of information (VI) between two vectors.
    This function uses the discretization using optimal bins algorithm proposed in the works of
    Hacine-Gharbi et al. (2012) and Hacine-Gharbi and Ravier (2018).
    Read Cornell lecture notes for more information about the variation of information:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes.
    :param x: (np.array) X vector.
    :param y: (np.array) Y vector.
    :param n_bins: (int) Number of bins for discretization, if None the optimal number will be calculated.
                         (None by default)
    :param normalize: (bool) True to normalize the result to [0, 1]. (False by default)
    :return: (float) Variation of information score.
    """

    if n_bins is None:
        corr_coef = np.corrcoef(x, y)[0][1]
        n_bins = get_optimal_number_of_bins(x.shape[0], corr_coef=corr_coef)

    contingency = np.histogram2d(x, y, n_bins)[0]
    mutual_info = mutual_info_score(None, None, contingency=contingency)  # Mutual information
    marginal_x = ss.entropy(np.histogram(x, n_bins)[0])  # Marginal for x
    marginal_y = ss.entropy(np.histogram(y, n_bins)[0])  # Marginal for y
    score = marginal_x + marginal_y - 2 * mutual_info  # Variation of information

    if normalize is True:
        joint_dist = marginal_x + marginal_y - mutual_info  # Joint distribution
        score /= joint_dist

    return score


# pylint: disable=invalid-name

def get_dependence_matrix(df: pd.DataFrame, dependence_method: str, theta: float = 0.5,
                          bandwidth: float = 0.01) -> pd.DataFrame:
    """
    This function returns a dependence matrix for elements given in the dataframe using the chosen dependence method.
    List of supported algorithms to use for generating the dependence matrix: ``information_variation``,
    ``mutual_information``, ``distance_correlation``, ``spearmans_rho``, ``gpr_distance``, ``gnpr_distance``.
    :param df: (pd.DataFrame) Features.
    :param dependence_method: (str) Algorithm to be use for generating dependence_matrix.
    :param theta: (float) Type of information being tested in the GPR and GNPR distances. Falls in range [0, 1].
                          (0.5 by default)
    :param bandwidth: (float) Bandwidth to use for splitting observations in the GPR and GNPR distances. (0.01 by default)
    :return: (pd.DataFrame) Dependence matrix.
    """
    # Get the feature names.
    features_cols = df.columns.values
    n = df.shape[1]
    np_df = df.values.T  # Make columnar access, but for np.array

    # Defining the dependence function.
    if dependence_method == 'information_variation':
        dep_function = lambda x, y: variation_of_information_score(x, y, normalize=True)
    elif dependence_method == 'mutual_information':
        dep_function = lambda x, y: get_mutual_info(x, y, normalize=True)
    elif dependence_method == 'distance_correlation':
        dep_function = distance_correlation
    elif dependence_method == 'spearmans_rho':
        dep_function = spearmans_rho
    elif dependence_method == 'gpr_distance':
        dep_function = lambda x, y: gpr_distance(x, y, theta=theta)
    elif dependence_method == 'gnpr_distance':
        dep_function = lambda x, y: gnpr_distance(x, y, theta=theta, bandwidth=bandwidth)
    else:
        raise ValueError(f"{dependence_method} is not a valid method. Please use one of the supported methods \
                            listed in the docsting.")

    # Generating the dependence_matrix
    dependence_matrix = np.array([
        [
            dep_function(np_df[i], np_df[j]) if j < i else
            # Leave diagonal elements as 0.5 to later double them to 1
            0.5 * dep_function(np_df[i], np_df[j]) if j == i else
            0  # Make upper triangle 0 to fill it later on
            for j in range(n)
        ]
        for i in range(n)
    ])

    # Make matrix symmetrical
    dependence_matrix = dependence_matrix + dependence_matrix.T

    #  Dependence_matrix converted into a DataFrame.
    dependence_df = pd.DataFrame(data=dependence_matrix, index=features_cols, columns=features_cols)

    if dependence_method == 'information_variation':
        return 1 - dependence_df  # IV is reverse, 1 - independent, 0 - similar

    return dependence_df

def get_distance_matrix(X: pd.DataFrame, distance_metric: str = 'angular') -> pd.DataFrame:
    """
    Applies distance operator to a dependence matrix.
    This allows to turn a correlation matrix into a distance matrix. Distances used are true metrics.
    List of supported distance metrics to use for generating the distance matrix: ``angular``, ``squared_angular``,
    and ``absolute_angular``.
    :param X: (pd.DataFrame) Dataframe to which distance operator to be applied.
    :param distance_metric: (str) The distance metric to be used for generating the distance matrix.
    :return: (pd.DataFrame) Distance matrix.
    """
    if distance_metric == 'angular':
        distfun = lambda x: ((1 - x).round(5) / 2.) ** .5
    elif distance_metric == 'abs_angular':
        distfun = lambda x: ((1 - abs(x)).round(5) / 2.) ** .5
    elif distance_metric == 'squared_angular':
        distfun = lambda x: ((1 - x ** 2).round(5) / 2.) ** .5
    else:
        raise ValueError(f'{distance_metric} is a unknown distance metric. Please use one of the supported methods \
                            listed in the docsting.')

    return distfun(X).fillna(0)