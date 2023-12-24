import numpy as np
import scipy.stats
import scipy.optimize
import functools
from numba import jit


# ClopperPearsonCI
#
# The function implements the Clopper-Pearson confidence intervals (CIs)
# for the probability of success of a Binomially-distributed count.
#
# SYNTAX
#  lower, upper = ClopperPearsonCI(k, N, p)
#
# INPUTS
# k: integer in {0,..., N}; the number of observed successes in N i.i.d. trials. 
# N: integer >= 1; the number of i.i.d. trials.
# p: float in (0,1]; nominal coverage probability of the CI.
#
# OUTPUTS
# lower: float in [0,1); lower CI endpoint.
# upper: float in (0,1]; upper CI endpoint.
#
# NOTES
# 1) The generated CIs are, in general, conservative, i.e., the actual 
#    coverage probability is larger than p. Hence, they are not of 
#    smallest-width CIs for a given value of p.
# 2) No validation of inputs is performed.
#
# DEPENDENCIES
#  import scipy.stats
# 
# AUTHOR
#  Georgios C. Anagnostopoulos, September 2020
#
def ClopperPearsonCI(k, N, p):
    alpha = 1.0 - p
    if k == 0:
        lower = 0.0
        upper = 1.0 - (alpha / 2.0)**(1.0 / N)
    elif k == N:
        lower = (alpha / 2.0)**(1.0 / N)
        upper = 1.0
    else:
        lower = scipy.stats.beta.ppf(alpha/2.0, k, N-k+1)
        upper = scipy.stats.beta.ppf(1.0 - alpha/2.0, k+1, N-k)
    return lower, upper


# ExactMcNemarsTest
#
# Implements the p-value (observed significance level) of the exact version of 
# McNemar's Test for comparing two binomial proportions in terms of matched
# pairs.
#
# SYNTAX
# pvalue, idxBest = ExactMcNemarsTest(MatchedPairs)
#
# INPUTS
# MatchedPairs: 2D numpy.ndarray of N>=1 rows and 2 columns that only takes 
#               values 0,1 or True, False. Each row contains the classification 
#               outcome (correct/incorrect prediction) of 2 classification 
#               models for a specific sample.
#
# OUTPUTS
# pvalue:  float in [0,1]; p-value (observed significance level) of the test.
# idxBest: integer in {0,1}; indicates the classification model (column index
#          of MatchedPairs) with the most number of 1/True's between the two.
#          The test's alternative hypothesis is that the model with index
#          idxBest is better performing than the other model.
#
# NOTES
# 1) Beware: no validity checking is performed on the entries of MatchedPairs.
# 2) Assume that MatchedPairs[:,0] and MatchedPairs[:,1] are paired, i.i.d 
#    samples generated from model0 and model1 respectively. The null hypothesis
#    of the test is that the probabilities of observing a discordant pair (1,0)
#    or (0,1) are equal, i.e., the two models are indistingushable is terms
#    of clasisfication performance.
#    Without loss of generality, assume that the number of 
#    1/True's is higher for model0. Then, the alternative hypothesis states
#    that the probability of observing (1,0) pairs is higher than observing
#    (0,1) pairs.
#    An alpha-level significance test would reject the null hypothesis if
#    p-value <= alpha and would conclude that model0 is preferable to model1.
# 3) If both columns of MatchedPairs are identical, idxBest defaults to 0.
# 4) If the user selects a significance level alpha, she would reject the
#    null hypothesis, if pvalue <= alpha, and would conclude that the model
#    with index idxBest is the best performing of the two.
#
# DEPENDENCIES
#  import numpy as np
#  import scipy.stats
# 
# AUTHOR
#  Georgios C. Anagnostopoulos, June 2020
#
def ExactMcNemarsTest(MatchedPairs):
    n0 = sum((MatchedPairs[:,0] == 1) & (MatchedPairs[:,1] == 0))
    n1 = sum((MatchedPairs[:,0] == 0) & (MatchedPairs[:,1] == 1))
    w = max(n0, n1)
    Nw = n0 + n1
    idxBest = 0 if n0 >= n1 else 1
    pvalue = scipy.stats.binom.sf(w-1, Nw, 0.5)
    return pvalue, idxBest


@jit(nopython=True)
def _logBinomialCDF(n, N, p):
    if n < 0:
        logCDF = np.NINF
    elif n >= N:
        logCDF = 0.0
    else:
        if p == 0.0:
            logCDF = 0.0
        elif p == 1.0:
            logCDF = np.NINF
        else:
            # At this point: 0 <= n < N and 0 < p < 1
            logPMF = N * np.log1p(-p)
            logCDF = logPMF
            for k in range(1, n+1):
                logPMF += np.log(float(N - k + 1) * p / (k * (1.0 - p)))
                logCDF = np.logaddexp(logCDF, logPMF)
            
    return logCDF if logCDF <= 0.0 else 0.0


@jit(nopython=True)
def _negNullPowerFunctionSuissaShuster(p, z, N):
    def inz(_n, _z):
        return np.ceil( (_z * np.sqrt(_n) + _n) / 2.0 )
    
    if z == 0.0:
        logValue = 0.0
    else:
        if p == 0.0:
            logValue = np.NINF
        else:
            k = np.ceil(z * z)
            logConst = np.log(1.0 - p) - np.log(p)
    
            logCoeff = N * np.log(p)
            logTerm = logCoeff + _logBinomialCDF(N - inz(N, z), N, 0.5)
            logValue = logTerm
            for n in range(N, k, -1): # n = N, ..., k+1    
                logCoeff += np.log(n) - np.log(N - n + 1) + logConst
                logTerm = logCoeff + _logBinomialCDF(n - 1 - inz(n - 1, z), n - 1, 0.5)
                logValue = np.logaddexp(logValue, logTerm)
        
    return -logValue


# SuissaShusterTest
#
# Implements the p-value (observed significance level) of the Suissa-Shuster 
# Test for comparing two binomial proportions in terms of matched pairs.
#
# SYNTAX
#  pvalue, idxBest = SuissaShusterTest(MatchedPairs)
#
# INPUTS
# MatchedPairs: 2D numpy.ndarray of N>=1 rows and 2 columns that only take values in 
#          {0,1} or {True, False}. Each row contains the classification outcome
#          (1/True, if the prediction was correct, and 0/False, if otherwise) 
#          of 2 classification models for a specific sample.
#
# OUTPUTS
# pvalue:  float in [0,1]; p-value (observed significance level) of the test.
# idxBest: integer in {0,1}; indicates the classification model with the most 
#          number of 1/True's between the two.
#
# NOTES
# 1) Assume that MatchedPairs[:,0] and MatchedPairs[:,1] are paired, i.i.d samples 
#    generated from model0 and model1 respectively. The null hypothesis of 
#    the test is that the probabilities of observing a discordant pair (1,0)
#    or (0,1) are equal.
#    Without loss of generality, assume that the number of 
#    1/True's is higher for model0. Then, the alternative hypothesis states
#    that the probability of observing (1,0) pairs is higher than observing
#    (0,1) pairs.
#    An alpha-level significance test would reject the null hypothesis if
#    p-value <= alpha.
# 2) If both columns of MatchedPairs are identical, idxBest defaults to 0.
# 3) The Suissa-Shuster Test is an unconditional test; it takes into account 
#    both concordant and discordant pairs of observations. It is described in
#    Samy Suissa and Jonathan J. Shuster. The 2x2 Matched-Pairs Trial: Exact 
#    Unconditional Design and Analysis. Biometrics, 47(2), 361-372, June 1991. 
#
# DEPENDENCIES
#  import numpy as np
#  import scipy.optimize
#  import functools
#  _negNullPowerFunctionSuissaShuster()
# 
# AUTHOR
#  Georgios C. Anagnostopoulos, March 2023
#
def SuissaShusterTest(MatchedPairs):
    N = len(MatchedPairs)
    ND0obs = sum((MatchedPairs[:, 0] == 1) & (MatchedPairs[:, 1] == 0))
    ND1obs = sum((MatchedPairs[:, 0] == 0) & (MatchedPairs[:, 1] == 1))
    NDobs = ND0obs + ND1obs
    if ND0obs >= ND1obs:
        idxBest = 0
        y = ND0obs
        x = ND1obs
    else:
        idxBest = 1
        y = ND1obs
        x = ND0obs

    if x == y:
        pvalue = 1.0
    else:
        z = float(y - x) / np.sqrt(x+y)
        nnpf = functools.partial(_negNullPowerFunctionSuissaShuster, z=z, N=N)
        minRes = scipy.optimize.minimize_scalar(nnpf, bounds=(
            0.0, 1.0), method='bounded')
        pvalue = np.exp(- minRes.fun)
    return pvalue, idxBest


# sort_matchedtuples_models
#
# 
def sort_matchedtuples_models(MatchedTuples, model_names):
    num_correct_decisions = np.sum(MatchedTuples, axis=0)
    sorted_indices = num_correct_decisions.argsort()[::-1]
    sorted_MatchedTuples = MatchedTuples[:, sorted_indices]
    sorted_model_names = [model_names[i] for i in sorted_indices]
    return sorted_MatchedTuples, sorted_model_names


# HolmBonferroniProcedure
#
# The function implements the Holm-Bonferroni (a.k.a. Holm's Step-Down) 
# procedure for the simultaneous testing of mutliple statistical hypotheses
# involving comparisons of classification accuracies and where each test 
# of hypothesis is based on matched pairs.
#
# SYNTAX
#  adj_pValues, ModelPairIndeces = HolmBonferroniProcedure(sorted_MatchedTuples, pValueFunc, mode)
#
# INPUTS
# sorted_MatchedTuples: 2D numpy.ndarray of N>=1 rows and M columns that only takes 
#                values in {0,1} or {True, False}. Each row contains the 
#                classification outcome (1/True, if the prediction was correct, 
#                and 0/False, if otherwise) of M>=2 classification models 
#                for a specific sample. There are a total of N samples.
#                IMPORTANT: columns (models) are sorted in a non-increasing
#                order of accuracy.
# pValueFunc:    callable; pointer to a function that computes the p-value
#                of a matched pair test and that has a signature of 
#                pvalue, idxbest = pValueFunc(MatchedPairs).
#                SuissaShusterTest() is such an example function. 
# mode:          string. If ='best_only', the most accurate of these M models
#                will be tested individually against each of the remaining 
#                models. This results in C=M-1 comparison tests.
#                If !='best_only', then the most accurate model will be 
#                individually tested against the rest; next, the 2nd most 
#                accurate model will be individually tested against the rest; 
#                and so forth. This results in C=M*(M-1)/2 comparison tests.
#
# OUTPUTS
# adj_pValues:      1D numpy.ndarray of C floats in [0, 1] containing the 
#                   adjusted p-values of the C model comparisons. Each of 
#                   its elements can be compared to a significance level alpha 
#                   to establish, whether one model is significantly (in a 
#                   statistical sense) more accurate than another.
# ModelPairIndeces: 2D numpy.ndarray of C rows and 2 columns. Its c-th row 
#                   provides the two column indices of sorted_MatchedTuples and, hence, 
#                   the model IDs being compared that correspond to the 
#                   value adj_pValues[c]. See note below for more details.
# mode:             string. If =='best_only', only the model with the highest
#                   accuracy (assumed to be unique) will be compared to the 
#                   rest of the models, which results in M-1 comparisons. 
#                   If !='best_only', except the model with the worst accuracy 
#                   (assumed to be unique), every model will be compared to 
#                   models of worse (or equal) accuracy, which results in 
#                   (M-1)*M/2 comparisons.
#
# NOTES
# 1) When the adjusted p-values are compared to a significance level alpha, 
#    the procedure guarantees that the Family-wise Error Rate (i.e., the 
#    probability of at least one true null hypothesis is falsely rejected), 
#    is less than or equal alpha.
# 2) adj_pValues[c] contains the adjusted p-value for testing that the model,
#    whose classification results are provided in column ModelPairIndeces[c,0] 
#    of sorted_MatchedTuples, is more accurate than the model, whose classification 
#    results are provided in column ModelPairIndeces[c,1] of sorted_MatchedTuples.
# 3) When mode=='best_only' and there is more than one model that has the best 
#    accuracy, only the first one (lowest column index) will be compared to 
#    the rest.
#
# DEPENDENCIES
#  import numpy as np
# 
# AUTHOR
#  Georgios C. Anagnostopoulos, March 2023
#
def HolmBonferroniProcedure(sorted_MatchedTuples, pValueFunc, mode):
    N, M = sorted_MatchedTuples.shape  # N: number of test samples, M: number of models
        
    # Determine how many comparisons should be considered
    if mode is 'best_only':
        # comparisons of best vs rest
        max_idx = 0
        C = M-1   # number of comparisons
    else:
        # comparisons of best vs rest, 2nd best vs rest, etc.
        max_idx = M-1   
        C = M * (M-1) // 2   # number of comparisons 
    
    # Compute p-values based on a pair-wise test
    pValues = np.empty(C, dtype=np.float64)
    ModelPairIndeces = np.empty((C,2), dtype=int)
    c=0
    for m1 in range(max_idx + 1):
        for m2 in range(m1+1,M):
            MatchedPairs = sorted_MatchedTuples[:,[m1,m2]]
            pValue, _ = pValueFunc(MatchedPairs)
            pValues[c] = pValue
            ModelPairIndeces[c,:] = [m1, m2]
            c += 1

    # Sort p-values in ascending order
    indices = pValues.argsort()
    pValues = pValues[indices] 
    ModelPairIndeces = ModelPairIndeces[indices, :]
    
    # compute adjusted p-values
    adj_pValues = np.empty_like(pValues)
    adj_pValues[0] = min(1.0, C * pValues[0])
    for c in range(1,C):
        adj_pValues[c] = max(adj_pValues[c-1], (C-c)*pValues[c])
    
    return adj_pValues, ModelPairIndeces
    

def mk_adj_pvalue_matrix(adj_pValues, ModelPairIndeces, mode):
    C = ModelPairIndeces.shape[0] 
    
    if mode is 'best_only':
        M = C+1
        AdjPvalueMatrix = np.ones((1,M), dtype=float)
    else:
        M = int((1 + np.sqrt(8*C+1))/2)
        AdjPvalueMatrix = np.ones((M,M), dtype=float)
    
    for c in range(C):
        row = ModelPairIndeces[c,0]
        col = ModelPairIndeces[c,1]
        AdjPvalueMatrix[row,col] = adj_pValues[c]

    return AdjPvalueMatrix


# Sample Demonstration Code 
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from matplotlib import colors
    
    # Some global settings for figure sizes
    normalFigSize = (8, 6) # (width,height) in inches
    largeFigSize = (12, 9)
    xlargeFigSize = (16, 12)
    
    # Generate simulated data #################################################
    import string

    M = 16  # number of models participating in the comparison; must be <= 26
    N = 150 # number of test samples used to compare the models

    # Model names are 'A', 'B', 'C', etc.
    # Column 0, 1, 2, etc. of MatchedTuples will contain the classification 
    # results (1, if a sample is correctly classified and 0, if otherwise) of
    # models 'A', 'B', 'C', etc.    

    model_names = list(string.ascii_uppercase[:M])
    true_accuracies = np.random.uniform(size=M)
    MatchedTuples = np.empty((N,M), dtype=int)
    for m in range(M):
        MatchedTuples[:, m] = np.random.binomial(1, true_accuracies[m], size=N)

    # Clopper-Pearson CIs #####################################################
    p = 0.99   # nominal coverage probability of CIs

    sorted_MatchedTuples, sorted_model_names = sort_matchedtuples_models(MatchedTuples, model_names)
    
    num_correct_decisions = np.sum(sorted_MatchedTuples, axis=0)
    estimated_accuracies = num_correct_decisions / N

    lb_array = np.empty(M, dtype=float)
    ub_array = np.empty(M, dtype=float)
    print("Estimated Accuracies & {}%-level Clopper-Pearson Intervals".format(100.0 * p))
    print()
    print("Model Name\tLB\tACC\tUB")
    for m in range(M):
        lb, ub = ClopperPearsonCI(num_correct_decisions[m], N, p)
        lb_array[m] = lb
        ub_array[m] = ub
        print("{}\t\t{:0.03f}\t{:0.03f}\t{:0.03f}".format(sorted_model_names[m], lb, estimated_accuracies[m], ub))
    
    # Create a plot
    fig = plt.figure(figsize=xlargeFigSize)
    ax = fig.add_subplot(1, 1, 1)

    fontsize = 16
    labelsize = 14
    ax.set_ylim([0.0, 1.0])
    ax.tick_params(axis='x', labelsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)
    ax.set_yticks(np.linspace(0.0, 1.0, 21))
    ax.grid(axis='y')
    plt.setp(ax.get_xticklabels(), rotation=90)

    # Stacked bar chart
    ax.bar(sorted_model_names, estimated_accuracies, bottom = estimated_accuracies - lb_array)
    ax.bar(sorted_model_names, ub_array, bottom = estimated_accuracies)

    ax.set_title("{}%-level Clopper-Pearson confidence itervals".format(100.0 * p), fontsize=14)
    ax.set_xlabel("", fontsize=fontsize)
    ax.set_ylabel("Estimated Accuracy", fontsize=fontsize)

    plt.show()
    #fig.savefig('accuracies_cis.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
    
    
    # Compute Adjusted p-Values via the HB procedure ##########################
    mode=''   #'best_only'
    adj_pValues, ModelPairIndeces = HolmBonferroniProcedure(sorted_MatchedTuples, ExactMcNemarsTest, mode)

    max_FWER = 0.01

    significant_indices = np.where(adj_pValues <= max_FWER)[0]
    ModelPairIndeces_significant = ModelPairIndeces[significant_indices, :]
    print('Significant differences:')
    for pair in range(ModelPairIndeces_significant.shape[0]):
        m1 = ModelPairIndeces_significant[pair, 0]
        m2 = ModelPairIndeces_significant[pair, 1]
        print("{} vs {}    log10(adj-p-value)={:.03f}".format(sorted_model_names[m1], sorted_model_names[m2], np.log10(adj_pValues[significant_indices[pair]])))
    print()
    num_total_comparisons = len(adj_pValues)
    num_significant_comparisons = ModelPairIndeces_significant.shape[0]
    percent_significant_comparisons = 100.0 * num_significant_comparisons / num_total_comparisons
    print("Out of a total of {} comparisons, {} were found significant ({:.02f}%) for an FWER not exceeding {}.".format(num_total_comparisons, num_significant_comparisons, percent_significant_comparisons, max_FWER))
    
    # Create & plot Significance Matrix #######################################
    AdjPvalueMatrix = mk_adj_pvalue_matrix(adj_pValues, ModelPairIndeces, mode)

    max_FWER_values = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    SignificanceMatrix = np.zeros_like(AdjPvalueMatrix, dtype=int)
    num_max_FWER_values = len(max_FWER_values)

    for level in range(num_max_FWER_values):
        max_FWER = max_FWER_values[level]
        idx = np.where(AdjPvalueMatrix <= max_FWER)
        SignificanceMatrix[idx] = level + 1
        
    # Plot Significance Matrix
    fig = plt.figure(figsize=xlargeFigSize)
    ax = fig.add_subplot(1, 1, 1)

    fontsize = 16
    labelsize = 14

    # Customize colormap
    cmap = plt.get_cmap(name='jet', lut=num_max_FWER_values+1)
    bounds =  np.arange(-1.0, num_max_FWER_values+1)+0.5
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # Customize axes
    ax.tick_params(axis='x', labelsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)
    ax.set_xticks(np.arange(M))
    ax.set_xticklabels(sorted_model_names)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.set_yticks(np.arange(M))
    ax.set_yticklabels(sorted_model_names)
    ax.set_title("Significant Model Differences", fontsize=fontsize)

    # Plot heatmap
    heatmap = ax.imshow(SignificanceMatrix, cmap=cmap, norm=norm, interpolation='nearest') 

    # Add a vertical colorbar
    cbar = plt.colorbar(heatmap, ticks=range(num_max_FWER_values+1))
    cbar_ticklabels = ['FWER>{}'.format(max_FWER_values[0])]
    for level in range(num_max_FWER_values):
        max_FWER = max_FWER_values[level]
        cbar_ticklabels.append('FWER<={}'.format(max_FWER))
    cbar.ax.set_yticklabels(cbar_ticklabels) 
    cbar.ax.tick_params(labelsize=labelsize)

    # Turn spines off
    for key, spine in ax.spines.items():
        spine.set_visible(False)

    # Add a white grid    
    ax.set_xticks(np.arange(M+1)-.5, minor=True)
    ax.set_yticks(np.arange(M+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.show()
    #fig.savefig('significant_model_differences.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)