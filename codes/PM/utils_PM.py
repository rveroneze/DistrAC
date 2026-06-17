#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 14:04:38 2025

@author: veroneze
"""

import numpy as np
from scipy.stats import chi2

# =============================================================================
# Implementation based on the paper:
# Venturini, L., Baralis, E., & Garza, P. (2017).
# Scaling associative classification for very large datasets.
# Journal of Big Data, 4(1), 44.

def calculate_gini(class_counts, total_records):
    """
    Calculates the Gini impurity for a set of class counts.
    Gini = Σ fi(1-fi) where fi is the frequency of class i.
    """
    if total_records == 0:
        return 0.0
    
    impurity = 0.0
    for class_label in class_counts:
        freq = class_counts[class_label] / total_records
        impurity += freq * (1 - freq)
        
    return impurity

def calculate_ig(item_i_class_counts, total_class_counts, total_records):
    """
    Calculates the Information Gain for an item - Eq. 2.
    IG_i = w_i * (Gini_D - Gini_i)
    """
    item_total_count = sum(item_i_class_counts.values())
    if item_total_count == 0:
        return 0.0
    
    gini_d = calculate_gini(total_class_counts, total_records)
    gini_i = calculate_gini(item_i_class_counts, item_total_count)
    
    w_i = item_total_count / total_records
    
    ig = w_i * (gini_d - gini_i)
    return ig
# =============================================================================


# =============================================================================
# The chi2 statistic was implemented based on the papers:
# 1)
# Alvarez, S. A. (2003).
# Chi-squared computation for association rules: preliminary results.
# Boston, MA: Boston College, 13.
# 2)
# Zimmermann, A., & De Raedt, L. (2004, October).
# Corclass: Correlated association rule mining for classification.
# In International Conference on Discovery Science (pp. 60-72). Berlin, Heidelberg: Springer Berlin Heidelberg.
# 3)
# Nijssen, S., & Kok, J. N. (2005, October).
# Multi-class correlated pattern mining.
# In International Workshop on Knowledge Discovery in Inductive Databases (pp. 165-187). Berlin, Heidelberg: Springer Berlin Heidelberg.

def calculate_chi_squared(pattern_counts, global_class_counts, total_records):
    """
    Calculates the chi-squared statistic for a rule.
    
    Args:
        pattern_freqs (dict): Number of instances for each class label inside the pattern
        global_class_freqs (dict): Number of instances for each class label
        total_records: Number of instances of the dataset

    Returns:
        Chi2 statistic
    """
        
    chi_squared = 0.0
    pattern_total_count = sum(pattern_counts.values())
    
    for class_label, class_count in global_class_counts.items():
        # Observed values
        o1 = pattern_counts.get(class_label, 0) # Pattern and class: sup(X->c)
        o2 = class_count - o1 # sup(notX -> c)
        
        # Expected values
        e1 = pattern_total_count * class_count / total_records # sup(X)*sup(c)/total_records
        e2 = (total_records - pattern_total_count) * class_count / total_records # sup(notX)*sup(c)/total_records
        
        if e1 > 0: chi_squared += ((o1 - e1) ** 2) / e1
        if e2 > 0: chi_squared += ((o2 - e2) ** 2) / e2

    return chi_squared


def chi2Tominsups(ns, theta):
    """
    Calculates the minsup for each class label
    
    Args:
        ns (list): Number of instances for each class label
        theta: Minimim chi2 threshold

    Returns:
        tuple: A tuple containing the relative and absolute minsups for each class label
    """
    
    N = np.sum(ns)
    rminsups = []
    minsups = []
    for ni in ns:
        rminsup = (theta * N) / (np.power(N,2) - ni*N + theta*ni)
        minsup = int(np.ceil(rminsup * ni))
        rminsups.append(rminsup)
        minsups.append(minsup)
    return (rminsups, minsups)
# =============================================================================

# Paper CMAR
def calculate_max_chi2(supP, supc, T):
    e = (1/(supP*supc)) + (1/(supP*(T-supc))) + (1/((T-supP)*supc)) + (1/((T-supP)*(T-supc)))
    return np.power(min(supP,supc) - (supP*supc)/T, 2) * T * e
    
# =============================================================================

def getChi2CriticalValue(alpha, df):
    """
    Args:
        alpha: probability of exceeding the crtitical value
        df: degrees of freedom
        
    Returns:
        float: Chi2 crtitical value
    """
    return chi2.ppf(1 - alpha, df)


def boolArray2Transactions(array):
    return [list(np.where(row)[0]) for row in array]

def boolDF2Transactions(df):
    return [list(df.columns[df.iloc[i].astype(bool)]) for i in range(len(df))]

def get_coverage(rules, X, y):
    """
    Return the number of times a trainning instance is covered by
    the rule's antecedent and by the complete rule
    
    Parameters
    ----------
    rules: dictionary
    X: training dataset
    y: Trainning target vector
    """
    n = len(y)
    cov_antecedent = np.zeros(shape=(n,), dtype=int)
    cov_rule = np.zeros(shape=(n,), dtype=int)
    for rule in rules:
        for i, record in enumerate(X):
            rec_set = frozenset(record)
            if rule['antecedent'].issubset(rec_set):
                cov_antecedent[i] += 1
                if rule['consequent'] == y[i]:
                    cov_rule[i] += 1
    return cov_antecedent, cov_rule

def solve_ties_default_class(classes_counts, data_classes_counts):
    """
    If there is an tie, we look in the training data to choose the default class label
    """
    if len(classes_counts) == 0:
        return max(data_classes_counts, key=data_classes_counts.get)
    
    max_count = max(classes_counts.values())
    tied = [classe for classe, count in classes_counts.items() if count == max_count]
    if len(tied) == 1:
        return tied[0]
    
    return max(tied, key=lambda c: data_classes_counts.get(c, 0))

def get_rankings():
    rankings = {}
    rankings[1] = (["confidence", "rsup_rule", "length"], [False, False, True])
    rankings[2] = (["confidence", "completeness", "rsup_rule", "length"], [False, False, False, True])
    return rankings    
