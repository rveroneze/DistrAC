#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 19:14:50 2025

@author: veroneze
"""

import collections
import numpy as np
import pandas as pd
from utils_PM import calculate_chi_squared, calculate_max_chi2

class CAR_base:
    def __init__(self):
        self.rules = []
        self.classifier_ = []
        self.default_class_ = None
        self.notCoveredRows_ = 0
        self.no_matches_ = 0

    def set_rules(self, rules):
        self.rules= rules

    def add_rule(self, conditions: dict, class_label, metrics={}, extra={}):
        """
        Add a new rule to the set.
        
        Parameters
        ----------
        conditions: dict of {attribute: value} (antecedent of the rule)
        class_label: consequent of the rule
        metrics: dict of {attribute: value}, such as {"confidence": 0.75, "chi2": 4}
        extra: dict of {attribute: value} -- other info of the rule
        """
        rule = {}
        rule['antecedent'] = frozenset(conditions.items())
        rule['consequent'] = class_label
        rule.update(metrics)
        rule.update(extra)
        self.rules.append(rule)
    
    def rinclosecvcp_output_to_rules(self, filepath, X_train, y_train):
        """
        Reads RIn-Close_CVCP's output and build the class association rules
        
        Parameters
        ----------
        filepath: string with the filepath of the RIn-Close_CVCP's output
        X_train: training dataset used during pattern mining
        y_train: training targets used during pattern mining
        """
        patterns = self._read_rinclose_output(filepath)
        self._build_rules_from_rinclosecvcp(patterns, X_train, y_train)
        
    def _read_rinclose_output(self, filename):
        bics = []
        with open(filename) as bicfile:
            for line in bicfile:
                exec(line)
        return bics
    
    def _build_rules_from_rinclosecvcp(self, patterns, X_train, y_train):
        n = y_train.shape[0] # number of instances in the training data
        classes, nsamples = np.unique(y_train, return_counts=True)
        class_counts_data = dict(zip(classes, nsamples))
        
        self.rules = []
        for i, pattern in enumerate(patterns):
            rows = np.array(pattern[0])
            cols = pattern[1]
            
            # Rule condition (antecedent):
            aux = X_train.iloc[rows[0], cols]
            condition = dict(zip(aux.index, aux.values))
            
            # Rule consequent (best class label):
            labelsOfRows = y_train[rows]
            classes_pattern, counts_pattern = np.unique(labelsOfRows, return_counts=True)
            pattern_counts = dict(zip(classes_pattern, counts_pattern))
            best_class = classes_pattern[np.argmax(counts_pattern)]
            
            metrics = {}
            metrics['sup_antecedent'] = rows.shape[0]
            metrics['sup_consequent'] = class_counts_data[best_class]
            metrics['sup_rule'] = pattern_counts[best_class]
            metrics['rsup_antecedent'] = metrics['sup_antecedent'] / n
            metrics['rsup_consequent'] = metrics['sup_consequent'] / n
            metrics['rsup_rule'] = metrics['sup_rule'] / n
            metrics['confidence'] = metrics['sup_rule'] / metrics['sup_antecedent']
            metrics['completeness'] = metrics['sup_rule'] / metrics['sup_consequent']
            metrics['lift'] = metrics['confidence'] / metrics['rsup_consequent']
            metrics['leverage'] = metrics['rsup_rule'] - metrics['rsup_antecedent'] * metrics['rsup_consequent']
            metrics['length'] = len(cols)
            metrics['chi2'] = calculate_chi_squared(pattern_counts, class_counts_data, n)
            metrics['max_chi2'] = calculate_max_chi2(metrics['sup_antecedent'], metrics['sup_consequent'], n) # From CMAR
            metrics['laplace_acc'] = (metrics['sup_rule'] + 1) / (metrics['sup_antecedent'] + len(classes)) # From CPAR - smooth confidence to penalize low support
            metrics['hm_conf_sup'] = 2 * (metrics['confidence'] * metrics['rsup_rule']) / (metrics['confidence'] + metrics['rsup_rule']) # harmonic mean confidence - support
            metrics['hm_conf_comp'] = 2 * (metrics['confidence'] * metrics['completeness']) / (metrics['confidence'] + metrics['completeness']) # harmonic mean confidence (precision) - completeness  (recall) => f1-measure
            
            extra = {}
            extra['covered_rows_antecedent'] = rows
            #extra['covered_rows_rule'] = rows[labelsOfRows == best_class] # I am not saving it to use less memory
            extra['class_counts'] = pattern_counts
            
            self.add_rule(condition, best_class, metrics, extra)
        
    def prune_by_thresholds(self, thresholds: dict):
        """
        Prune rules that do not meet all given metric thresholds.
        
        Parameters
        ----------
        thresholds : dictionary where keys are metric names (str) and values are thresholds.
        Example: {"confidence": 0.75, "chi2": 4}
        """
        keep_rules = []
        for rule in self.rules:
            keep = True
            for metric, threshold in thresholds.items():
                if metric not in rule:
                    keep = False
                    break
                if rule[metric] < threshold:
                    keep = False
                    break
            if keep:
                keep_rules.append(rule)
        
        self.rules = keep_rules
    
    def rank_rules(self, sort_by=["confidence", "rsup_rule"], ascending=None):
        """
        Rank rules based on given metrics (must be numerical).

        Parameters
        ----------    
        sort_by: list of metric names
        ascending: list of bools (True=ascending, False=descending) — defaults to descending
        """
        
        if ascending is None:
            ascending = [False] * len(sort_by)  # default descending for all
        
        self.rules = sorted(
            self.rules,
            key=lambda r: tuple(
                r[m] if asc else -r[m]
                for m, asc in zip(sort_by, ascending)
            )
        )
    
    def get_coverage(self, rules, y_train):
        """
        Return the number of times a trainning instance is covered by
        the antecedent and the complete rule
        
        Parameters
        ----------
        y_train: Trainning target vector
        """
        n = len(y_train)
        cov_antecedent = np.zeros(shape=(n,), dtype=int)
        cov_rule = np.zeros(shape=(n,), dtype=int)
        for rule in rules:
            cov_antecedent[rule['covered_rows_antecedent']] += 1
            
            idx = rule['covered_rows_antecedent']
            mask = y_train[idx] == rule['consequent']
            cov_rule[idx[mask]] += 1
            
        return cov_antecedent, cov_rule
        
    
    def _solve_ties_else(self, classes, counts, y):
        """
        If there is an tie, we look in the training data to choose the class label
        """
        max_count = np.max(counts)
        tied = classes[counts == max_count]
        if len(tied) == 1:
            return classes[np.argmax(counts)]
        
        global_values, global_counts = np.unique(y, return_counts=True)
        freq = dict(zip(global_values, global_counts))
        return max(tied, key=lambda c: freq.get(c, 0))
    
    def select_coverage(self, y_train, delta=1, sort_by=["confidence", "rsup_rule"], ascending=None):
        """
        Greedy heuristic for coverage-based selection
        Try to cover each trainning instance by delta rules
    
        Parameters
        ----------
        y_train: Trainning target vector
        delta: number of times each instance should be covered by a rule
        sort_by: list of metric names
        ascending: list of bools (True=ascending, False=descending) — defaults to descending
        """
        # Step 1: Rank rules
        self.rank_rules(sort_by=sort_by, ascending=ascending)
        
        # Step 2: iterate over ranked rules
        allCovered = False
        self.classifier_ = []
        coveredRows = np.zeros(shape=(len(y_train),), dtype=int)
        for rule in self.rules:
            idx = rule["covered_rows_antecedent"]
            mask = coveredRows[idx] < delta
            if not np.any(mask):
                continue # The rule does not cover any new instance
            
            # Test if the rule classifies at least one instance
            # that is covered by less than delta rules
            if np.any(y_train[idx[mask]] == rule["consequent"]):
                self.classifier_.append(rule)
                coveredRows[rule['covered_rows_antecedent']] += 1
                if np.all(coveredRows >= delta):
                    allCovered = True
                    print(f'All instances are covered by at least {delta} rules')
                    break
        
        # Define the default label -- used when there is no rule that matches the data point
        if allCovered:
            values, counts = np.unique(y_train, return_counts=True)
            self.default_class_ = values[np.argmax(counts)]
            self.notCoveredRows_ = 0
        else:
            values, counts = np.unique(y_train[coveredRows < delta], return_counts=True)
            self.default_class_ = self._solve_ties_else(values, counts, y_train)
            self.notCoveredRows_ = np.max(counts)
        
        return coveredRows
    
    # Liu, B., Hsu, W., & Ma, Y. (1998, August). Integrating classification and association rule mining. In Kdd (Vol. 98, pp. 80-86).
    def select_coverage_CBA_M1(self, y_train, sort_by=["confidence", "rsup_rule"], ascending=None):
        """
        CBA's greedy heuristic for coverage-based selection (M1 algorithm).
        It is kind of the same as 'select_coverage' with 'delta=1'. But it also computes the
        label of the else rule at each iteration and, after building a rule list, it
        prunes rules at the end of the list based on the number of errors.
        
        Parameters
        ----------
        y_train: Trainning target vector
        sort_by: list of metric names
        ascending: list of bools (True=ascending, False=descending) — defaults to descending
        """
        
        # Step 1: Rank rules
        self.rank_rules(sort_by=sort_by, ascending=ascending)
        
        # Step 2: iterate over ranked rules
        stop_loop = False
        sumFP = 0
        errosTotal = []
        elses = []
        count_notCovered = []
        self.classifier_ = []
        notCoveredRows = np.ones(shape=(len(y_train),), dtype=int)
        for rule in self.rules:
            idx = rule["covered_rows_antecedent"]
            mask = notCoveredRows[idx] == 1
            if not np.any(mask):
                continue # The rule does not cover any new instance
                
            # Test if the rule correctly classifies at least one not covered instance
            if np.any(y_train[idx[mask]] == rule["consequent"]):
                self.classifier_.append(rule)
                notCoveredRows[rule['covered_rows_antecedent']] = 0 # "delete" all instances covered by the select rule
                FP = rule['sup_antecedent'] - rule['sup_rule'] # False positives
                sumFP += FP
                
                # ELSE rule:
                values, counts = np.unique(y_train[notCoveredRows == 1], return_counts=True)
                if len(values) > 0:
                    erroElse = notCoveredRows.sum() - max(counts)
                    else_label = self._solve_ties_else(values, counts, y_train)
                    count_notCovered.append(np.max(counts))
                else:
                    # Else Label will be the majority one in the training data
                    stop_loop = True
                    print('All instances are covered by at least 1 rule')
                    values, counts = np.unique(y_train, return_counts=True)
                    erroElse = 0 # All training instances are covered
                    else_label = values[np.argmax(counts)]
                    count_notCovered.append(0)
                
                erroTotal = sumFP + erroElse
                errosTotal.append(erroTotal)
                elses.append(else_label)
            if stop_loop: break
        
        # Prune phase:
        if len(self.classifier_) > 0:
            idx = np.argmin(errosTotal)
            self.default_class_ = elses[idx]
            self.notCoveredRows_ = count_notCovered[idx]
            self.classifier_ = self.classifier_[:idx+1]
        else:
            values, counts = np.unique(y_train, return_counts=True)
            self.default_class_ = values[np.argmax(counts)]
            self.notCoveredRows_ = np.max(counts)
            
        
        return sumFP, errosTotal, elses
    
    
    # LAZY SELECTION is the selection used by the L3 algorithm
    # Ref1:
    # Baralis, E., & Garza, P. (2002, December).
    # A lazy approach to pruning classification rules.
    # In 2002 IEEE International Conference on Data Mining, 2002. Proceedings. (pp. 35-42).
    # Ref2:
    # Baralis, E., Chiusano, S., & Garza, P. (2008).
    # A lazy approach to associative classification.
    # IEEE Transactions on Knowledge and Data Engineering, 20(2), 156-171.
    def select_lazy(self, y_train, sort_by=["confidence", "rsup_rule"], ascending=None):
        """
        Lazy heuristic for rule selection.
        It is the same as 'select_coverage' with 'delta=1'. But it keeps the non-'harmful' rules in the end of the list.
        
        Parameters
        ----------
        y_train: Trainning target vector
        sort_by: list of metric names
        ascending: list of bools (True=ascending, False=descending) — defaults to descending
        """
        # Step 1: Rank rules
        self.rank_rules(sort_by=sort_by, ascending=ascending)
        
        # Step 2: iterate over ranked rules
        notCoveredRows = np.ones(shape=(len(y_train),), dtype=int)
        self.classifier_ = []
        notHarmful = []
        for i, rule in enumerate(self.rules):
            idx = rule["covered_rows_antecedent"]
            mask = notCoveredRows[idx] == 1
            
            # Test if the rule covers some uncovered instance
            if np.any(mask):
                # Test if the rule correctly classifies at least one not covered instance
                if np.any(y_train[idx[mask]] == rule["consequent"]):
                    self.classifier_.append(rule)
                    notCoveredRows[rule['covered_rows_antecedent']] = 0 # "delete" all instances covered by the select rule
                    if np.sum(notCoveredRows) == 0:
                        print('All instances are covered by at least 1 rule')
                    break
            else: # no correct or wrong classification
                notHarmful.append(rule)
        self.classifier_.extend(notHarmful)
        self.classifier_.extend(self.rules[i+1:]) # non-tested rules (also considered not harmful)
        
    def select_top_k_per_class(self, k=10, sort_by=["confidence", "rsup_rule"], ascending=None):
        """
        Selects the top-k rules per class based on ranking order.
        
        Parameters
        ----------
        k: Number of rules to keep per class.
        sort_by: list of metric names
        ascending: list of bools (True=ascending, False=descending) — defaults to descending
        """
        # Step 1: Rank rules
        self.rank_rules(sort_by=sort_by, ascending=ascending)
    
        # Step 2: Group and prune
        self.classifier_ = []
        rules_by_class = collections.defaultdict(list)
        for rule in self.rules:
            rules_by_class[rule['consequent']].append(rule)
    
        for class_label, rules in rules_by_class.items():
            self.classifier_.extend(rules[:k])  # take top k for this class
    
    def predict_rule_list(self, X):
        """
        Predict the class label for each record in X
        using an ordered rule set (rule list).
        
        Parameters
        ----------
        X: pandas dataframe of shape (n_samples, n_features) OR list of dictionaries
        """
        if type(X) == pd.core.frame.DataFrame:
            X = X.to_dict(orient="records")
        if len(self.classifier_) == 0:
            self.classifier_ = self.rules
        self.no_matches_ = 0
        return [self._predict_one(instance) for instance in X]

    def _predict_one(self, instance: dict):
        """
         Predict the class label for a single instance.
         Returns the label of the first matching rule.
        """
        inst_set = frozenset(instance.items())
        for rule in self.classifier_:
            if rule['antecedent'].issubset(inst_set):
                return rule['consequent']
        self.no_matches_ += 1
        return self.default_class_ # No rule matches the instance
        
    
    def predict_rule_set(self, X, y, strategy='avg', **kwargs):
        """
        Predict the class label for each record in X
        using an unordered rule set (rule set).
        
        Parameters
        ----------
        X: pandas dataframe of shape (n_samples, n_features) OR list of dictionaries
        strategy: voting scheme
        """
        
        strategies = {
            "majority": self.voting_majority,
            "cmar": self.voting_cmar,
            "top_k_avg": self.voting_top_k_avg,
        }
        if strategy not in strategies:
            raise ValueError(f"Unknown voting strategy '{strategy}'. "f"Available: {list(strategies.keys())}")
        
        if type(X) == pd.core.frame.DataFrame:
            X = X.to_dict(orient="records")
        if len(self.classifier_) == 0:
            self.classifier_ = self.rules
        
        # analise = []
        self.no_matches_ = 0
        y_pred = [self.default_class_] * len(X)
        for i, instance in enumerate(X):
            matching_rules_by_class = self._match_rules(instance)
            if len(matching_rules_by_class) > 0:
                y_pred[i] = strategies[strategy](matching_rules_by_class, **kwargs)
            else:
                self.no_matches_ += 1
        return y_pred#, analise
    
    def predict_rule_set_all(self, X):
        if type(X) == pd.core.frame.DataFrame:
            X = X.to_dict(orient="records")
        if len(self.classifier_) == 0:
            self.classifier_ = self.rules
        
        self.no_matches_ = 0
        
        preds = {
            "majority": [],
            "cmar": [],
            "top_k_avg": []
        }
        for i, instance in enumerate(X):
            matching_rules_by_class = self._match_rules(instance)
            if len(matching_rules_by_class) > 0:
                preds["majority"].append(self.voting_majority(matching_rules_by_class))
                preds["cmar"].append(self.voting_cmar(matching_rules_by_class))
                preds["top_k_avg"].append(self.voting_top_k_avg(matching_rules_by_class))
            else:
                self.no_matches_ += 1
                for key in preds:
                    preds[key].append(self.default_class_)
        return preds
    
    def _match_rules(self, instance: dict):
        inst_set = frozenset(instance.items())
        
        # Find all rules where the antecedent is a subset of the record's items
        matching_rules_by_class = collections.defaultdict(list)
        for rule in self.classifier_:
            if rule['antecedent'].issubset(inst_set):
                matching_rules_by_class[rule['consequent']].append(rule)
        return matching_rules_by_class

    
    def voting_majority(self, matching_rules_by_class):
        """
        Select the class with the highest number of match rules
        """
        return max(matching_rules_by_class, key=lambda k: len(matching_rules_by_class[k]))
    
    def voting_max(self, matching_rules_by_class, metric='confidence'):
        """
        Select the class whose rules contain the maximum value of the chosen metric.
        """
        return max(matching_rules_by_class, key=lambda label: max(rule[metric] for rule in matching_rules_by_class[label]))
    
    def voting_avg(self, matching_rules_by_class, metric='confidence'):
        """
        Select the class with the highest average of a given metric
        across its matching rules.
    
        Parameters
        ----------
        matching_rules_by_class : collections.defaultdict(list)
            Keys are class labels, values are lists of rules (dicts with metrics).
        metric : The metric name to use for computing the average.
    
        Returns
        -------
        best_class : The class label with the highest average score.
        """
        best_class = None
        best_score = float("-inf")    
        for class_label, rules in matching_rules_by_class.items():
            score = sum(rule[metric] for rule in rules) / len(rules)
            if score > best_score:
                best_score = score
                best_class = class_label
        return best_class
    
    def voting_sum(self, matching_rules_by_class, metric='confidence'):
        """
        Select the class with the highest sum of a given metric
        across its matching rules.
        """
        best_class = None
        best_score = float("-inf")
        for class_label, rules in matching_rules_by_class.items():
            score = sum(rule[metric] for rule in rules)
            if score > best_score:
                best_score = score
                best_class = class_label    
        return best_class
    
    def voting_cmar(self, matching_rules_by_class):
        best_class = None
        best_score = float("-inf")
        for class_label, rules in matching_rules_by_class.items():
            score = sum(np.power(rule['chi2'],2)/rule['max_chi2'] for rule in rules)
            if score > best_score:
                best_score = score
                best_class = class_label    
        return best_class

    def voting_weighted_avg(self, matching_rules_by_class, metric='confidence', metric_w='completeness'):
        best_class = None
        best_score = float("-inf")
        for class_label, rules in matching_rules_by_class.items():
            numerador = denominador = 0
            for rule in rules:
                numerador += rule[metric_w] * rule[metric]
                denominador += rule[metric_w]
            score = numerador / denominador
            if score > best_score:
                best_score = score
                best_class = class_label
        return best_class    
    
    # Voting scheme used by CPAR
    def voting_top_k_avg(self, matching_rules_by_class, metric='laplace_acc', k=5):
        """
        Select the class label with the highest avg of a metric,
        considering only the top-k rules for each class.
        
        Parameters
        ----------
        matching_rules_by_class : collections.defaultdict(list)
            Keys are class labels, values are lists of rules (dicts with metrics).
        metric : The metric name to use for ranking and summing
        k : The number of top rules to consider per class.
    
        Returns
        -------
        best_class : The class label with the highest avg of the metric among top-k rules.
        """
        best_class = None
        best_score = float("-inf")
        for class_label, rules in matching_rules_by_class.items():    
            # sort rules descending by metric
            sorted_rules = sorted(rules, key=lambda r: r.get(metric, 0), reverse=True)
            
            # take top-k rules (or all if fewer than k)
            top_k_rules = sorted_rules[:k]
            
            # avg of the metric values
            score = sum(rule[metric] for rule in top_k_rules) / len(top_k_rules)
            
            if score > best_score:
                best_score = score
                best_class = class_label
                
        return best_class
    
    
    def __len__(self):
        return len(self.rules)

    def __repr__(self):
        return f"RuleSet({len(self.rules)} rules)"


if __name__ == '__main__':
    # Create rule set
    rules = CAR_base()
    
    # Add rules
    rules.add_rule({"x1": "A", "x2": "B"}, "C1", {'confidence': 1, 'chi2': 1})
    rules.add_rule({"x3": "X", "x4": "Y"}, "C2", {'confidence': 1, 'chi2': 1})
    
    # Predict multiple instances
    dataset = [
        {"x1": "A", "x2": "B", "x3": "Z"},
        {"x3": "X", "x4": "Y"},
        {"x1": "Q", "x2": "W"}
    ]
    print(rules.predict_rule_list(dataset))  # → ['C1', 'C2', None]
