#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 19:39:29 2026

@author: veroneze
"""

from collections import defaultdict
from utils_PM import calculate_chi_squared, calculate_max_chi2, solve_ties_default_class
from base import CAR_base

class Consolidation:

    def __init__(
        self,
        local_classifiers,
        local_datasets=None,
        global_classes_count=None,
        exact_count = True,
        generate_intersections = True,
        min_thresholds={'confidence': 0.51},
        ranking_by=(["confidence", "rsup_rule"], None),
        approximation_fn=None
        ):
        """
        Parameters
        ----------
        local_classifiers : list[list[dict]]
            List of rule lists (one per site).

        local_datasets : list[Dataset-like], optional
            Required for exact strategies (1 and 2).
            
        approximation_fn : callable
            Function(rule, local_rule_instances) -> class_counts
            Used in approximated counting.
        """
        self.local_classifiers = local_classifiers
        self.local_datasets = local_datasets
        self.global_classes_count = global_classes_count # I could replace it and use the local datasets to compute it
        
        self.exact = exact_count
        self.generate_intersections = generate_intersections
        self.min_thresholds = min_thresholds
        self.ranking_by = ranking_by
        self.approximation_fn = approximation_fn
        
        self.global_rules = {}
        self.final_global_rules = [] #after pruning and ranking
        self.uncovered_counts = defaultdict(int)
        self.defaul_class = None
        
    def consolidate(self):
        """
        Main entry point.
        """
        self._merge_same_antecedent_rules()
        
        if self.generate_intersections:
            self._generate_intersection_rules()
            
        if self.exact:            
            self._compute_exact_class_counts()
        else:
            self._compute_approximate_class_counts()
            
        self._finalize_rules()
        self._prune_and_rank_rules()
                
        if self.exact:            
            self._compute_final_uncovered()
            self.defaul_class = solve_ties_default_class(self.uncovered_counts, self.global_classes_count)
        
        return self.final_global_rules, self.defaul_class
    
    def _merge_same_antecedent_rules(self):
        merged = {}
        
        for site_rules in self.local_classifiers:
            for rule in site_rules:
                ant = rule['antecedent']
                
                if ant not in merged:
                    merged[ant] = {
                        'antecedent': ant,
                        'class_counts': defaultdict(int)
                    }
                    
                for label, count in rule['class_counts'].items():
                    merged[ant]['class_counts'][label] += count
                    
        self.global_rules = merged
        self.n_rules_after_merge_ = len(self.global_rules)
        print('Number of rules after merging by antecedent:', self.n_rules_after_merge_)
        
    def _generate_intersection_rules(self):
        antecedents = list(self.global_rules.keys())
        
        for i in range(len(antecedents)):
            a1 = antecedents[i]
            for j in range(i + 1, len(antecedents)):
                a2 = antecedents[j]
                
                new_ant = a1.intersection(a2)
                
                if not new_ant:
                    continue
    
                if new_ant not in self.global_rules:
                    self.global_rules[new_ant] = {
                        'antecedent': new_ant,
                        'class_counts': defaultdict(int)
                    }
        print('Number of rules after generate intersections:', len(self.global_rules))

    def _compute_exact_class_counts(self):
        # For later computing communication costs:
        self.round2_n_rules_ = len(self.global_rules)
        self.round2_n_items_antecedent_ = sum([len(rule['antecedent']) for rule in self.global_rules.values()])
                
        # Reset class counts for all global rules
        for rule in self.global_rules.values():
            rule['class_counts'] = defaultdict(int)
                
        # Iterate over sites
        for dataset in self.local_datasets:
            X_train, y_train = dataset
            
            # For each rule, compute coverage
            for rule in self.global_rules.values():
                antecedent = rule['antecedent']
                for i, inst_set in enumerate(X_train):
                    if antecedent.issubset(inst_set):
                        label = y_train[i]
                        rule['class_counts'][label] += 1
        print('Exact support counted.')
                    
    def _compute_approximate_class_counts(self):
        # TODO
        if self.approximation_fn is None:
            raise ValueError("Approximation function required.")
            
        for ant, rule in self.global_rules.items():
            rule['class_counts'] = self._approximation_fn()

    def _approximation_fn(self):
        # TODO
        return defaultdict(int)
    
    def _finalize_rules(self):
        total_instances = sum(self.global_classes_count.values())
        n_labels = len(self.global_classes_count.keys())
        
        for ant, rule in self.global_rules.items():
            total_covered = sum(rule['class_counts'].values())
            
            majority_class = max(
                rule['class_counts'],
                key=rule['class_counts'].get
            )
            
            rule['consequent'] = majority_class
            rule['sup_antecedent'] = total_covered
            rule['sup_consequent'] = self.global_classes_count[majority_class]
            rule['sup_rule'] = rule['class_counts'][majority_class]
            rule['rsup_antecedent'] = rule['sup_antecedent'] / total_instances
            rule['rsup_consequent'] = rule['sup_consequent'] / total_instances
            rule['rsup_rule'] = rule['sup_rule'] / total_instances
            rule['confidence'] = rule['sup_rule'] / rule['sup_antecedent']
            rule['completeness'] = rule['sup_rule'] / rule['sup_consequent']
            rule['lift'] = rule['confidence'] / rule['rsup_consequent']
            rule['leverage'] = rule['rsup_rule'] - rule['rsup_antecedent'] * rule['rsup_consequent']
            rule['length'] = len(ant)
            rule['chi2'] = calculate_chi_squared(rule['class_counts'], self.global_classes_count, total_instances)
            rule['max_chi2'] = calculate_max_chi2(rule['sup_antecedent'], rule['sup_consequent'], total_instances)
            rule['laplace_acc'] = (rule['sup_rule'] + 1) / (rule['sup_antecedent'] + n_labels)
            rule['hm_conf_sup'] = 2 * (rule['confidence'] * rule['rsup_rule']) / (rule['confidence'] + rule['rsup_rule'])
            rule['hm_conf_comp'] = 2 * (rule['confidence'] * rule['completeness']) / (rule['confidence'] + rule['completeness'])
        print('Rules finalized.')
            
    def _prune_and_rank_rules(self):
        pm = CAR_base()
        pm.set_rules(list(self.global_rules.values()))
        pm.prune_by_thresholds(self.min_thresholds)
        pm.rank_rules(sort_by=self.ranking_by[0], ascending=self.ranking_by[1])
        self.final_global_rules = pm.rules
        print('Number of rules after prunning: ', len(self.final_global_rules))
        
    def _compute_final_uncovered(self):
        # For later computing communication costs:
        self.final_n_items_antecedent_ = sum([len(rule['antecedent']) for rule in self.final_global_rules])
        
        # Reset uncovered counts
        self.uncovered_counts = defaultdict(int)
        
        # Iterate over sites
        for dataset in self.local_datasets:
            X_train, y_train = dataset
            
            covered_mask = [False] * len(X_train)
            
            # For each rule, compute coverage
            for rule in self.final_global_rules:
                antecedent = rule['antecedent']
                for i, inst_set in enumerate(X_train):
                    if antecedent.issubset(inst_set):
                        covered_mask[i] = True
                        
            # Track uncovered instances for this site
            for i, was_covered in enumerate(covered_mask):
                if not was_covered:
                    label = y_train[i]
                    self.uncovered_counts[label] += 1
        print('Count for the default class done.')
