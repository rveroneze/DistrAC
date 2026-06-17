#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 14:02:49 2025

@author: veroneze
"""

# -----------------------------------------------------------------------------
# Implementation of the Distributed Associative Classifier (DAC) algorithm
# Proposed in the paper:
# Venturini, L., Baralis, E., & Garza, P. (2017).
# Scaling associative classification for very large datasets.
# Journal of Big Data, 4(1), 44.
# -----------------------------------------------------------------------------


import collections
import numpy as np
import math
from sklearn.metrics import accuracy_score
from utils_PM import calculate_gini, calculate_ig, calculate_chi_squared


class CAPNode:
    """
    A node in the CAP-tree.
    """
    def __init__(self, item_id, parent):
        self.item_id = item_id
        self.parent = parent
        self.children = {}
        # The paper mentions an array of frequencies for classes.
        # A dictionary is used here for flexibility with class labels.
        self.class_freqs = collections.defaultdict(int)
        self.node_link = None # for linking to next node with same item_id

    def increment_class(self, class_label):
        """Increments the frequency count for a given class label."""
        self.class_freqs[class_label] += 1


class CAPTree:
# --- Algorithm 1: CAP-tree building  ---
    def __init__(self, min_sup=0, min_conf=0, min_chi2=0):
        self.min_sup = min_sup
        self.min_conf = min_conf
        self.min_chi2 = min_chi2
        self.root_ = None
        self.header_table_ = None
        self.total_class_counts_ = None
        self.total_records_ = None
    
    def build_cap_tree(self, X, y):
        """
        Builds a CAP-tree from a TRANSACTIONAL DATABASE X and labels y.
    
        Args:
            X (list): A list of tuples, where each tuple is a transaction
            y (list): A list of class labels
            min_sup_threshold (float): The minimum support threshold in [0, 1].
    
        Returns:
            tuple: A tuple containing the CAP-tree root and the header table.
        """
                
        self.total_records_ = len(X)
                
        # The paper uses a support count, so we calculate it from the min_sup in %.
        min_sup_count = self.min_sup * self.total_records_
        
        item_counts = collections.defaultdict(int)
        item_class_counts = collections.defaultdict(lambda: collections.defaultdict(int))
        self.total_class_counts_ = collections.defaultdict(int)
    
        # --- Step 1: First DB scan ---
        # Scan the DB once. Collect L, the list of frequent items.
        for i, transaction in enumerate(X):
            class_label = y[i]
            self.total_class_counts_[class_label] += 1
            for item in transaction:
                item_counts[item] += 1
                item_class_counts[item][class_label] += 1
        
        # Filter for frequent items
        frequent_items = {item: count for item, count in item_counts.items() if count >= min_sup_count}
                
        # Calculate Information Gain for each frequent item and filter out items with IG <= 0.
        l = []
        for item, _ in frequent_items.items():
            ig = calculate_ig(item_class_counts[item], self.total_class_counts_, self.total_records_)
            if ig > 0: l.append((item, ig))
        
        # Sort L by decreasing IG.
        l.sort(key=lambda x: x[1], reverse=True)
        
        # Create the header table with links to nodes in the tree
        self.header_table_ = {item[0]: [item_counts[item[0]], None] for item in l}
        
        self.root_ = CAPNode(item_id='null', parent=None)
        self.root_.class_freqs = self.total_class_counts_
        
        # --- Step 2 & 3: Second DB scan to build the tree ---
        # Create the root of a CAP-tree T and label it as null.        
        for i, transaction in enumerate(X):
            class_label = y[i]
            # Select only the items in t that appear in L 
            # and sort them according to the order in L.
            ordered_items = []
            for item_ig_pair in l:
                item = item_ig_pair[0]
                if item in transaction:
                    ordered_items.append(item)
            
            # Call insert(t', T)
            self._insert_tree(ordered_items, self.root_, class_label)
        
    
    def _insert_tree(self, items, node, class_label):
        """
        Recursively inserts a transaction into the CAP-tree.
        This corresponds to the 'insert' function in the paper's pseudocode.
        """
        if len(items) < 1:
            return
        
        first_item = items[0]
        child = node.children.get(first_item)
        
        if child is None:
            # Create a new node T'
            child = CAPNode(item_id=first_item, parent=node)
            node.children[first_item] = child
            
            # Update the header table
            # Find the last node in the chain and link to the new child
            if self.header_table_[first_item][1] is None:
                self.header_table_[first_item][1] = child
            else:
                current_node = self.header_table_[first_item][1]
                while current_node.node_link is not None:
                    current_node = current_node.node_link
                current_node.node_link = child
        
        # Increment class frequency for the node
        child.increment_class(class_label)
        
        # Recursive call for remaining items
        remaining_items = items[1:]
        if len(remaining_items) > 0:
            self._insert_tree(remaining_items, child, class_label)
    
    def print_tree(self):
        if self.root_ != None:
            self._print_tree(self.root_)
        else:
            print('There is no tree to display.')
    
    def _print_tree(self, node, indent=""):
        if node.item_id != 'null':
            print(f"{indent}{node.item_id} {dict(node.class_freqs)}")
        for child_id, child_node in node.children.items():
            self._print_tree(child_node, indent + "  ")


# --- Algorithm 2: CAP-growth ---
    def _calculate_node_ig(self, node):
        """Calculates Information Gain for a node relative to its parent - Eq. 3"""
        parent_freqs = node.parent.class_freqs
        parent_total = sum(parent_freqs.values())
        node_freqs = node.class_freqs
        node_total = sum(node_freqs.values())
    
        #if parent_total == 0 or node_total == 0:
        #    return 0.0
    
        gini_parent = calculate_gini(parent_freqs, parent_total)
        gini_node = calculate_gini(node_freqs, node_total)
        
        w_t = node_total / parent_total
        
        return w_t * (gini_parent - gini_node)
    
    
    def _project_tree(self, pattern):
        """
        Projects the CAP-tree to find the true frequency counts for a pattern.
        This is done by recursively building conditional trees.
        """
        # Find all paths for the last item in the pattern
        item_id = pattern[-1]
        
        # If the pattern has only one item, the frequencies are the sum of class_freqs from all nodes
        if len(pattern) == 1:
            final_freqs = collections.defaultdict(int)
            curr = self.header_table_[item_id][1]
            while curr is not None:
                for classe, count in curr.class_freqs.items():
                    final_freqs[classe] += count
                curr = curr.node_link
            return final_freqs
            
        # Trace paths upwards from the item's nodes
        conditional_paths = []
        current_node = self.header_table_[item_id][1]
        while current_node is not None:
            path = []
            temp_node = current_node
            # We can exclude the current node itself, and only trace the path starting from its parent:
            while temp_node.parent is not None and temp_node.parent.item_id != 'null':
                path.append(temp_node.parent.item_id)
                temp_node = temp_node.parent
            if len(path) > 0:
                ## Suppose a path: root -> A -> C -> D -> E, then path=[E, D, C, A].
                ## For the reversed order (i.e., path=[A, C, D, E]), do: 
                # conditional_paths.append((list(reversed(path)), current_node.class_freqs))
                ## But, due to the designed of the CAP-tree, I think it will not make any difference in this routine
                conditional_paths.append((set(path), current_node.class_freqs))            
            current_node = current_node.node_link
        
        # This is a simplified projection that sums up frequencies from valid conditional paths
        # A full re-implementation would build an entirely new conditional tree.
        remaining_pattern = set(pattern[:-1])
        final_freqs = collections.defaultdict(int)
        for path, freqs in conditional_paths:
            # Check if the remaining pattern is a subset of the conditional path
            if remaining_pattern.issubset(path):
                for classe, count in freqs.items():
                    final_freqs[classe] += count
                    
        return final_freqs
    
    
    def cap_growth(self):
        """
        Main function to run the CAP-growth algorithm.
    
        Args:
    
        Returns:
            list: A list of generated Class Association Rules (CARs).
        """
        rules = []
        for child in self.root_.children.values():
            rules.extend(self._extract(child))
        return rules
    
    def _extract(self, node):
        """Recursive function to extract rules from a node and its children."""
        rules = []
        
        # Stopping criterion 1: Negative Information Gain
        if self._calculate_node_ig(node) <= 0:
            return []
    
        # Stopping criterion 2: Pure node
        gini = calculate_gini(node.class_freqs, sum(node.class_freqs.values()))
        if gini == 0:
            return self._generate_rule(node)
        
        # Recursive call for children
        for child in node.children.values():
            rules.extend(self._extract(child))
        
        # If no children produced a rule, try to generate one from the current node
        if len(rules) < 1:
            return self._generate_rule(node)
        
        return rules
    
    def _generate_rule(self, node):
        """Tries to generate a valid CAR from a node path."""
        # Reconstruct antecedent from path to root
        antecedent = [node.item_id]
        curr = node
        while curr.parent is not None and curr.parent.item_id != 'null':
            antecedent.append(curr.parent.item_id)
            curr = curr.parent
        antecedent.reverse()
        
        #if not antecedent: return []
        
        # Project tree to get true frequencies for the full pattern
        pattern_freqs = self._project_tree(antecedent)
        sup_count_antecedent = sum(pattern_freqs.values())
        
        #if not pattern_freqs: return []
        
        # Consequent is the class with the highest value
        consequent, sup_count_rule = max(pattern_freqs.items(), key=lambda item: item[1])
        
        # Calculate metrics
        support = sup_count_rule / self.total_records_
        confidence = sup_count_rule / sup_count_antecedent
        chi2 = calculate_chi_squared(pattern_freqs, self.total_class_counts_, self.total_records_)
        
        # Check if the rule is valid
        if support >= self.min_sup and confidence >= self.min_conf and chi2 >= self.min_chi2:
            rule = {
                'antecedent': frozenset(antecedent),
                'consequent': consequent,
                'support': support,
                'confidence': confidence,
                'chi2': chi2,
                'sup_count_rule': sup_count_rule,           
                'sup_count_antecedent': sup_count_antecedent,
                'sup_antecedent': sup_count_antecedent / self.total_records_,
                'recall': sup_count_rule / self.total_class_counts_[consequent]
                }
            # rule = f"{set(antecedent)} => {consequent}"
            # print(f"Generated Rule: {rule} (sup: {support:.3f}, conf: {confidence:.3f}, chi2: {chi2:.3f})")
            return [rule]
        
        return []


class DAC:
    def __init__(self, min_sup=0, min_conf=0, min_chi2=0):
        self.min_sup=min_sup
        self.min_conf = min_conf
        self.min_chi2 = min_chi2
        self.CAPTree_ = CAPTree(min_sup, min_conf, min_chi2)
        self.rules_ = None
        self.classes_ = None
        self.global_class_distribution_ = None
        self._no_matches = 0
        

    def fit(self, X, y):
        print(f"Building CAP-Tree with minimum support: {self.min_sup}")
        self.CAPTree_.build_cap_tree(X, y)
        print('CAP-Tree is built.\n')
        
        print(f"Growing CAP-Tree with minimum support: {self.min_sup},  minimum confidence: {self.min_conf},  minimum chi2: {self.min_chi2}")
        self.rules_ = self.CAPTree_.cap_growth()
        print('Done.\n')
        
        self._set_classes()
        self._set_global_class_distribution()
        #classes, n_per_label = np.unique(y, return_counts=True)
        #self.classes_ = set(classes)
        #self.global_class_distribution_ = dict(zip( classes, n_per_label/len(y) ))
    
    def _set_classes(self):
        self.classes_ = set(self.CAPTree_.total_class_counts_.keys())
    
    def _set_global_class_distribution(self):
        self.global_class_distribution_ = {}
        for key, value in self.CAPTree_.total_class_counts_.items():
            self.global_class_distribution_[key] = value / self.CAPTree_.total_records_
    
    
    def _predict_scores(self, unlabeled_record):
        """
        Computes a vector of prediction scores for an unlabeled record.
        
        Args:
            unlabeled_record (set): A set of items representing the record to classify.
        
        Returns:
            dict: A dictionary mapping each class label to its normalized prediction score.
        """
        # Find all rules where the antecedent is a subset of the record's items
        matching_rules_by_class = collections.defaultdict(list)
        for rule in self.rules_:
            if rule['antecedent'].issubset(unlabeled_record):
                matching_rules_by_class[rule['consequent']].append(rule)
    
        # If no rules match at all, default to the global class probabilities
        if len(matching_rules_by_class) < 1:
            self._no_matches += 1
            return self.global_class_distribution_
        
        scores_p = {}
        
        # For each label with matching rules, calculate its score p_i
        # The default setting uses the max confidence of the matching rules
        for class_label, rules in matching_rules_by_class.items():
            # m is the confidence measure, f() is the max() function
            confidences = [r['confidence'] for r in rules]
            scores_p[class_label] = max(confidences)
        
        # Identify classes that did not have any matching rules
        classes_with_matches = set(matching_rules_by_class.keys())
        classes_without_matches = self.classes_ - classes_with_matches
        
        # Calculate scores for classes without matching rules
        if len(classes_without_matches) > 0:
            # pX is defined under a naive assumption of independence
            # The formula given is pX = sum(1 - pj) for classes j with matching rules
            pX = math.prod(1 - score for score in scores_p.values())
            
            # The score is distributed among all classes without a match
            score_for_unmatched = pX / len(classes_without_matches)
            
            for class_label in classes_without_matches:
                scores_p[class_label] = score_for_unmatched
    
        # The score vector is finally normalized to sum to one
        total_score = sum(scores_p.values())
        if total_score == 0:
            # Fallback to global distribution if all scores are zero to avoid division by zero
            return self.global_class_distribution_
        
        normalized_scores = {k: v / total_score for k, v in scores_p.items()}
        
        return normalized_scores
    
    
    def predict_proba(self, X):
        self._no_matches = 0
        probas = np.zeros(shape=(len(X),len(self.classes_)))
        for i, record in enumerate(X):
            if not isinstance(record, set):
                record = set(record)
            scores = self._predict_scores(record)
            probas[i] = [scores[classe] for classe in self.classes_]
        return probas, self._no_matches

    
    def predict(self, X):
        probas, no_matches = self.predict_proba(X)
        class_indices = np.argmax(probas, axis=1)
        return np.array(list(self.classes_))[class_indices], no_matches
    
    
    def score(self, X, y):
        y_pred, _ = self.predict(X)
        return accuracy_score(y, y_pred)
        
    

# --- Algorithm 3: Model consolidation ---
class DAC_ensemble:
    def __init__(self, models, label_support_metric='support'):
        self.models = models # A list of models (each model is a DAC instance)
        self.label_support_metric = label_support_metric # gambiarra
        self._model_ = None
        
    def _strategy_func(self, supports, confs, chis):
        """
        Default consolidation strategy function.
        
        This function takes lists of supports, confidences, and chi-squared values
        for a set of identical rules and returns a single value for each metric.
        The default behavior returns the maximum of each, as an upper-bound 
        estimation.
        
        Args:
            supports (list): A list of support values.
            confs (list): A list of confidence values.
            chis (list): A list of chi-squared values.
    
        Returns:
            tuple: A tuple containing the new support, confidence, and chi-squared value.
        """
        return (max(supports), max(confs), max(chis))
    
    def _aggregate(self, rules):
        """
        Aggregates a group of identical rules into a single new rule.
        
        Args:
            rules (list): A list of rule dictionaries, all with the same 
                          antecedent and consequent.
        Returns:
            dict: A single, new rule dictionary.
        """
        if not rules:
            return None
        
        # Get the antecedent and consequent from the first rule
        first_rule = rules[0]
        antecedent = first_rule['antecedent']
        consequent = first_rule['consequent']
        
        # Collect all metrics from the group of rules
        supports = [r[self.label_support_metric] for r in rules]
        confs = [r['confidence'] for r in rules]
        chis = [r['chi2'] for r in rules]
        
        # Use the strategy function to calculate the new metrics
        new_support, new_confidence, new_chi2 = self._strategy_func(supports, confs, chis)
        
        # Create the new aggregated rule
        new_rule = {
            'antecedent': antecedent,
            'consequent': consequent,
            self.label_support_metric: new_support,
            'confidence': new_confidence,
            'chi2': new_chi2,
            'length': len(antecedent)
        }
        
        return new_rule
    
    def _merge(self, model1, model2):
        """
        Merges two models (lists of rules) into one.
        
        This function combines the rules from two models. For rules that are
        identical (in antecedent and consequent), it applies the aggregate
        function to combine them
        
        Args:
            model1 (list): The first model, represented as a list of rule dictionaries.
            model2 (list): The second model.
    
        Returns:
            list: The new, merged model.
        """
        # Combine the rules from both models
        combined_rules = model1 + model2
        
        # Group rules by the same antecedent and consequent
        grouped_rules = collections.defaultdict(list)
        for rule in combined_rules:
            #key = (frozenset(rule['antecedent']), rule['consequent'])
            key = (rule['antecedent'], rule['consequent'])
            grouped_rules[key].append(rule)
            
        # Aggregate each group of identical rules into a single rule
        merged_model = []
        for key, rule_group in grouped_rules.items():
            merged_model.append(self._aggregate(rule_group))
            
        return merged_model
    
    def model_consolidation(self):
        """
        Consolidates a list of models into a single model.
        
        This function recursively reduces a list of models by merging them
        two by two until only one remains.
        
        Args:
    
        Returns:
            list: A single, consolidated model.
        """
        if self.models == None or len(self.models) < 1:
            return []
        
        self._model_ = DAC()
                
        self._model_.CAPTree_.total_class_counts_ = self.models[0].CAPTree_.total_class_counts_.copy()
        self._model_.CAPTree_.total_records_ = self.models[0].CAPTree_.total_records_
        self._model_.classes_ = self.models[0].classes_
        self._model_.rules_ = self.models[0].rules_
        
        # Recursively merge models two by two
        for i in range(1, len(self.models)):
            self._model_.rules_ = self._merge(self._model_.rules_, self.models[i].rules_)
            self._model_.CAPTree_.total_records_ += self.models[i].CAPTree_.total_records_
            self._model_.classes_ = self._model_.classes_.union(self.models[i].classes_)
            for key, value in self.models[i].CAPTree_.total_class_counts_.items():
                self._model_.CAPTree_.total_class_counts_[key] += value
        self._model_._set_global_class_distribution()
        
    def model_consolidation_external(self):
        if self.models == None or len(self.models) < 1:
            return []
        
        new_model = self.models[0]
        
        # Recursively merge models two by two
        for i in range(1, len(self.models)):
            new_model = self._merge(new_model, self.models[i])
        
        return new_model
        
    
    def get_total_class_counts(self):
        return self._model_.CAPTree_.total_class_counts_

    def get_total_records(self):
        return self._model_.CAPTree_.total_records_
    
    def get_classes(self):
        return self._model_.classes_
    
    def get_rules(self):
        return self._model_.rules_

    def predict_proba(self, X):
        if self._model_ == None:
            print('There is no consolidate model!')
            return
        return self._model_.predict_proba(X)

    def predict(self, X):
        if self._model_ == None:
            print('There is no consolidate model!')
            return
        return self._model_.predict(X)
    
    def score(self, X, y):
        if self._model_ == None:
            print('There is no consolidate model!')
            return
        return self._model_.score(X,y)

if __name__ == '__main__':
    print('Testing - Algorithm 1: CAP-tree building - and - Algorithm 2: CAP-growth')
    print('Note: When items tie in IG: the order in which they appear impacts the outcome!!!')
    print('This test uses the example dataset from Table 1 of the paper.')
    print('We used the mininum thresholds proposed in the example of the paper.\n')
    
    # Dataset in Table 1 of the paper, but with change in the order of the transactions,
    # to provide the exactly same tree in the paper.
    example_X = [
        ['A', 'B', 'C', 'D', 'E'],
        ['A', 'B', 'D', 'E'],
        ['B', 'C', 'E'],
        ['A', 'B', 'D', 'E'],
        ['A', 'B', 'C', 'E'], 
        ['B', 'C', 'D']
    ]
    example_y = ['+', '+', '-', '+', '-', '-']

    min_support_threshold = 0.3 # 2 records for a 6-record dataset.
    min_confidence_threshold = 0.51
    min_chi2_threshold = 0.0
    
    clf = DAC(min_sup=min_support_threshold, min_conf=min_confidence_threshold, min_chi2=min_chi2_threshold)
    clf.fit(example_X, example_y)

    print("\n--- CAP-Tree Structure ---")
    clf.CAPTree_.print_tree()
    
    print("\n--- Header Table ---")
    print("Item\tSupport Count")
    for item, data in clf.CAPTree_.header_table_.items():
        print(f"{item}\t{data[0]}")
    
    print("\n--- Model of the example ---")
    if len(clf.rules_) > 0:
        for i, r in enumerate(clf.rules_):
            print('Rule ', i, ':', r)
    else:
        print("No rules generated.")
    
    # I am consolidating the same model 3x just to test the code
    clf_merged = DAC_ensemble([clf, clf, clf])
    clf_merged.model_consolidation()
    consolidated_rules = clf_merged.get_rules()
    
    print('\nConsolidate model')
    for rule in consolidated_rules:
        print(f"  {rule}")
    print('CLASSES AFTER MERGED', clf_merged.get_classes())
    
    X_test = [
        ['C'],
        ['A', 'D', 'F'],
        ['E']
    ]
    y_test = ['-', '+', '-']
    
    print('\nTest Accuracy', clf_merged.score(X_test, y_test))