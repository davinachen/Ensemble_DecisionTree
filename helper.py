import numpy as np
import pandas as pd

# 1 Check if a certain partition contains only one class?
def check_purity(y):
    
    if len(np.unique(y)) == 1:
        return True
    else:
        return False


# 2 Create leaf if the branch is impure based on the task indicated
def create_leaf(y, task):  
    
    # regression
    # create the new leaf by calculating the mean of the dependent variables (continuous)
    if task == "regression":
        new_leaf = np.mean(y)
        
    # classfication
    # create the new leaf that most frequently appears class in the set 
    else:
        class_labels, class_occur = np.unique(y, return_counts=True)
        new_leaf_idx = class_occur.argmax()
        new_leaf = class_labels[new_leaf_idx]
    
    return new_leaf


# 3 List out all possible splits
def possible_splits(X):
    
    # Create an empty dictionary
    possible_splits = {}

    # turn X into nparray for later use
    X = np.array(X)
    # loop through every column (features) in X, collect the unique values of each column
    for idx in range(X.shape[1]):
        values = X[:, idx]

        # adding the unique_values for each feature into the dictionary
        possible_splits[idx] = np.unique(values)
    
    # returns a collection of all possible splits based on features
    return possible_splits


# 4 Determine the Best Split
# 4.1 Calculate evaluation metrics for splits based on the task
def calculate_metrics(y,task):
    # Calculate Gini Impurity in classification task
    if task == 'classification':
        _, class_occur = np.unique(y, return_counts=True)

        # Gini Impurity = 1 - Σ (pi)^2 
        p = class_occur / class_occur.sum()
        metrics = 1 - sum(p**2)

    # Calculate MSE in regression task
    else:
        if len(y) == 0:
            metrics = 0
        else:
            metrics = np.mean((y - np.mean(y))**2)

    return metrics


# 4.2 Calculate overall evaluation metric for the task
def calculate_overall_metric(left, right, task):
    
    # calculate the probability of left leaf and right leaf seperately
    # multiply by the value (gini impurity or MSE) calculate
    overall_metric =  ((len(left) / (len(left) + len(right))) * calculate_metrics(left, task) 
                     + (len(right) / (len(left) + len(right))) * calculate_metrics(right, task))
    
    return overall_metric


# 4.3 Determine whether the feature is categorical or continuous
def determine_type_of_feature(df):
    
    feature_types = []

    for feature in df.columns:
        # if the first element in the column is str then we consider it 'categorical'
        if (isinstance(df[feature][0], str)):
            feature_types.append("categorical")
        else:
            feature_types.append("continuous")
    
    return feature_types


# 4.4 Split data
def split_data(X, split_column, split_value):
    
    split_column_values = X[split_column]
    type_of_feature = determine_type_of_feature(X)[split_column]

    if type_of_feature == "continuous":
        left = X[split_column_values <= split_value]
        right = X[split_column_values >  split_value]
    
    # feature being categorical   
    else:
        left = X[split_column_values == split_value]
        right = X[split_column_values != split_value]
    
    return left, right


# 4.5 Determine the best split from all possible splits based on the evaluation metric according to task
def determine_best_split(X, possible_splits, task):
    
    first_iter = True
    for idx in possible_splits:
        for value in possible_splits[idx]:
            left, right = split_data(X, split_column=idx, split_value=value)
            current_overall_metric = calculate_overall_metric(left, right, task)
            
            if first_iter or current_overall_metric <= best_overall_metric:
                first_iter = False
                
                best_overall_metric = current_overall_metric
                best_split_column = idx
                best_split_value = value
    
    return best_split_column, best_split_value


