import numpy as np
import pandas as pd

# Helper Functions
def create_leaf(y, ml_task):
    
    if ml_task == "regression":
        leaf = float(np.mean(y))
    else:
        counts = y.value_counts().reset_index()
        leaf = counts.iloc[0,0]
    
    return leaf


def get_potential_splits(data):
    
    X = data.drop(columns='target')
    potential_splits = {}
    columns = X.columns.tolist()
    for column in columns:

        values = X[[column]]
        unique_values = np.unique(values)
        
        potential_splits[column] = unique_values - 1
    
    return potential_splits


def calculate_gini(y):
    
    counts = y.value_counts().to_numpy()
    probabilities = counts / counts.sum()
    gini = np.sum(probabilities*(1-probabilities))
     
    return gini


def calculate_mse(y):
    
    if len(y) == 0:
        mse = 0
    else:
        mse = np.mean((y - np.mean(y)) **2)
    
    return mse


def total_impurity(data_left, data_right, metric_function):\

    n = len(data_left) + len(data_right)
    prop_left = len(data_left) / n
    prop_right = len(data_right) / n

    overall_metric =  (prop_left * metric_function(data_left['target']) 
                     + prop_right * metric_function(data_right['target']))
    
    return overall_metric


def split_data(data, column_types, split_column, split_value):
    
    type_of_feature = column_types[split_column]

    if type_of_feature == "continuous":
        data_left = data[data[split_column] <= split_value]
        data_right = data[data[split_column] >  split_value]
    
    else:
        data_left = data[data[split_column] == split_value]
        data_right = data[data[split_column] != split_value]
    
    return data_left, data_right


def determine_best_split(data, column_types, potential_splits, ml_task):

    best_overall_metric = np.inf
    for column, splits in potential_splits.items():
        for split in splits:
            
            data_left, data_right = split_data(data, column_types, split_column=column, split_value=split)
            
            if ml_task == "regression":
                node_impurity = total_impurity(data_left, data_right, metric_function=calculate_mse)
            else:
                node_impurity = total_impurity(data_left, data_right, metric_function=calculate_gini)
            
            if node_impurity <= best_overall_metric:
                best_overall_metric = node_impurity
                best_split_column = column
                best_split_value = split
    
    return best_split_column, best_split_value


# Main Algorithm
def decision_tree_algorithm(df, column_types, ml_task, min_samples=2, max_depth=5):
    
    leaves = []
    path = 'root'
    datasets = [(df,path)]
    split_conditions = []
    for current_depth in range(max_depth+1):
        next_set = []
        for dataset in datasets:
            data = dataset[0]
            path = dataset[1]
            
            if (len(data.target.unique()) == 1) or (len(data) < min_samples):
                leaf = create_leaf(data[['target']], ml_task)
                leaves.append((path,leaf))
                continue

            potential_splits = get_potential_splits(data)
            split_column, split_value = determine_best_split(data, column_types, potential_splits, ml_task)
            data_left, data_right = split_data(data, column_types, split_column, split_value)

            if len(data_left) == 0 or len(data_right) == 0:
                leaf = create_leaf(data[['target']], ml_task)
                leaves.append((path,leaf))
                continue
            print(len(data_left),len(data_right))
            split_conditions.append((path,split_column,split_value))
            next_set.append((data_left,path+',l'))
            next_set.append((data_right,path+',r'))

        datasets = next_set

    for dataset in datasets:
        data = dataset[0]
        path = dataset[1]
        leaf = create_leaf(data[['target']], ml_task)
        leaves.append((path,leaf))

    return leaves, split_conditions


# Make predictions & Calculate accuracy/MSE

def make_predictions(df, column_types, leaves, split_conditions):

    df['path'] = 'root'
    df['value'] = 0
    
    for split_condition in split_conditions:
        path = split_condition[0]
        column = split_condition[1]
        value = split_condition[2]

        if column_types[column] == "continuous":
            df.loc[(df['path']==path)&(df[column]<= value),'path'] = path+',l'
            df.loc[(df['path']==path)&(df[column]> value),'path'] = path+',r'
        else:
            df.loc[(df['path']==path)&(df[column]== value),'path'] = path+',l'
            df.loc[(df['path']==path)&(df[column]!= value),'path'] = path+',r'

    df['prediction'] = df['path'].map(dict(leaves))

    return df


def calculate_accuracy(df, column_types, ml_task, leaves, split_conditions):
    predictions = make_predictions(df, column_types, leaves, split_conditions).prediction
    
    if ml_task == 'regression':    
        predictions_array = predictions.values
        target_array = df.target.values
        metric = sum((predictions_array - target_array)**2) / len(predictions_array)
        
    else:
        predictions_correct = predictions == df.target
        metric = predictions_correct.mean()
    
    return  metric