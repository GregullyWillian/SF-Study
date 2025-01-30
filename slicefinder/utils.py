from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, log_loss
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from skopt import BayesSearchCV
from collections import Counter
import pandas as pd
import numpy as np
import pickle
import time

def encoder(data, name):
    encoders = {}
    for column in data.columns:
        if data.dtypes[column] == object:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            encoders[column] = le
            
    X, y = data[data.columns.difference(["Target"])], data["Target"]
    pickle.dump(encoders, open(f"encoders_{name}.pkl", "wb"), protocol=2)
    return data, X, y, encoders
    
def model_train(X, y):
    param_dist = {
    'n_estimators': Integer(10, 200),
    'max_depth': Integer(1, 20),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 20),
    'max_features': Categorical(['sqrt', 'log2', None]),
    'bootstrap': Categorical([True, False]),
    }

    rf = RandomForestClassifier(random_state=42)

    opt = BayesSearchCV(
        estimator=rf,
        search_spaces=param_dist,
        n_iter=32,
        cv=5,
        random_state=42,
        n_jobs=-1
    )

    opt.fit(X, y)

    best_params = opt.best_params_

    rf_optimized = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        bootstrap=best_params['bootstrap'],
        random_state=42
    )
    
    return rf_optimized

def region_inspect(df, region):
    
    return df.query(region)
    
def reason_in_class(y):
    count = Counter(y)
    majority_class = max(count.values())
    minority_class = min(count.values())
    
    return majority_class / minority_class

def apply_SMOTE_region(df, region):
    df_lenght = len(df)
    
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    
    if 1 / reason_in_class(df.query(region)['Target']) > 0.10:
        region_remove = df.query(region)
        region_remove_idx = df.query(region).index
    else :
        return "Não aplicavel"
    
    df.drop(region_remove_idx, errors='ignore', inplace=True)
    y_slice = region_remove['Target']
    X_slice = region_remove.drop(columns='Target')
    
    start_time = time.time()
    X_slice_smote, y_slice_smote = smote.fit_resample(X_slice, y_slice)
    region_SMOTE = (pd.concat([pd.DataFrame(X_slice_smote, columns=X_slice.columns),
                               pd.Series(y_slice_smote, name='Target')], axis=1))
    end_time = time.time()
    execution_time = end_time - start_time
    
    df_new = pd.concat([df, region_SMOTE], ignore_index=True)
    instance_add = len(df_new) - df_lenght
    
    y_SMOTE_data = df_new['Target']
    X_SMOTE_data = df_new.drop(columns='Target')
    
    return instance_add, execution_time, X_SMOTE_data, y_SMOTE_data

def region_smote(df, slices):
    df_lenght = len(df)
    slice_remove = []
    slice_remove_idx = [] 
    X_slice = []
    y_slice = []

    region_SMOTE = []
    smote = SMOTE(k_neighbors=1, sampling_strategy=0.9, random_state=42)

    for slice_condition in slices:
        if 1 / reason_in_class(df.query(slice_condition)['Target']) > 0.10:
            slice_remove.append(df.query(slice_condition))
            slice_remove_idx.append(df.query(slice_condition).index)

    for i in range(len(slice_remove_idx)):
        df.drop(slice_remove_idx[i], errors='ignore', inplace=True)
        X_slice.append(slice_remove[i].drop(columns='Target'))
        y_slice.append(slice_remove[i]['Target'])
        
        X_slice_smote, y_slice_smote = smote.fit_resample(X_slice[i], y_slice[i])
        region_SMOTE.append(pd.concat([pd.DataFrame(X_slice_smote, columns=X_slice[i].columns), pd.Series(y_slice_smote, name='Target')], axis=1))

    df_SMOTE_slice = pd.concat(region_SMOTE + [df], ignore_index=True)
    df_SMOTE_slice.drop_duplicates(inplace=True)
    
    df_SMOTE_lenght = len(df_SMOTE_slice)
    instances_add = df_SMOTE_lenght - df_lenght
    X_SMOTE_data = df_SMOTE_slice.drop(columns='Target')
    y_SMOTE_data = df_SMOTE_slice['Target']
    
    return instances_add, X_SMOTE_data, y_SMOTE_data

def region_remove(df, region):
    region = df.query(region).index
    df.drop(region, inplace=True)
    
    y = df['Target']
    X = df.drop(columns=['Target'])
    
    return X, y


from imblearn.over_sampling import BorderlineSMOTE

def apply_SMOTE_region_bismote(df, region):
    df_lenght = len(df)
    
    blsmote = BorderlineSMOTE(sampling_strategy='minority', kind='borderline-1', random_state=42)
    
    if 1 / reason_in_class(df.query(region)['Target']) > 0.10:
        region_remove = df.query(region)
        region_remove_idx = df.query(region).index
    else :
        return "Não aplicavel"
    
    df.drop(region_remove_idx, errors='ignore', inplace=True)
    y_slice = region_remove['Target']
    X_slice = region_remove.drop(columns='Target')
    
    start_time = time.time()
    X_slice_smote, y_slice_smote = blsmote.fit_resample(X_slice, y_slice)
    region_SMOTE = (pd.concat([pd.DataFrame(X_slice_smote, columns=X_slice.columns),
                               pd.Series(y_slice_smote, name='Target')], axis=1))
    end_time = time.time()
    execution_time = end_time - start_time
    
    df_new = pd.concat([df, region_SMOTE], ignore_index=True)
    instance_add = len(df_new) - df_lenght
    
    y_SMOTE_data = df_new['Target']
    X_SMOTE_data = df_new.drop(columns='Target')
    
    return instance_add, execution_time, X_SMOTE_data, y_SMOTE_data

def apply_SMOTE_region(df, region):
    df_lenght = len(df)
    
    blsmote = SMOTE(sampling_strategy='auto', random_state=42)
    
    if 1 / reason_in_class(df.query(region)['Target']) > 0.10:
        region_remove = df.query(region)
        region_remove_idx = df.query(region).index
    else :
        return "Não aplicavel"
    
    df.drop(region_remove_idx, errors='ignore', inplace=True)
    y_slice = region_remove['Target']
    X_slice = region_remove.drop(columns='Target')
    
    start_time = time.time()
    X_slice_smote, y_slice_smote = blsmote.fit_resample(X_slice, y_slice)
    region_SMOTE = (pd.concat([pd.DataFrame(X_slice_smote, columns=X_slice.columns),
                               pd.Series(y_slice_smote, name='Target')], axis=1))
    end_time = time.time()
    execution_time = end_time - start_time
    
    df_new = pd.concat([df, region_SMOTE], ignore_index=True)
    instance_add = len(df_new) - df_lenght
    
    y_SMOTE_data = df_new['Target']
    X_SMOTE_data = df_new.drop(columns='Target')
    
    return instance_add, execution_time, X_SMOTE_data, y_SMOTE_data