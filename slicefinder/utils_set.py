import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def import_datasets(name):
    if name == 'census':
        adult_data_train = pd.read_csv("data/adult.data",
        names=[
            "Age", "Workclass", "fnlwgt", "Education", "EducationNum", "MartialStatus",
            "Occupation", "Relationship", "Race", "Sex", "CapitalGain", "CapitalLoss",
            "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

        # drop nan values
        adult_data_train = adult_data_train.dropna()

        # Encode categorical features
        encoders = {}
        for column in adult_data_train.columns:
            if adult_data_train.dtypes[column] == np.object_:
                le = LabelEncoder()
                adult_data_train[column] = le.fit_transform(adult_data_train[column])
                encoders[column] = le
                print(column, le.classes_, le.transform(le.classes_))

        X_train, y_train = adult_data_train[adult_data_train.columns.difference(["Target"])], adult_data_train["Target"]
        
        adult_data_test = pd.read_csv("data/adult.data",
        names=[
            "Age", "Workclass", "fnlwgt", "Education", "EducationNum", "MartialStatus",
            "Occupation", "Relationship", "Race", "Sex", "CapitalGain", "CapitalLoss",
            "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

        # drop nan values
        adult_data_test = adult_data_test.dropna()

        # Encode categorical features
        encoders = {}
        for column in adult_data_test.columns:
            if adult_data_test.dtypes[column] == np.object_:
                le = LabelEncoder()
                adult_data_test[column] = le.fit_transform(adult_data_test[column])
                encoders[column] = le
                print(column, le.classes_, le.transform(le.classes_))

        X_val, y_val = adult_data_test[adult_data_test.columns.difference(["Target"])], adult_data_test["Target"]
        
        return X_train, y_train, X_val, y_val