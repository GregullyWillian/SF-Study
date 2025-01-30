import pandas as pd

def import_datasets(name):
    if name == 'census':
        df_00  = pd.read_csv("./DataSets/census/adult.data", 
                        names=["Age", "Workclass", "fnlwgt", "Education", "EducationNum", "MartialStatus",
                        "Occupation", "Relationship", "Race", "Sex", "CapitalGain", "CapitalLoss",
                        "Hoursperweek", "Country", "Target"],
                        sep=r'\s*,\s*',
                        engine='python',
                        na_values="?")
        df_01 = pd.read_csv("./DataSets/census/adult.test", 
                        names=["Age", "Workclass", "fnlwgt", "Education", "EducationNum", "MartialStatus",
                        "Occupation", "Relationship", "Race", "Sex", "CapitalGain", "CapitalLoss",
                        "Hoursperweek", "Country", "Target"],
                        sep=r'\s*,\s*',
                        engine='python',
                        na_values="?")
        df_census = pd.concat([df_00, df_01], axis=0, ignore_index=True)
        df_census['Target'] = df_census['Target'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})
        df_census = df_census.dropna()
        return df_census
    if name == 'census_train':
        df_census = pd.read_csv("./DataSets/census/adult.data", 
                        names=["Age", "Workclass", "fnlwgt", "Education", "EducationNum", "MartialStatus",
                        "Occupation", "Relationship", "Race", "Sex", "CapitalGain", "CapitalLoss",
                        "Hoursperweek", "Country", "Target"],
                        sep=r'\s*,\s*',
                        engine='python',
                        na_values="?")
        df_census['Target'] = df_census['Target'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})
        df_census = df_census.dropna()
        return df_census
    elif name == 'census_val':
        df_census = pd.read_csv("./DataSets/census/adult.test", 
                        names=["Age", "Workclass", "fnlwgt", "Education", "EducationNum", "MartialStatus",
                        "Occupation", "Relationship", "Race", "Sex", "CapitalGain", "CapitalLoss",
                        "Hoursperweek", "Country", "Target"],
                        sep=r'\s*,\s*',
                        engine='python',
                        na_values="?")
        df_census['Target'] = df_census['Target'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})
        df_census = df_census.dropna()
        return df_census
    elif name == 'bank':
        df_bank = pd.read_csv("DataSets/bank+marketing/bank/bank-full.csv",
                    names=["age","job","marital","education","default","balance","housing","loan","contact","day",
                    "month","duration","campaign","pdays","previous","poutcome","Target"],
                    sep=r'\s*,\s*',
                    engine='python',
                    na_values="?")
        df_bank = df_bank.dropna()
        return df_bank
    elif name == 'heart':
        df_heart = pd.read_csv("DataSets/heart/heart.csv",
            names=["age", "sex", "cp", "trtbps", "chol", "fbs", "restecg",
               "thalachh", "exng", "oldpeak", "slp", "caa", "thall" ,"Target"],
                sep=r'\s*,\s*',
                engine='python',
                na_values="?")
        df_heart = df_heart.dropna()
        return df_heart
    elif name == 'mush':
        df_mush = pd.read_csv(
        "DataSets/mushroom/agaricus-lepiota.csv",
        names=["Target", "capshape", "capsurface", "capcolor", "bruises", "odor", "gillattachment",
               "gillspacing", "gillsize", "gillcolor", "stalkshape", "stalkroot", "stalksurfaceabovering",
               "stalksurfacebelowring", "stalkcolorabovering", "stalkcolorbelowring", "veiltype",
               "veilcolor", "ringnumber", "ringtype", "sporeprintcolor", "population", "habitat"],
                sep=r'\s*,\s*',
                engine='python',
                na_values="?")
        df_mush = df_mush.dropna()
        return df_mush
    elif name == 'rice':
        df_rice = pd.read_csv(
            "DataSets/rice+cammeo+and+osmancik/Rice_Cammeo_Osmancik.csv",
            names=["Area", "Perimeter", "Major_Axis_Length", "Minor_Axis_Length", "Eccentricity", "Convex_Area",
                "Extent", "Target"],
                    sep=r'\s*,\s*',
                    engine='python',
                    na_values="?")
        df_rice = df_rice.dropna()
        return df_rice
    elif name == 'shop':
        df_shop = pd.read_csv(
        "DataSets/online_shoppers/online_shoppers_intention.csv",
        names=["Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated",
            "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month", "OperatingSystems",
            "Browser", "Region", "TrafficType", "VisitorType", "Weekend", "Target"],
                sep=r'\s*,\s*',
                engine='python',
                na_values="?")
        df_shop = df_shop.dropna()
        return df_shop
    elif name == 'agro':
        df_agro = pd.read_csv('./Teste_final_CIA-Agro.csv')
        df_agro = df_agro.dropna()
        return df_agro
    elif name == 'churn':
        df_churn = pd.read_csv(
            "DataSets/Churn_Modelling.csv",
            names=['RowNumber','CustomerId','Surname','CreditScore','Geography','Gender','Age',
                   'Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember',
                   'EstimatedSalary','Target'],
                    sep=r'\s*,\s*',
                    engine='python',
                    na_values="?")
        df_churn = df_churn.dropna()
        return df_churn
