import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def data_exploration(df):
    print("Sample of the data: \n", df.head(1))
    print(df.info())
    print("Missing values in each column: \n", df.isna().sum())
    print("Distribution of target classes: \n", df['target'].value_counts())
    print("Description of the data: \n", df.describe())
    