from pandas import read_csv, cut, DataFrame, get_dummies, concat
from sklearn.preprocessing import MinMaxScaler


def import_data():
    """Import dataset and remove row numbers column
    :return: dataframe
    """
    df = read_csv('datasets/cs-training.csv')
    df = df.iloc[:, [1, 2, 3, 4, 5, 6, 7, 9, 10, 11]]
    return df


def clean_data(df):
    """Clean dataframe and impute missing values by mean and mode
    :param df: input dataframe
    :return: clean dataframe
    """
    df['ages'] = cut(x=df['age'], bins=[0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 159],
                     labels=['0s', '10s', '20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s', '100s'])
    mode_age = int(df['NumberOfDependents'].mode())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(mode_age)
    mean_income = int(df['MonthlyIncome'].mean())
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(mean_income)
    df.dropna(inplace=True)
    missing = (df.isnull().values.any())
    if missing:
        print(df.isnull().sum())
    return df


def normalize_columns(df, colnames):
    """Performs Normalization using MinMaxScaler class in Sckikit-learn"""
    scaler = MinMaxScaler()
    for col in colnames:
        # Create x, where x the 'scores' column's values as floats
        x = df.loc[:, [col]]
        x = df[[col]].values.astype(float)
        # Create a minimum and maximum processor object
        # Create an object to transform the data to fit minmax processor
        x_scaled = scaler.fit_transform(x)
        # Run the normalizer on the dataframe
        df[col] = DataFrame(x_scaled)
    print(f'''Normalized Columns: {colnames} using MinMaxScaler.''')
    return df


def one_hot_encode(df, colnames):
    """This function performs one-hot encoding of the columns
    :param df: input df
    :param colnames: columns to be one-hot encoded
    :return: dataframe
    """
    for col in colnames:
        oh_df = get_dummies(df[col], prefix=col)
        df = concat([oh_df, df], axis=1)
        df = df.drop([col], axis=1)
    return df


if __name__ == '__main__':
    df = import_data()
    df = clean_data(df)
    df = normalize_columns(df, colnames=['age', 'MonthlyIncome'])
    df = one_hot_encode(df, colnames=['NumberOfDependents', 'ages'])
    pass
