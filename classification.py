from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, KBinsDiscretizer
from sklearn.feature_selection import f_classif
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold


def changeNameToSurname(data, column):
    """
    Changes the 'Name' column in the DataFrame to contain only surnames.
    --------
    Input:
    data : pandas.DataFrame
        Input DataFrame
    column : str
        Name of the column to be changed.
    --------
    Output:
    None
    """

    assert isinstance(data, pd.DataFrame), "Input data must be a pandas DataFrame"
    assert column in data.columns, f"Column {column} column not found in the DataFramee"

    for index, row in data.iterrows():
        assert isinstance(row.Name, str), "Name column must be a string"
        surname = row.Name.split(',')[0]
        data.at[index, column] = surname


def dropColumns(data, columns_to_stay):
    """
    Drops columns from the DataFrame that are not in the specified list which is a parameter.
    --------
    Input:
    data : pandas.DataFrame
        Input DataFrame.
    columns_to_stay : list
        List of column names to keep in DataFrame
    --------
    Output:
    None
    """
    assert isinstance(data, pd.DataFrame), "Input data must be a pandas DataFrame"
    assert isinstance(columns_to_stay, list), "columns_to_stay must be a list"

    for column in columns_to_stay:
        assert column in data.columns, f"Column {column} not found in DataFrame"

    for column in data.columns.tolist():
        if column not in columns_to_stay:
            data.drop(column, axis=1, inplace=True)

def computePCA(X, X_test):
    """
    Computes Principal Component Analysis (PCA) on the input data.
    --------
    Input:
    - X: pandas.DataFrame
        Feature dataset for training.
    - X_test: pandas.DataFrame
        Feature dataset for testing.
    --------
    Output:
    - X_pca: numpy array
        Transformed feature dataset for training after PCA.
    - X_pca_test: numpy array
        Transformed feature dataset for testing after PCA.
    - pca_df: pandas.DataFrame
        DataFrame containing the transformed features for training.
    - pca_df_test: pandas.DataFrame
        DataFrame containing the transformed features for testing.
    """
    assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
    assert isinstance(X_test, pd.DataFrame), "X_test must be a pandas DataFrame"
    assert X.columns.to_list() == X_test.columns.to_list(), "X and X_test must have the same columns"

    pca_transformer = PCA(n_components=4)

    X_pca = pca_transformer.fit_transform(X)
    X_pca_test = pca_transformer.transform(X_test)
    pca_df = pd.DataFrame(X_pca)
    pca_df_test = pd.DataFrame(X_pca_test)  
    pca_components = pca_transformer.components_
    print("PCA components: ", pca_components)

    return X_pca, X_pca_test, pca_df, pca_df_test


def computeAnova(normalized_data, target_values):
    """
    Computes ANOVA scores for feature selection based on normalized data
    --------
    Input:
    normalized_data : pandas.DataFrame
        Input DataFrame containing normalized features.
    target_values : one column of pandas.DataFrame, pd.Series
        Target values for classification
    --------
    Output:
    pandas.DataFrame
        DataFrame with ANOVA scores for each feature.
    """
    assert isinstance(normalized_data, pd.DataFrame), "normalized_data must be a pandas DataFrame"
    assert isinstance(target_values, pd.Series), "target_values must be a pandas Series"

    cols = normalized_data.columns.tolist()
    scores_anova, p_vals_anova = f_classif(normalized_data, target_values)
    dataframe = pd.DataFrame(scores_anova, cols)

    return dataframe


def printVariableType(data):
    """
    Makes a DataFrame which shows variable types and percentage of uniques of each column in the data.
    --------
    Inputs:
    data : pandas.DataFrame
    --------
    Outputs:
    variable_df : pandas.DataFrame
        DataFrame with columns indicating variable types and uniquess of each column.
    """
    assert isinstance(data, pd.DataFrame), "Input data must be a pandas DataFrame"

    headers = data.columns.to_list()
    procentage = []
    variable_types = []
    for head in headers:
        procentage.append( round((data[head].nunique()/ len(data[head])),3))
        dtype = data[head].dtype
        if dtype == 'object':
            variable_types.append(str(dtype) + ' (caterogical)')
        elif dtype == 'int64':
            variable_types.append(str(dtype) + ' (discrete)')
        else:
            variable_types.append(str(dtype) + ' (continuous)')

    variable_df = pd.DataFrame({'Column': headers, 'Variable_Type ': variable_types, 'the percentage values of unique values': procentage})
    return variable_df


def normalizeData(data , target_column):
    """
    Normalizes the data using Min-Max scaling from sklearn
    --------
    Input:
    data : pandas.DataFrame
        DataFrame containing the data to be normalized.
    target_column : str
        Name of the target column in the DataFrame which will not be normalized.
    --------
    Output:
    pandas.DataFrame
        Normalized DataFrame.
    """
    assert isinstance(data, pd.DataFrame), "Input data must be a pandas DataFrame"
    assert target_column in data.columns, f"Column {target_column} not found in DataFrame"

    target_column_index = data.columns.get_loc(target_column)

    X = pd.concat([data.iloc[:, :target_column_index], data.iloc[:, target_column_index+1:]], axis=1)

    cols_all = data.columns.tolist()
    cols = cols_all[:target_column_index] + cols_all[target_column_index+1:]

    norm = MinMaxScaler(feature_range=(0,1)).fit(X)
    normalized_data = pd.DataFrame(norm.transform(X), columns=cols)

    return normalized_data

def changeContinuousToDescrite(data):
    """
    Discretizes continuous values in columns using KBinsDiscretizer.
    --------
    Input:
    data : pandas.DataFrame
    --------
    Output:
    pandas.DataFrame
        DataFrame with discretized continuous columns.
    """
    assert isinstance(data, pd.DataFrame), "Input data must be a pandas DataFrame"

    columns_names = data.columns.to_list()
    columns_to_change = []

    for column in columns_names:
        if data[column].dtype == 'float64':
            columns_to_change.append(column)

    kbin = KBinsDiscretizer(n_bins=5, strategy='kmeans', encode='ordinal', subsample=None)
    kbin.fit(data[columns_to_change])
    discretized_data = kbin.transform(data[columns_to_change])
    data[columns_to_change] = discretized_data

    return data


def changeCaterogicalToDescrite(data):
    """
    Encodes categorical columns from dataFrame using LabelEncoder.
    --------
    Input:
    data : pandas.DataFrame
    --------    
    Output:
    None
    """
    assert isinstance(data, pd.DataFrame), "Input data must be a pandas DataFrame"

    columns_names = data.columns.to_list()
    columns_to_change = []

    for column in columns_names:
        if data[column].dtype == 'object':
            columns_to_change.append(column)

    label_encoder = LabelEncoder()
    for column in columns_to_change:
        data[column] = label_encoder.fit_transform(data[column])



def changeNullValuesToMean(data):
    """
    Replaces null values with mean or mode for columns with null percentage > 0.05.
    --------
    Input:
    data : pandas.DataFrame
    --------
    Output:
    None
    """
    assert isinstance(data, pd.DataFrame), "Input data must be a pandas DataFrame"

    columns_names = data.columns.to_list()

    # iterate over all columns
    for column in columns_names:
        value = None

        ## calculate the percentage of null values in the column
        null_percentage = data[column].isnull().mean()

        if null_percentage > 0.05:

            if data[column].dtype == 'object': ##for caterogical type
                value = data[column].mode()[0] ### takes first one
            else:
                value = round(data[column].mean(),2) 

        if value != None:
            for index, row in data.iterrows():
                if pd.isnull(row[column]):
                    data.at[index, column] = value


def changeNullValuesToNewValue(data):
    """
    Replaces null values with new values based on the data type of the column.
    For numerical columns, replaces null values with the maximum value in the column plus 1.
    For categorical columns, replaces null values with the string 'Null'.
    --------
    Input:
    data : pandas.DataFrame
        Input DataFrame containing the data with null values.
    --------
    Output:
    None
    """

    assert isinstance(data, pd.DataFrame), "Input data must be a pandas DataFrame"
    
    columns_names = data.columns.to_list()

    # iterate over all columns
    for column in columns_names:
        value = None

        ## calculate the percentage of null values in the column
        null_percentage = data[column].isnull().mean()

        if null_percentage > 0.05:

            if data[column].dtype == 'object': ##for caterogical type
                value = 'Null' ### takes first one
            else:
                value = round(data[column].max()+1,2) 

        if value != None:
            for index, row in data.iterrows():
                if pd.isnull(row[column]):
                    data.at[index, column] = value

    return data




def getTwoDatasets(data):
    """
    Splits data into two datasets: train and test.
    --------
    Input:
    data : pandas.DataFrame
        Input DataFrame.
    --------
    Output:
    pandas.DataFrame, pandas.DataFrame
        Train and Test DataFrames.
    """
    assert isinstance(data, pd.DataFrame), "Input data must be a pandas DataFrame"

    total_rows = len(data)
    split_index = int(0.6 * total_rows)

    train_df = pd.DataFrame(data.iloc[:split_index])
    test_df = pd.DataFrame(data.iloc[split_index:])

    return train_df, test_df


def getAccuracy( train_X, train_Y, test_X, test_Y):
    """
    Gets accuracy of a Support Vector Classifier (SVC) model.
    --------
    Input:
    train_X : pandas.DataFrame
        Input features of training data.
    train_Y : pandas.DataFrame
        Target labels of training data.
    test_X : pandas.DataFrame
        Input features of test data.
    test_Y : pandas.DataFrame
        Target labels of test data.
    --------
    Output:
    float
        Accuracy score of the model.
    """

    assert isinstance(train_X, pd.DataFrame), "train_X must be a pandas DataFrame"
    assert isinstance(train_Y, pd.DataFrame), "train_Y must be a pandas DataFrame"
    assert isinstance(test_X, pd.DataFrame), "test_X must be a pandas DataFrame"
    assert isinstance(test_Y, pd.DataFrame), "test_Y must be a pandas DataFrame"

    classifier = SVC()

    if not isinstance(train_X, np.ndarray):
        X = train_X.to_numpy()
    else:
        X = train_X

    if not isinstance(train_Y, np.ndarray):
        y = train_Y.to_numpy()
    else:
        y = train_Y

    if not isinstance(test_X, np.ndarray):
        X_test = test_X.to_numpy()
    else:
        X_test = test_X

    if not isinstance(test_Y, np.ndarray):
        y_test = test_Y.to_numpy()
    else:
        y_test = test_Y

    classifier.fit(X,y)
    predicts = classifier.predict(X_test)
    report = classification_report(y_test, predicts)
    accuracy = accuracy_score(y_test, predicts)

    return accuracy
