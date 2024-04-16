from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


def normalizeData(data , target_column):
    target_column_index = data.columns.get_loc(target_column)

    X = pd.concat([data.iloc[:, :target_column_index], data.iloc[:, target_column_index+1:]], axis=1)

    cols_all = data.columns.tolist()
    cols = cols_all[:target_column_index] + cols_all[target_column_index+1:]

    norm = MinMaxScaler(feature_range=(0,1)).fit(X)
    normalized_data = pd.DataFrame(norm.transform(X), columns=cols)

    return normalized_data


def changeCaterogicalToDescrite(data):
    
    columns_names = data.columns.to_list()
    columns_to_change = []

    for column in columns_names:
        if data[column].dtype == 'object':
            columns_to_change.append(column)

    print('columns to changed: ',columns_to_change)

    label_encoder = LabelEncoder()
    for column in columns_to_change:
        data[column] = label_encoder.fit_transform(data[column])

    return data



def changeNullValuesToMean(data):
    columns_names = data.columns.to_list()

    # iterate over all columns
    for column in columns_names:
        value = None

        ## calculate the percentage of null values in the column
        null_percentage = data[column].isnull().mean()

        if null_percentage > 0.05:
            print(column, 'was with NaN values')

            if data[column].dtype == 'object': ##for caterogical type
                value = data[column].mode()[0] ### takes first one
            else:
                value = round(data[column].mean(),2) 

        if value != None:
            for index, row in data.iterrows():
                if pd.isnull(row[column]):
                    data.at[index, column] = value

    return data


def getTwoDatasets(data):

    total_rows = len(data)
    split_index = int(0.6 * total_rows)

    train_df = pd.DataFrame(data.iloc[:split_index])
    test_df = pd.DataFrame(data.iloc[split_index:])

    return train_df, test_df

def testing(): ### at the end, testowanie różńych setów, cross validation !
    ...

def getAccuracy( train_X, train_Y, test_X, test_Y):

    classifier = SVC()

    X = train_X.to_numpy()
    y = train_Y.to_numpy()

    X_test = test_X.to_numpy()
    y_test = test_Y.to_numpy()

    classifier.fit(X,y)
    predicts = classifier.predict(X_test)
    report = classification_report(y_test, predicts)
    accuracy = accuracy_score(y_test, predicts)

    return accuracy
