from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd

classifier = SVC()
pd.options.mode.copy_on_write = True


train_data = pd.read_csv('train.csv')
X = train_data[['Pclass','Age','Fare','Sex']]
X['Sex'] = X.Sex.copy().apply(lambda x : 1.0 if x == 'male' else 2.0).copy()
X = X.fillna(0)
X = X.to_numpy()
y = train_data['Survived']

y = y.to_numpy()

test_data = pd.read_csv('test.csv')
X_test = train_data[['Pclass','Age','Fare','Sex']]
X_test.Sex = X_test.Sex.apply(lambda x : 1.0 if x == 'male' else 2.0)
X_test = X_test.fillna(0)
X_test = X_test.to_numpy()
y_test = train_data['Survived']
y_test = y_test.to_numpy()



classifier.fit(X,y)
predicts = classifier.predict(X_test)

print(classification_report(y_test, predicts))

