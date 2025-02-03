import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

DATA_PATH = "dataset/Training.csv"
# axis = 0 means drop rows that contains 'NaN'
# axis = 1 means drop cols that contains 'NaN'
data = pd.read_csv(DATA_PATH).dropna(axis=1)

# check whether the target column is balanced
# value_counts returns a series.
# 索引是出現過的唯一值，對應的值是這些唯一值出現的次數，按次數降冪排列。 
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease":disease_counts.index,
    "Counts":disease_counts.values
})

plt.figure(figsize=(18, 8))
sns.barplot(x="Disease", y="Counts", data=temp_df)
#plt.sticks is used to adjust the form of x axis
plt.xticks(rotation=90)
plt.show()

# Prognosis column is of object datatype, we need to convert it to the 
# numerical datatype
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

# split the data for training and testing 
# train:test = 8:2
# x contains all rows, and all cols except last col(the target variable)
# y contains the last column
X = data.iloc[:,:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24
)

print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test; {X_test.shape}, {y_test.shape}")


def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))

# initialize models
models = {
    "SVC":SVC(), 
    "Gaussian NB":GaussianNB(),
    "Random Forest":RandomForestClassifier(random_state=18)
}

# produce cross validation scores for models
for model_name in models:
    model = models[model_name]
    scores = cross_val_score(model, X, y, cv=10, n_jobs=-1,scoring=cv_scoring)
    print("==" * 30)
    print(model_name)
    print(f"Scores: {scores}")
    print(f"Mean score: {np.mean(scores)}")
