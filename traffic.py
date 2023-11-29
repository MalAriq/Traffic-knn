import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("Traffic.csv")

# Assuming df is your DataFrame
# Define a mapping from day names to numerical values
day_mapping = {
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
    "Sunday": 7,
}
traffic_situation_mapping = {"low": 0, "normal": 1, "high": 2, "heavy": 3}

# Apply the mapping to the 'Day of the week' column
data["Day of the week"] = data["Day of the week"].map(day_mapping)
data["Traffic Situation"] = data["Traffic Situation"].map(traffic_situation_mapping)

# Assuming df is your DataFrame
data["Day of the week"] = data["Day of the week"].astype(int)

# Assuming 'Time' is the name of the column in your DataFrame
data["Time"] = (
    pd.to_datetime(data["Time"]).dt.hour * 3600
    + pd.to_datetime(data["Time"]).dt.minute * 60
    + pd.to_datetime(data["Time"]).dt.second
)

traffic_df = data.drop("Date", axis=1)

# Split Data
array = traffic_df.values
X = array[:, 0:7]
Y = array[:, 7]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
    X, Y, test_size=validation_size, random_state=seed
)

# Test options and evaluation metric
seed = 7
scoring = "acuracy"

# Memanggil fungsi Naive Bayes
nb = GaussianNB()

# Memasukkan data training pada fungsi klasifikasi naive bayes
data_training = nb.fit(X, Y)

import pickle

pickle.dump(data_training, open("model.pkl", "wb"))
