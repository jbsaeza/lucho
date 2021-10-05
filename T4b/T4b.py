import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("data.csv").drop(columns="Environmental_risk")

df_training = df.copy()
df_training.dropna(inplace=True)

cat_vars = ['PM2.5']
label_encoder = LabelEncoder()
for i in cat_vars:
    df_training[i] = label_encoder.fit_transform(df_training[i])

training_set, test_set = train_test_split(df_training.copy(), test_size=0.3)
training_set, validation_set = train_test_split(
    training_set.copy(), test_size=0.1)

scaler = StandardScaler()
features = ['Year', 'Month', 'Day', 'O3']

# training_set[features] = scaler.fit_transform(training_set[features])
# validation_set[features] = scaler.transform(validation_set[features])
# test_set[features] = scaler.transform(test_set[features])


def training_and_eval(model, training, eval, features, target):
    model.fit(training[features], training[target])
    predictions = model.predict(eval[features])
    mqe = metrics.mean_squared_error(predictions, eval[target])
    print(f"Error Cuadratico Medio: {mqe: .2}")


target = 'PM2.5'
model = DecisionTreeClassifier()
training_and_eval(model, training_set, test_set, features, target)
model = SVC()
training_and_eval(model, training_set, test_set, features, target)
model = KNeighborsClassifier()
training_and_eval(model, training_set, test_set, features, target)

# El modelo con mejor rendimiento es el Decision Tree Classifier
df_to_pred = df[df["PM2.5"].isnull()]

scaler = StandardScaler()
features = ['Year', 'Month', 'Day', 'O3']

# training_set[features] = scaler.fit_transform(df_training[features])
# test_set[features] = scaler.transform(df_to_pred[features])


model = DecisionTreeClassifier()
model.fit(training_set[features], training_set[target])
predictions = model.predict(test_set[features])
