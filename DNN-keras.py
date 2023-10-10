import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

path = r"C:\Users\Lucian\Desktop\Masterat CTI\Machine Learning\mushrooms.csv"
df = pd.read_csv(path)
features = list(df.keys())
features_X = features[1:]
feature_y = [features[0]]
df_X = df[features_X].copy()
df_y = df[feature_y].copy()

le_X = dict()
for col in df_X.columns:
    le_X[col] = LabelEncoder()
    df_X[col] = le_X[col].fit_transform(df_X[col])
le_y = LabelEncoder()
df_y[feature_y[0]] = le_y.fit_transform(df_y[feature_y[0]])

X = df_X.values
y = df_y.values.ravel()

le = dict()

def transform_new_data(new_data):
    new_data_list = []
    for index, item in enumerate(new_data):
        print(index, item)
        new_data_list.append(list(le_X[features_X[index]].transform([item]))[0])
    return new_data_list

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40)

model = keras.models.Sequential([
    keras.layers.Dense(22, activation='relu'),
    keras.layers.Dense(11, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_test, y_test, batch_size = 32, epochs = 20)
print(model.summary())
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print(test_acc)