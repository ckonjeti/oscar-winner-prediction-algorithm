import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np


df = pd.read_csv('movie_data.csv', index_col=0)

features = list(df.columns.values)
features.remove('winner')
features.remove('film')
#print(features)
X = df[features]
y = df['winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(7,)),
    keras.layers.Dense(7, activation=tf.nn.relu),
	keras.layers.Dense(7, activation=tf.nn.relu),
	#keras.layers.Dense(2, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=1)

#test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

#print('\nTest accuracy:', test_acc)

pred = model.predict(X_test)
#print

winners = 0
for actual, predicted in zip(y_test, pred):
    if actual == 1 and predicted > 0.5:
        winners += 1
print(winners/len(pred))
