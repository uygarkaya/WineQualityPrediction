import pandas as pd
import numpy as np
import math

from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Read the data
file_name = '../Dataset/WineQualityNew_R.csv'
dataFrame = pd.read_csv(file_name)
dataset = dataFrame['quality'].values

# We split the data into Training and Test sets
output_node = 64
input_node = len(dataset) - output_node
step_back = input_node
reshape_input = output_node
input_dataset, output_dataset = dataset[:input_node], dataset[input_node:]


# step_back is a number of increment. Also, it is a input layer number
# trainTestXY Method divide the data with number of step_back
def trainTestXY(dataset, step_back):
  x_array, y_array = list(), list()
  for i in range(len(dataset)):
    if(i+step_back) > len(dataset)-1:
      break
    x_array.append(dataset[i:(i+step_back)])
    y_array.append(dataset[i+step_back])
  return np.array(x_array), np.array(y_array)

# Defining the step_back veriable and calling the trainTestXY Function
x_array, y_array = trainTestXY(dataset, step_back)

def visulize_loss(model, title):
  loss = model.history['loss']
  epochs = range(len(loss))
  plt.figure()
  plt.plot(epochs, loss, 'blue', label="Model Loss")
  plt.title(title, fontweight="bold", style="italic")
  plt.xlabel("Epochs", fontweight="bold", style="italic")
  plt.ylabel("Loss", fontweight="bold", style="italic")
  plt.legend()
  plt.show()

def visulize_predict_test(y_actual, forecast):
  plt.figure(figsize=(10, 7))
  plt.plot(y_actual ,label='Actual Data')
  plt.plot(forecast, color='red',label='Predicted Data')
  plt.legend(loc='upper right')
  plt.show()

def calculate_error(y_true, y_predict):
  mse = mean_squared_error(y_true, y_predict)
  mae = mean_absolute_error(y_true, y_predict)
  mape = mean_absolute_percentage_error(y_true, y_predict)
  rmse = math.sqrt(mean_squared_error(y_true, y_predict))
  return mse, mae, mape, rmse

# Hyperparameter Selection
# Selecting the batch size effect the run-time
# When there is more than 1 Hidden Layer, we can give the Learning Rate high first and then lower it.

batch_size = 256
num_epochs = 200
learning_rate = 0.001
embedding_dim = 1
# dropout_rate = 0.2 or 0.5

# Defining MLP Model for Zero Hidden Layer
model = Sequential()
model.add(Dense(units = embedding_dim, activation='relu', input_dim=input_node))
opt = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=opt, loss='mape', metrics=['accuracy'])
model.summary()

# Before the fit(), we should make the numpy array
x_array = np.array(x_array)
y_array = np.array(y_array)

fit_model = model.fit(x_array, y_array, epochs=num_epochs, batch_size=batch_size, verbose=2)

# Predict the future value
input = x_array
input = input.reshape((reshape_input, input_node))
prediction = model.predict(input)

print('Zero Hidden Layer Prediction:\n', prediction)
visulize_loss(fit_model, 'Zero Hidden Layer - Fitting Model Loss')
visulize_predict_test(y_array, prediction)

for index in range(len(y_array)):
  mse, mae, mape, rmse = calculate_error(y_array, prediction)

print('MSE:', mse)
print('MAE:', mae)
print('MAPE:', mape)
print('RMSE:', rmse)
