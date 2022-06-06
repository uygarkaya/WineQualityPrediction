import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from tensorflow import keras
import tensorflow_addons as tfa
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

learning_rate = 0.005

# Read the data
file_name = '../Dataset/WineQualityNew_C.csv'
dataFrame = pd.read_csv(file_name)

df1 = dataFrame[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                 'pH', 'sulphates', 'alcohol', 'type_red', 'type_white', 'quality']]

# Split the data into training and testing data with using the independent variables(y) and dependent variable(x)
X_train, X_test, Y_train, Y_test = train_test_split(
    df1.iloc[:, 0:13].values, df1.loc[:, "quality"].values, test_size=0.25, random_state=147)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = []
models.append(MLPClassifier(random_state=1, max_iter=300))

# slp = keras.models.Sequential()
# slp.add(keras.layers.Dense(units=3, input_dim=13, activation='sigmoid'))
# slp_opt = tfa.optimizers.AdamW(
#     learning_rate=learning_rate, weight_decay=0.0001,
# )
# slp.compile(optimizer=slp_opt, loss='mape', metrics=['accuracy'])
# models.append(slp)

# mlp = keras.models.Sequential()
# mlp.add(keras.layers.Dense(units=4, input_dim=13, activation='relu'))
# mlp.add(keras.layers.Dense(units=3, activation='sigmoid'))
# mlp_opt = tfa.optimizers.AdamW(
#     learning_rate=learning_rate, weight_decay=0.0001,
# )
# mlp.compile(optimizer=mlp_opt, loss='mape', metrics=['accuracy'])
# models.append(mlp)

for model in models:
    model.fit(X_train_scaled, Y_train)
    #model.fit(X_train_scaled, Y_train, epochs=300, batch_size=256)
    #model.evaluate(X_test_scaled, Y_test)

# mlps = []
# for index in range(len(x_train_scaled)):
#     model = models[index].fit(x_train_scaled[index],
#                               y_train[index], epochs=200, batch_size=256)
#     mlps.append(model)

prediction = []
for index in range(len(models)):
    prediction.append(models[index].predict(X_test_scaled))

accuracy = []
for index in range(len(models)):
    accuracy.append(models[index].score(
        X_test_scaled, Y_test))

# Printing the each score of the models
for index, value in enumerate(accuracy):
    print(f'accuracy {index+1}: {value}')

for index, prediction in enumerate(prediction):
    cm = confusion_matrix(Y_test, prediction, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Low", "Medium", "High"])
    print(cm)
    disp.plot()
    plt.show()
    print(f"CR of models{index+1}:\n",
          classification_report(Y_test, prediction))
