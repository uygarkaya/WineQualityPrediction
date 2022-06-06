import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor, kneighbors_graph
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

file_name = '../Dataset/WineQualityNew_R.csv'
dataFrame = pd.read_csv(file_name)

df1 = dataFrame[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                 'pH', 'sulphates', 'alcohol', 'type_red', 'type_white', 'quality']]
df2 = dataFrame[['alcohol', 'density', 'volatile acidity',
                 'chlorides', 'type_red', 'type_white', 'quality']]
df3 = dataFrame[['fixed acidity', 'free sulfur dioxide',
                 'total sulfur dioxide', 'sulphates', 'residual sugar', 'pH', 'quality']]
df4 = dataFrame[['alcohol', 'pH', 'density', 'residual sugar',
                 'volatile acidity', 'sulphates', 'quality']]

X1_train, X1_test, Y1_train, Y1_test = train_test_split(
    df1.iloc[:, 0:13].values, df1.loc[:, "quality"].values, test_size=0.25, random_state=147)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(
    df2.iloc[:, 0:6].values, df2.loc[:, "quality"].values, test_size=0.25, random_state=147)
X3_train, X3_test, Y3_train, Y3_test = train_test_split(
    df3.iloc[:, 0:6].values, df3.loc[:, "quality"].values, test_size=0.25, random_state=147)
X4_train, X4_test, Y4_train, Y4_test = train_test_split(
    df4.iloc[:, 0:6].values, df4.loc[:, "quality"].values, test_size=0.25, random_state=147)

x_train = [X1_train, X2_train, X3_train, X4_train]
y_train = [Y1_train, Y2_train, Y3_train, Y4_train]

x_test = [X1_test, X2_test, X3_test, X4_test]
y_test = [Y1_test, Y2_test, Y3_test, Y4_test]

scaler = []
for index in range(4):
    scaler.append(MinMaxScaler(feature_range=(0, 1)))

x_train_scaled = []
for train_data_index in range(len(x_train)):
    train_data = scaler[train_data_index].fit_transform(
        x_train[train_data_index])
    x_train_scaled.append(train_data)

x_test_scaled = []
for test_data_index in range(len(x_test)):
    test_data = scaler[test_data_index].transform(x_test[test_data_index])
    x_test_scaled.append(test_data)

model = []
for index in range(4):
    model.append(KNeighborsRegressor(n_neighbors=5))

kneighbors_regressor = []
for index in range(len(x_train_scaled)):
    regression = model[index].fit(x_train_scaled[index], y_train[index])
    kneighbors_regressor.append(regression)

prediction = []
for index in range(len(kneighbors_regressor)):
    prediction.append(kneighbors_regressor[index].predict(x_test_scaled[index]))

score = []
for index in range(len(kneighbors_regressor)):
    score.append(kneighbors_regressor[index].score(
        x_test_scaled[index], y_test[index]))

for index, value in enumerate(score):
    print(f'score{index+1}: {value}')


def calculate_error_rate(kneighbors_regressor, y_test, prediction):
    mse = []
    for index in range(len(kneighbors_regressor)):
        mse.append(mean_squared_error(y_test[index], prediction[index]))

    mae = []
    for index in range(len(kneighbors_regressor)):
        mae.append(mean_absolute_error(y_test[index], prediction[index]))

    return mse, mae

mse, mae = calculate_error_rate(kneighbors_regressor, y_test, prediction)
for index, value in enumerate(mse):
    print(f'MSE{index+1}: {value}')
for index, value in enumerate(mae):
    print(f'MAE{index+1}: {value}')


def visulize_error_rate(x_label, y_label, models, error_rate):
    plt.xlabel(x_label, fontweight="bold", style="italic")
    plt.ylabel(y_label, fontweight="bold", style="italic")
    plt.scatter(models, error_rate, s=75, marker='o', color='b')
    plt.show()


visulize_error_rate('Models', 'Mean Squared Error', [
                    "M{}".format(index+1) for index in range(4)], mse)
visulize_error_rate('Models', 'Mean Absolute Error', [
                    "M{}".format(index+1) for index in range(4)], mae)
