import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Read the data
file_name = '../Dataset/WineQualityNew.csv'
dataFrame = pd.read_csv(file_name)

# in df1, we use all the data what we pre-process
# in df2, we use 6 different dependent variables data with the highest correlation with the independent variable 
# in df3, we use 6 different dependent variables data with the lowest correlation with the independent variable 
# in df4, we use 6 different dependent variables data that have one highest and one lowest correlation with the independent variable
df1 = dataFrame.iloc[:, 1:]
df2 = dataFrame[['alcohol', 'density', 'volatile acidity', 'chlorides', 'citric acid', 'fixed acidity', 'quality']]
df3 = dataFrame[['fixed acidity', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates', 'residual sugar', 'pH', 'quality']]
df4 = dataFrame[['alcohol', 'pH', 'density', 'residual sugar', 'volatile acidity', 'sulphates', 'quality']]

# split the data into training and testing data with using the independent variables(y) and dependent variable(x)
X1_train, X1_test, Y1_train, Y1_test = train_test_split(df1.iloc[:,0:11].values, df1.iloc[:,-1].values, test_size=0.25, random_state=147)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(df2.iloc[:,0:6].values, df2.iloc[:,-1].values, test_size=0.25, random_state=147)
X3_train, X3_test, Y3_train, Y3_test = train_test_split(df3.iloc[:,0:6].values, df3.iloc[:,-1].values, test_size=0.25, random_state=147)
X4_train, X4_test, Y4_train, Y4_test = train_test_split(df4.iloc[:,0:6].values, df4.iloc[:,-1].values, test_size=0.25, random_state=147)

x_train = [X1_train, X2_train, X3_train, X4_train]
y_train = [Y1_train, Y2_train, Y3_train, Y4_train]

x_test = [X1_test, X2_test, X3_test, X4_test]
y_test = [Y1_test, Y2_test, Y3_test, Y4_test]

# Use min-max scaler and transform each feature accordingly. 
# We put each feature value to a certain range (in general (0,1))
scaler = []
for index in range(4):
    scaler.append(MinMaxScaler(feature_range=(0,1)))

# We should scale the x_train & x_test to make the standardization
x_train_scaled = []
for train_data_index in range(len(x_train)):
    train_data = scaler[train_data_index].fit_transform(x_train[train_data_index])
    x_train_scaled.append(train_data)

x_test_scaled = []
for test_data_index in range(len(x_test)):
    test_data = scaler[test_data_index].transform(x_test[test_data_index])
    x_test_scaled.append(test_data)

# initalize the model
model = []
for index in range(4):
    model.append(LinearRegression())

# Fit the training data to the model (training)
linear_regression = []
for index in range(len(x_train_scaled)):     
    regression = model[index].fit(x_train_scaled[index], y_train[index])
    linear_regression.append(regression)

# Predict the values by using all test data
prediction = []
for index in range(len(linear_regression)):
    prediction.append(linear_regression[index].predict(x_test_scaled[index]))

# Calculate the score of the model in the test data 
# We want to desire higher values
score = []
for index in range(len(linear_regression)):
    score.append(linear_regression[index].score(x_test_scaled[index], y_test[index]))

# Printing the each score of the model
for index, value in enumerate(score):
    print(f'score{index+1}: {value}')

def calculate_error_rate(linear_regression, y_test, prediction):
    # Calculate the mean squared error
    mse = []
    for index in range(len(linear_regression)):
        mse.append(mean_squared_error(y_test[index], prediction[index]))
    
    # Calculate the mean absolute error
    mae = []
    for index in range(len(linear_regression)):
        mae.append(mean_absolute_error(y_test[index], prediction[index]))
    
    return mse, mae

# Printing the each error metrics of the model
mse, mae = calculate_error_rate(linear_regression, y_test, prediction)
for index, value in enumerate(mse):
    print(f'MSE{index+1}: {value}')
for index, value in enumerate(mae):
    print(f'MAE{index+1}: {value}')

def visulize_error_rate(x_label, y_label, models, error_rate):
    # Visulize the error rate
    plt.xlabel(x_label, fontweight="bold", style="italic")
    plt.ylabel(y_label, fontweight="bold", style="italic")
    plt.scatter(models, error_rate, s=75, marker='o', color='b')
    plt.show()

visulize_error_rate('Models', 'Mean Squared Error', ["M{}".format(index+1) for index in range(4)], mse)
visulize_error_rate('Models', 'Mean Absolute Error', ["M{}".format(index+1) for index in range(4)], mae)
