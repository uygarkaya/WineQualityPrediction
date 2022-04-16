import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# Read the data
file_name = '../Dataset/WineQuality.csv'
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
X1_train, X1_test, Y1_train, Y1_test = train_test_split(df1.iloc[:,0:7].values, df1.iloc[:,-1].values, test_size=0.25, random_state=147)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(df2.iloc[:,0:7].values, df2.iloc[:,-1].values, test_size=0.25, random_state=147)
X3_train, X3_test, Y3_train, Y3_test = train_test_split(df3.iloc[:,0:7].values, df3.iloc[:,-1].values, test_size=0.25, random_state=147)
X4_train, X4_test, Y4_train, Y4_test = train_test_split(df4.iloc[:,0:7].values, df4.iloc[:,-1].values, test_size=0.25, random_state=147)

x_train = [X1_train, X2_train, X3_train, X4_train]
y_train = [Y1_train, Y2_train, Y3_train, Y4_train]

x_test = [X1_test, X2_test, X3_test, X4_test]
y_test = [Y1_test, Y2_test, Y3_test, Y4_test]

# Use min-max scaler and transform each feature accordingly. 
# We put each feature value to a certain range (in general (0,1))
scaler = MinMaxScaler(feature_range=(0,1))

# We should scale the x_train & x_test to make the standardization
x_train_scaled = []
for train_data in x_train:
    train_data = scaler.fit_transform(train_data)
    x_train_scaled.append(train_data)

x_test_scaled = []
for test_data in x_test:
    test_data = scaler.transform(test_data)
    x_test_scaled.append(test_data)

# initalize the model
linearRegression = LinearRegression()

# # Fit the training data to the model (training)
# linear_regression = []
# for index in range(len(x_train_scaled)):
#     # There is a mistake line 57: Input contains NaN, infinity or a value too large for dtype('float64')     
#     regression = linearRegression.fit(x_train_scaled[index], y_train[index])
#     linear_regression.append(regression)

