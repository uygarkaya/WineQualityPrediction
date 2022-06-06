import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier

file_name = '../Dataset/WineQualityNew_C.csv'
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
    scaler.append(StandardScaler())

x_train_scaled = []
for train_data_index in range(len(x_train)):
    train_data = scaler[train_data_index].fit_transform(
        x_train[train_data_index])
    x_train_scaled.append(train_data)

x_test_scaled = []
for test_data_index in range(len(x_test)):
    test_data = scaler[test_data_index].transform(x_test[test_data_index])
    x_test_scaled.append(test_data)

def pca(data, n_components=6):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca

x_pca = pca(x_train_scaled[0])
print('Explained variance ratio: ', x_pca.explained_variance_ratio_)

x_train_scaled.append(x_pca.transform(x_train_scaled[0]))
x_test_scaled.append(x_pca.transform(x_test_scaled[0]))
y_train.append(y_train[0])
y_test.append(y_test[0])

model = []
for index in range(len(x_train_scaled)):
    model.append(KNeighborsClassifier(n_neighbors=5))

kneighbors_classifier = []
for index in range(len(x_train_scaled)):
    classification = model[index].fit(x_train_scaled[index], y_train[index])
    kneighbors_classifier.append(classification)

prediction = []
for index in range(len(kneighbors_classifier)):
    prediction.append(kneighbors_classifier[index].predict(x_test_scaled[index]))

accuracy = []
for index in range(len(kneighbors_classifier)):
    accuracy.append(kneighbors_classifier[index].score(
        x_test_scaled[index], y_test[index]))

# Printing the each score of the model
for index, value in enumerate(accuracy):
    print(f'accuracy {index+1}: {value}')

for index, (prediction, y_t) in enumerate(zip(prediction, y_test)):
    cm = confusion_matrix(y_t, prediction, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Low", "Medium", "High"])
    disp.plot()
    plt.show()
    print(f"CR of model{index+1}:\n", classification_report(y_t, prediction))

