# !pip install -U scikit-learn

"""## 2.Dataset Exploration (10%)"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import metrics
from sklearn.svm import SVC
import matplotlib.gridspec as gridspec
from mlxtend.plotting import plot_decision_regions
from mlxtend.preprocessing import shuffle_arrays_unison
from tqdm.notebook import tqdm_notebook as tqdm
from sklearn import svm
from sklearn.utils import shuffle
from matplotlib.colors import ListedColormap
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam

import seaborn as sns
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import sklearn.linear_model as skl_lm
import matplotlib.pyplot as plt

df_full=pd.read_csv('/content/mhealth_raw_data.csv')
df_full

df_full.head()

df_full.describe()

df_full.shape

df = df_full.groupby('Activity').apply(lambda x: x.sample(n=1000))
df = df.sample(frac=1).reset_index(drop=True)

df.shape

df['Activity'].value_counts()

# # 160745
# # 6745
# df = df_full.iloc[:60745]

df.shape

X = df.drop(['Activity', 'subject'], axis=1)
y = df['Activity']

df['Activity'].value_counts()

y

y.value_counts()

# df_full['Activity'].value_counts().plot(kind= 'bar')
df['Activity'].value_counts().plot(kind= 'bar')

df_numerical = df.drop(['Activity', 'subject'], axis=1)

df_numerical.shape

#check for missing data
print('missing values -> {}'.format (df.isna().sum()))  # -> why ??

df.dropna(inplace = True)

#check duplicates
print('dubblicate values -> {}'.format (df.duplicated()))

#drop duplicates
df.drop_duplicates(inplace = True)
#test after remove the duplicates
print(df.duplicated().sum())

X.shape

X.head()

y.head()

X_train, X_test,y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle=True, random_state=0)
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

#check the traing set size and test set size:
print("Training set size:", len(X_train), "samples")
print("Test set size:", len(X_test), "samples")

#check the traing set shape and test set shape:
print("Training set shape:", X_train.shape, "samples")
print("Test set shape:", X_test.shape, "samples")

"""##3. Model Training (40%)

## KNN
"""

def distance_ecu(x_train, x_test_point):
    distances = []
    for row in range(len(x_train)):
        current_train_point = x_train[row]
        current_distance = 0
        for col in range(len(current_train_point)):
            current_distance += (current_train_point[col] - x_test_point[col]) ** 2
        current_distance = np.sqrt(current_distance)
        distances.append(current_distance)
    distances = pd.DataFrame(data=distances, columns=['index'])
    return distances

def nearest_neighbors(distance_point , k):
    df_nearest = sorted(distance_point)
    df_nearest = df_nearest[:k]
    return df_nearest

def voting(df_nearest , y_train):
    counter_vote  = collections.Counter(y_train[df_nearest])
    y_pred = counter_vote.most_common(1)[0][0]
    return y_pred

def KNN_from_scratch(X_train, y_train, X_test, K):
    y_pred = []
    for i in range(len(X_test)):
          # Loop over all the test set and perform the three steps
        distances = []
        for j in range(len(X_train)):
            distances.append([np.sqrt(np.sum(np.square(X_test[i] - X_train[j]))), j])
        distances.sort(key=lambda x: x[0])
        df_nearest = np.array(distances[:K])[:, 1].astype(int)
        y_pred.append(voting(df_nearest, y_train))
    return y_pred

data_types = df.dtypes
print("Data type of each column:")
print(data_types)

# print("\n Before normalization:")
# K = 3
# y_pred_scratch = KNN_from_scratch(X_train, y_train, X_test, K)
# # print(y_pred_scratch)

# accuracy_scratch = accuracy_score(y_test, y_pred_scratch)
# print(f'The accuracy of implementation is {accuracy_scratch*100} %')

# precision = precision_score(y_test, y_pred_scratch, average='micro')
# recall = recall_score(y_test, y_pred_scratch, average='micro')
# f1 = f1_score(y_test, y_pred_scratch, average='micro')

# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)

# conf_matrix = confusion_matrix(y_test, y_pred_scratch)
# conf_matrix

scale = Normalizer().fit(X_train)
x_train_normalized = scale.transform(X_train)
x_test_normalized = scale.transform(X_test)

# print("After normalization:")
# k = 3
# y_pred_normalized = KNN_from_scratch(x_train_normalized, y_train, x_test_normalized, k)
# # print(y_pred_normalized)

# accuracy_normalized = accuracy_score(y_test, y_pred_normalized)
# print(f'The accuracy of normalized implementation is {accuracy_scratch*100} %')

# precision = precision_score(y_test, y_pred_normalized, average='micro')
# recall = recall_score(y_test, y_pred_normalized, average='micro')
# f1 = f1_score(y_test, y_pred_normalized, average='micro')

# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)

# conf_matrix = confusion_matrix(y_test, y_pred_normalized)
# conf_matrix

knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(x_train_normalized, y_train)

print("X train before Normalization")
print(X_train[0:5])
print("\nX train after Normalization")
print(x_train_normalized[0:5])

knn_y_pred = knn.predict(X_test)
knn_y_pred = np.round(knn_y_pred)

Knn_accuracy = accuracy_score(y_test, knn_y_pred)
knn_precision = precision_score(y_test, knn_y_pred, average='micro')
knn_recall = recall_score(y_test, knn_y_pred, average='micro')
knn_f1_score = f1_score(y_test, knn_y_pred, average='micro')
print("Knn_accuracy:", Knn_accuracy)
print("knn_precision:", knn_precision)
print("knn_recall:", knn_recall)
print("F1 knn_f1_score:", knn_f1_score)

"""##LInear Regression"""

regression= LinearRegression()
regression.fit(X_train, y_train)

regression_pred = regression.predict(X_test)
regression_pred = np.round(regression_pred)
regression_accuracy = accuracy_score(y_test, regression_pred)
print("Regression accuracy:", regression_accuracy)
mse = mean_squared_error(y_test, regression_pred)
print(f'linear regression Mean Squared Error:{mse}')

"""## SVM"""

df

scaler = StandardScaler().fit(X_train)
x_train_scaled = scaler.transform(X_train)
x_test_scaled = scaler.transform(X_test)

svm = SVC(C=10000, kernel='poly')
svm.fit(x_train_scaled, y_train)

svm_y_pred = svm.predict(x_test_scaled)
svm_y_pred = np.round(svm_y_pred)

svm_accuracy = accuracy_score(y_test, svm_y_pred)
svm_precision = precision_score(y_test, svm.predict(X_test), average='micro')
svm_recall = recall_score(y_test, svm.predict(X_test), average='micro')
svm_f1_score = f1_score(y_test, svm.predict(X_test), average='micro')

print("svm_accuracy:", svm_accuracy*100,'%')
print("svm_precision:", svm_precision)
print("svm_recall:", svm_recall)
print("F1 svm_f1_score:", svm_f1_score)

conf_matrix = confusion_matrix(y_test, svm.predict(X_test))
conf_matrix

"""## Neural Network"""

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

# scaler = StandardScaler().fit(X_train)
# x_train_normalized = scaler.transform(X_train)
# x_test_normalized = scaler.transform(X_test)

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, y, test_size=0.3, random_state=1)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5, random_state=1)
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

model = Sequential([
    Dense(64, activation='relu', input_shape=(12,)),
    Dense(14, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

df_X_train = pd.DataFrame(Y_train)
df_X_train.value_counts()

hist = model.fit(X_train, Y_train,
          batch_size=32,epochs=10,
          validation_data=(X_val, Y_val))

y_pred_nn = model.predict(X_test)
y_pred_nn = np.round(y_pred_nn)

NN2_precision = precision_score(Y_test, y_pred_nn, average='micro')
NN2_recall = recall_score(Y_test, y_pred_nn, average='micro')
NN2_f1_score = f1_score(Y_test, y_pred_nn, average='micro')
NN2_Accuracy = accuracy_score(Y_test, y_pred_nn)
print("NN2_Accuracy:", NN2_Accuracy)
print("NN2_precision:", NN2_precision)
print("NN2_recall:", NN2_recall)
print("NN2_f1_score:", NN2_f1_score)

print("confusion_matrix:")
conf_matrix = confusion_matrix(Y_test, y_pred_nn)
conf_matrix

"""##Logistic Regression"""

df

print('Digits dataset structure= ', dir(df)) # dir() function returns all properties and methods of the specified object, without the values.
print('Data shape= ', df.shape)
print('Data conatins pixel representation of each image, \n', df)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")

y_train = y_train[:X_train.shape[0]]
X_train = X_train[:y_train.shape[0]]

X_test = X_test[:y_test.shape[0]]
y_test = y_test[:X_test.shape[0]]

lm = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')
lm.fit(X_train, y_train)

print('Predicted value is =', lm.predict([X_test[200]]))

y_pred = lm.predict(X_test)
LM_accuracy = accuracy_score(y_test, y_pred)
LM_precision = precision_score(y_test, y_pred, average='micro')
LM_recall = recall_score(y_test, y_pred, average='micro')
LM_f1_score = f1_score(y_test, y_pred, average='micro')

print("LM_accuracy:", LM_accuracy)
print("LM_precision:", LM_precision)
print("LM_recall:", LM_recall)
print("LM_f1_score:", LM_f1_score)

predictions=lm.predict(X_test)
conf_matrix=confusion_matrix(y_test, predictions)
conf_matrix

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(8,6), dpi=100)
display = ConfusionMatrixDisplay(conf_matrix, display_labels=lm.classes_)

# set the plot title using the axes object
ax.set(title='Confusion Matrix')
display.plot(ax=ax);
