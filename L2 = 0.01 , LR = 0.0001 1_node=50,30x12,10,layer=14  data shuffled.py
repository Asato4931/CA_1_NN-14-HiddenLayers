from sklearn import preprocessing
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


from sklearn.compose import ColumnTransformer


from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support

import pandas as pd



######################データの前処理#####################

#ヘッダー無しのデータ読み込み
#data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/CA課題用/CA課題1/covtype.data", header=None)
data = pd.read_csv("C:/Users/81806/Desktop/CA課題1/covtype.data", header=None)
#print(data)

data = data.sample(frac=1, random_state = 2023)

#x_train,x_test,y_train,y_test 分割＆yはonehot化


x = data.loc[:, 0:9]
x2 = data.loc[:, 10:53]


#スケーラー3種
scaler = StandardScaler()
#scaler = MinMaxScaler(feature_range=(0,1) , copy = True)


#入力をスケーリング
scaler.fit(x)

x = scaler.transform(x)
x = pd.DataFrame(x)
x = pd.concat([x,x2],axis=1)




#入力データの分割
#x_train = x[0:348608]
#x_cv = x[348609:464812]
x_train = x[0:464812]
x_test = x[464813:581012]


y = data.loc[:, 54:54]

enc = OneHotEncoder(categories='auto', sparse_output = False)
y = enc.fit_transform(y)

#教師データの分割
#y_train = y[0:348608]
#y_cv = y[348609:464812]
y_train = y[0:464812]
y_test = y[464813:581012]





######################ニューラルネットワークの学習#####################


clf = MLPClassifier(activation = 'tanh' , solver='adam', alpha=1e-2, hidden_layer_sizes=(50,30,30,30,30,30,30,30,30,30,30,30,30,10), max_iter=10000, learning_rate_init=0.0001, random_state=2023, batch_size = 200,warm_start=True )



#loss_curve_, validation_scores_ = clf.fit(x_train, y_train)


######################Cross Validation#####################
#Trainingデータを5分割して１つをCV用に、それを5回分シャッフル。　学習する母データの数を6回変えて行う。 test scoresは母データ÷5で常に行ってくれているはず。


#train_sizes, train_scores, valid_scores = learning_curve(clf, x_train, y_train, cv = 4,  train_sizes=[0.01,0.1, 0.2, 0.5,0.75,0.9,1])

#train_score_average = np.mean(train_scores, axis=1)
#valid_score_average = np.mean(valid_scores, axis=1)

#plt.plot(train_sizes, train_score_average, color='green', label="train")
#plt.plot(train_sizes, valid_score_average, color='red',  label="validation")
#plt.title("learning curve [50,30,30,30,30,30,10 nodes 7 layer NN]")
#plt.xlabel("sample_size")
#plt.ylabel("accuracy")
#plt.legend()
#plt.show()





######################モデルの評価#####################

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
score = metrics.accuracy_score(y_test, y_pred)



print(score)



