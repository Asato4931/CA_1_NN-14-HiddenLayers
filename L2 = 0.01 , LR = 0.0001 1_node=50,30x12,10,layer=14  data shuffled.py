from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pandas as pd



######################データの前処理#####################

#ヘッダー無しのデータ読み込み
data = pd.read_csv("C:/Users/81806/Desktop/CA課題1/covtype.data", header=None)

data = data.sample(frac=1, random_state = 2023)

x = data.loc[:, 0:9]
x2 = data.loc[:, 10:53]

scaler = StandardScaler()

#入力をスケーリング
scaler.fit(x)

x = scaler.transform(x)
x = pd.DataFrame(x)
x = pd.concat([x,x2],axis=1)

#入力データの分割
x_train = x[0:464812]
x_test = x[464813:581012]


y = data.loc[:, 54:54]
enc = OneHotEncoder(categories='auto', sparse_output = False)
y = enc.fit_transform(y)

#教師データの分割
y_train = y[0:464812]
y_test = y[464813:581012]


######################ニューラルネットワークの学習#####################
clf = MLPClassifier(activation = 'tanh' , solver='adam', alpha=1e-2, 
                    hidden_layer_sizes=(50,30,30,30,30,30,30,30,30,30,30,30,30,10),
                      max_iter=10000, learning_rate_init=0.0001, random_state=2023, 
                      batch_size = 200,warm_start=True )


######################モデルの評価#####################
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
score = metrics.accuracy_score(y_test, y_pred)
print(score)



