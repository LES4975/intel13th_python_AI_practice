# -*- coding: utf-8 -*-
"""AI_exam05_titanic.ipynb의 사본

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1CQHbYCkhci62NdeScVeHPwIkY2wVKPHz

### 이진분류기 (타이타닉) ###
"""

import seaborn as sns
import pandas as pd
import numpy as np

raw_data = sns.load_dataset('titanic')
print(raw_data.head())
# NaN값이 많아 데이터를 버리면 데이터손실이 많을 것 같음

raw_data.info()

# 모델은 입력 2개 주고 맨 마지막에서는 출력 1개
# pandas 공부

"""pandas 공부가 끝났다!!!

데이터 전처리!!!
"""

raw_data.isnull().sum() # 결측치 개수 확인

# 필요없는 column 지우기
clean_data = raw_data.dropna(axis=1, thresh=500) # NULL이 500개 이상인 column 드롭
print(clean_data.columns)

# 비어있는 age 데이터는 평균값으로 채우기
clean_data['age'].fillna(clean_data['age'].mean(), inplace=True)
print(clean_data.isnull().sum())

clean_data.drop(['embark_town', 'alive', 'class'], axis=1, inplace=True)
print(clean_data.columns)

# 이전 값으로 결측치 채우기
clean_data['embarked'].fillna(method='ffill', inplace=True)

print(clean_data.isnull().sum())

clean_data.info()

# 성별 데이터를 문자열에서 정수로 대체
clean_data['sex'].replace({'male':0, 'female':1}, inplace=True)
print(clean_data['sex'].unique())

# embarked 데이터를 문자열에서 정수로 대체
print(clean_data.embarked.value_counts())

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
clean_data['embarked'] = encoder.fit_transform(clean_data['embarked'])
print(clean_data['embarked'].value_counts())

clean_data['who'] = encoder.fit_transform(clean_data['who'])
print(clean_data['who'].value_counts())

# bool 타입인 데이터는 type 변환하기
clean_data['adult_male'] = clean_data['adult_male'].astype('int64')
clean_data['alone'] = clean_data['alone'].astype('int64')

clean_data.info()

# 데이터 분리
target = pd.DataFrame(clean_data.iloc[:, 0], columns=['survived']) # 첫 번째 열(0번째)에 해당하는 데이터를 선택해서 데이터프레임으로 바꾼다
training_data = clean_data.drop(['survived'], axis=1)
print(target.head())
print(training_data.head())

value_data = training_data[['age', 'fare']]
print(value_data.head())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(value_data)
value_data = pd.DataFrame(scaled_data, columns=['age', 'fare'])
print(value_data)

print(value_data.describe())

training_data.drop(['age', 'fare'], axis=1, inplace=True)
print(training_data.head())

# 더미화?
onehot_data = pd.get_dummies(training_data, columns=training_data.columns)
print(onehot_data.head())

onehot_data = onehot_data.astype('int64') # bool -> int64
onehot_data.info()

training_data = pd.concat([onehot_data, value_data], axis=1)
training_data.info()

# 학습, 검증을 위해 데이터셋 분리하기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(training_data, target, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

"""모델 만들기!!!!"""

from keras.models import Sequential
from keras.layers import Dense, Dropout

print(X_train.shape)

# 모델 구축
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.02)) # 2% 정도 학습하지 않음(Dropout)
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.02))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.02))
model.add(Dense(1, activation='sigmoid')) # 이진분류기에서 출력을 0 또는 1로 내보내는 역할을 함
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['binary_accuracy'])
fit_hist = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2, verbose=1)

import matplotlib.pyplot as plt

plt.plot(fit_hist.history['binary_accuracy'])
plt.plot(fit_hist.history['val_binary_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss', score[0])
print('Test Accuracy', score[1])