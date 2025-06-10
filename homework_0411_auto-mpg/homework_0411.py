# 1. auto mpg 데이터셋의 horsepower column을 저출력/중간출력/고출력으로 카테고리화
# 2. auto mpg 데이터셋을 출력 카테고리로 분류하는 다중분류 모델을 만들고 source 파일을 제출하세용
# 단, 'horsepower' column은 사용하지 말고!

import pandas as pd
import numpy as np

# 데이터셋 가져 오기
raw_df = pd.read_csv('auto-mpg.csv',
               names=['mpg', 'cylinders', 'displacement', 'horsepower',
                      'weight', 'acceleration', 'model year', 'origin', 'name'])

raw_df.info()
# 'horsepower'와 'name' column만 dtype이 object이기 때문에 과제 2번을 수행할 때는 결측치 제거 불필요

# -----------------------------------------------------------------------------------------

# 1. auto mpg 데이터셋의 horsepower column을 저출력/중간출력/고출력으로 카테고리화하시오.
h1_df = raw_df

# 'horsepower' column의 결측치를 어떻게 좀 하기
h1_df.horsepower = h1_df.horsepower.replace('?', np.nan)
h1_df.dropna(subset=['horsepower'], axis=0, inplace=True) # 'horsepower'에 있는 NaN 값을 드롭한다. axis=0(row)에 대해서
h1_df.horsepower = h1_df.horsepower.astype('float') # type 바꾸기
h1_df.info()

# horsepower를 category화하기
count, bin_dividers = np.histogram(h1_df['horsepower'], bins=3) # 최솟값, 최댓값을 찾아서 3개의 구간으로 나눈다.
print(count)
print(bin_dividers)

bin_names = ['저출력', '중간출력', '고출력']
h1_df['hp_bin'] = pd.cut(x=h1_df['horsepower'], bins=bin_dividers, labels=bin_names, include_lowest=True)
print(h1_df[['horsepower','hp_bin']])
h1_df.info()

# -------------------------------------------과제 1번 끝---------------------------------------

# 2. auto mpg 데이터셋을 출력 카테고리로 분류하는 다중분류 모델을 만들고 source 파일을 제출하시오.

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *

h2_df = h1_df # 과제 1번에 썼던 데이터프레임 그대로 쓰기

h2_df = h2_df.drop('horsepower', axis=1)
h2_df = h2_df.drop('name', axis=1)

h2_df.info()

# 타겟 카테고리 column을 숫자로 바꾸기
h2_df['hp_bin'] = h2_df['hp_bin'].astype(str)  # 문자열로 변환
h2_df['hp_bin'] = h2_df['hp_bin'].replace({'저출력':0, '중간출력':1, '고출력':2}).astype(int)
print(h2_df['hp_bin'].unique())

print("auto-mpg의 타겟 컬럼 모양")
print(h2_df.hp_bin.shape)
print("h2_df 정보")
h2_df.info()

# 산점도로 확인해 보기
# 저만 보겠습니다
# pd.plotting.scatter_matrix(h2_df, c=h2_df.hp_bin, figsize=(6, 6), s=60, marker='0')
# plt.show()

# 데이터셋 전처리

target = pd.DataFrame(h2_df.iloc[:, 7], columns=['hp_bin']) # hp_bin 카테고리를 타겟 데이터프레임으로 변환
h2_df = h2_df.drop('hp_bin', axis=1) # 본래 데이터프레임에 hp_bin을 제거해서 학습용으로 만듦
# h2_df의 인덱스 중에 빠진 번호가 있기 때문에, 인덱스를 새로 붙여 줌
h2_df = h2_df.reset_index(drop=True)
print(h2_df.head())

# 스케일링해야 할 데이터 떼어 놓기
not_scaled_data = h2_df.iloc[:, 0:5]
print(not_scaled_data)

# 스케일링
scaler = StandardScaler()
scaled_data = scaler.fit_transform(not_scaled_data)
scaled_data = pd.DataFrame(scaled_data, columns=not_scaled_data.columns)
print(scaled_data)

# 정수 값만 가지고 있는 컬럼으로 이루어진 데이터프레임 뽑아내기
int_data = h2_df.iloc[:, 5:7]

# 더미화
onehot_data = pd.get_dummies(int_data, columns=int_data.columns)
onehot_data = onehot_data.astype('int64') # bool -> int64로 형변환
onehot_data.info()

# training용 데이터프레임으로 재조립
training_data = pd.concat([scaled_data, onehot_data], axis=1)
print(training_data.keys())

print(training_data)
print(training_data.describe())

# 잠깐 데이터셋 모양 좀 확인할게요
print(training_data.shape)
print(target.shape)
print(target)

# One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False) # sparse_output으로 희소행렬 표현 방식을 설정할 수 있다
encoded_target = encoder.fit_transform(target)
print(encoded_target.shape)
print(encoded_target)

# 학습용, 검증용 데이터셋 분리하기
X_train, X_test, y_train, y_test = train_test_split(training_data, encoded_target, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# 학습 모델 만들기
model = Sequential()
model.add(Dense(64, input_shape=(21,), activation='relu'))
model.add(Dropout(0.02))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.02))
model.add(Dense(3, activation='softmax'))
model.summary()

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
fit_hist = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

# 학습 정확도 시각화
plt.plot(fit_hist.history['accuracy'], label='Training Accuracy')
plt.plot(fit_hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# 학습 손실 시각화
plt.plot(fit_hist.history['loss'], label='Training Loss')
plt.plot(fit_hist.history['val_loss'], label='Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# 모델 검증
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss', score[0])
print('Test Accuracy', score[1])

# 모델 저장
model.save('auto-mpg_horsepower {}.h5'.format(np.around(score[1], 3)))

# 과제 2번 완료, 이상입니다.