from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split

# 경로 및 카테고리 설정
img_dir_horse = './dataset/horses/'
img_dir_human = './dataset/humans/'
categories = ['horses', 'humans']

image_w = 150
image_h = 150
pixel = image_w * image_h * 3

# 데이터와 라벨을 저장할 리스트
X = [] # 이미지 리스트
Y = [] # 라벨 리스트

# 말 데이터 정리
for i, img_path in enumerate(glob.glob(img_dir_horse + '*.png')):
    try:
        img = Image.open(img_path).convert('RGB')  # 이미지 열기
        img = img.resize((image_w, image_h))  # 혹시 모르니까 리사이징
        img = np.array(img) / 255.0  # 정규화
        X.append(img)  # 이미지 정보를 X에 저장
        Y.append(0)  # 0이면 말
        if i % 30 == 0:
            print('horse', ':', img_path)
    except:  # 예외가 발생하면
        print('error: ', img_path)

# 사람 데이터 정리
for i, img_path in enumerate(glob.glob(img_dir_human + '*.png')):
    try:
        img = Image.open(img_path).convert('RGB')  # 이미지 열기
        img = img.resize((image_w, image_h))  # 혹시 모르니까 리사이징
        img = np.array(img) / 255.0  # 정규화
        X.append(img)  # 이미지 정보를 X에 저장
        Y.append(1)  # 1이면 사람
        if i % 30 == 0:
            print('human', ':', img_path)
    except:  # 예외가 발생하면
        print('error: ', img_path)

X = np.array(X) # 이미지 리스트를 array 타입으로 변환
Y = np.array(Y) # 라벨 리스트를 array 타입으로 변환
print(X[0])
print(Y[0])
print(X.shape)
print(Y.shape)

# 데이터셋 나누기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
np.save('binary_data/horse_human_x_train.npy', X_train)
np.save('binary_data/horse_human_y_train.npy', Y_train)
np.save('binary_data/horse_human_x_test.npy', X_test)
np.save('binary_data/horse_human_y_test.npy', Y_test)