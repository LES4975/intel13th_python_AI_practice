# 데이터 전처리

from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split

img_dir = './imgs/cat_dog/train/' # 경로 설정
categories = ['cat', 'dog'] # 카테고리 설정

image_w = 64 # 이미지 너비
image_h = 64 # 이미지 높이
pixel = 64 * 64 * 3 # 이미지 크기에 RGB 색상을 갖고 있으니 pixel 값은 64 * 64 * 3
X = []
Y = []
for idx, category in enumerate(categories):
    for i, img_path in enumerate(glob.glob(img_dir + category + '*.jpg')):
        try:
            img = Image.open(img_path) # 이미지 열기
            img = img.resize((image_w, image_h)) # 64*64로 리사이징
            img = np.array(img) # 이미지 파일을 array로 변환
            X.append(img) # 이미지 정보를 X에 저장
            Y.append(idx) # 고양이 이미지라면 0을 넣고, 강아지 이미지라면 1을 넣고
            if i % 300 == 0: # 중간중간마다 잘 실행되고 있는지 확인용
                print(category, ':', img_path)
        except: # 예외 발생 시 에러 처리
            print('error: ', category, img_path) # 문제가 발생한 파일 경로 출력
X = np.array(X) # 이미지 리스트를 array 타입으로 변환
Y = np.array(Y) # 라벨 리스트를 array 타입으로 변환
X = X / 255 # 스케일링
print(X[0])
print(Y[0])
print(X.shape)
print(Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
np.save('binary_data/cat_dog_x_train.npy', X_train)
np.save('binary_data/cat_dog_y_train.npy', Y_train)
np.save('binary_data/cat_dog_x_test.npy', X_test)
np.save('binary_data/cat_dog_y_test.npy', Y_test)