from PIL import Image
import numpy as np
from keras.models import load_model

model = load_model('./models/horse_or_human_model 1.0.h5') # 경로 설정
categories = ['horse', 'human'] # 카테고리 설정

# 사진 불러와서 예측해 보기
img = Image.open('./dataset/horses/horse29-3.png').convert('RGB')
img = img.resize((150, 150))
img = np.array(img)
img = img/255
img = img.reshape(1, 150, 150, 3)

pred = model.predict(img)
print(categories[int(np.around(pred))])

# 사진 불러와서 예측해 보기
img = Image.open('./dataset/horses/horse06-3.png').convert('RGB')
img = img.resize((150, 150))
img = np.array(img)
img = img/255
img = img.reshape(1, 150, 150, 3)

pred = model.predict(img)
print(categories[int(np.around(pred))])

# 사진 불러와서 예측해 보기
img = Image.open('./dataset/humans/human05-05.png').convert('RGB')
img = img.resize((150, 150))
img = np.array(img)
img = img/255
img = img.reshape(1, 150, 150, 3)

pred = model.predict(img)
print(categories[int(np.around(pred))])

# 사진 불러와서 예측해 보기
img = Image.open('./dataset/humans/human12-27.png').convert('RGB')
img = img.resize((150, 150))
img = np.array(img)
img = img/255
img = img.reshape(1, 150, 150, 3)

pred = model.predict(img)
print(categories[int(np.around(pred))])