from PIL import Image
import numpy as np
from keras.models import load_model

model = load_model('./models/cat_dog_model 0.817.h5')
categories = ['cat', 'dog']

# 사진 불러와서 예측해 보기
img = Image.open('./imgs/cat_dog/train/cat.1.jpg')
img = img.resize((64, 64))
img = np.array(img)
img = img/255
img = img.reshape(1, 64, 64, 3)

pred = model.predict(img)
print(categories[int(np.around(pred))])

img = Image.open('./cat_test01.png')
img = img.resize((64, 64))
img = np.array(img)
img = img/255
img = img.reshape(1, 64, 64, 3)

pred = model.predict(img)
print(categories[int(np.around(pred))])

img = Image.open('./dog_test01.jpg')
img = img.resize((64, 64))
img = np.array(img)
img = img/255
img = img.reshape(1, 64, 64, 3)

pred = model.predict(img)
print(categories[int(np.around(pred))])

img = Image.open('./dog_test02.jpg')
img = img.resize((64, 64))
img = np.array(img)
img = img/255
img = img.reshape(1, 64, 64, 3)