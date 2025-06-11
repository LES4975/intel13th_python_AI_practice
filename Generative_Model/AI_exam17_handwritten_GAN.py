import matplotlib.pyplot as plt
import numpy as np
from keras.models import *

number_GAN_models = []
for i in range(10):
    number_GAN_models.append(load_model('./models/MNIST_handwritten_number/generator_{}.h5'.format(i)))
numbers = 314159253589793 # 생성하고 싶은 숫자
imgs = []
numbers = str(numbers) # 숫자를 문자열로 변환
for i in numbers: # 각 문자열에 해당하는 숫자마다 이미지 생성
    print(i)
    i = int(i)
    z = np.random.normal(0, 1, (1, 100))
    fake_img = number_GAN_models[i].predict(z)
    fake_img = fake_img * 0.5 + 0.5
    print(fake_img.shape)
    imgs.append(fake_img.reshape(28, 28))

_, axs = plt.subplots(1, len(numbers), figsize=(10, 40), sharex=True)
for i in range(len(numbers)):
    axs[i].imshow(imgs[i], cmap='gray')
    axs[i].axis('off')
plt.show()

# 이미지 하나로 이어붙이기
img = imgs[0]
for i in range(1, len(numbers)):
    img = np.append(img, imgs[i], axis=1)
plt.imshow(img)
plt.axis('off')
plt.show()