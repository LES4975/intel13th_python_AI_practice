import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist

# 모델 불러 오기
autoencoder = load_model('./models/autoencoder.h5')
# autoencoder = load_model('./models/autoencoder_noisy.h5') # 노이즈 낀 데이터로 학습한 모델

# 데이터셋 불러 오기
(_, _), (x_test, _) = mnist.load_data()

x_test = x_test[:10] / 255 # 10개만 하자

conv_x_test = x_test.reshape(-1, 28, 28, 1)
# conv_x_test = np.zeros(shape=(10, 28, 28, 1)) # 입력으로 노이즈만 섞인 이미지를 줘 보자.

noise_factor = 0.5
conv_x_test_noisy = conv_x_test + np.random.normal(
    loc=0.0, scale=1.0, size=conv_x_test.shape) * noise_factor # 노이즈를 테스트 데이터셋에 더하기
conv_x_test_noisy = np.clip(conv_x_test_noisy, 0.0, 1.0) # 0보다 작으면 0으로 저장, 1보다 크면 1로 저장
decoded_img = autoencoder.predict(conv_x_test_noisy[:10])

# 예측 결과 출력(noise가 포함된 경우)
n = 10
plt.figure(figsize=(20, 4))
# autoencoder에 데이터를 넣어본 결과
for i in range(n):
    ax = plt.subplot(2, 10, i + 1)
    # plt.imshow(x_test[i]) # 깨끗한 x_test
    plt.imshow(conv_x_test_noisy[i]) # 비교군 - 노이즈가 낀 x_test
    # x축과 y축 선을 비표시
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 1 + n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    # x축과 y축 선을 비표시
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()