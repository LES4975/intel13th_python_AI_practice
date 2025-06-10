# AI_exam14_autoencoder_CNN.py의 내용을 복사하여 일부 수정
import numpy as np
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.datasets import mnist

# 모델 구축 (레이어의 출력 이미지 크기를 주석으로 달아놓았다)
# encoder
input_img = Input(shape=(28, 28, 1, )) # Conv 레이어에 넣을 수 있도록 크기 조정
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img) # 28 * 28
x = MaxPool2D((2, 2), padding='same')(x) # 이미지 크기를 반으로 줄임/ # -> 14 * 14
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x) # 14 * 14
x = MaxPool2D((2, 2), padding='same')(x) # -> 7 * 7
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x) # 7 * 7
encoded = MaxPool2D((2, 2), padding='same')(x) # -> 4 * 4

#decoder
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded) # 4 * 4
x = UpSampling2D((2, 2))(x) # 이미지의 데이터 1px을 복사해서 2*2px씩으로 늘림(크기는 늘어나지만 데이터 손실 발생) / -> 8 * 8
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x) # 8 * 8
x = UpSampling2D((2, 2))(x) #-> 16 * 16
x = Conv2D(16, (3, 3), activation='relu')(x) # padding을 사용하지 않음으로써 합성곱을 할 때 이미지 크기를 줄임 / -> 14 * 14
x = UpSampling2D((2, 2))(x) # -> 28 * 28
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x) # 최종적으로 1장의 이미지를 출력 / 28 * 28


# autoencoder 구축
autoencoder = Model(input_img, decoded)
autoencoder.summary()
# exit()

# 모델 컴파일
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# MNIST 손글씨 데이터셋을 이용한다
(x_train, _), (x_test, _) = mnist.load_data()

# 전처리
# 스케일링
x_train = x_train / 255
x_test = x_test / 255

# reshaping
conv_x_train = x_train.reshape(-1, 28, 28, 1)
conv_x_test = x_test.reshape(-1, 28, 28, 1)
print(conv_x_train.shape)
print(conv_x_test.shape)

# exam_13 파일에서 가져 온 것을 수정함
noise_factor = 0.5
conv_x_train_noisy = conv_x_train + np.random.normal(
    loc=0.0, scale=1.0, size=conv_x_train.shape) * noise_factor # 노이즈를 학습 데이터셋에 더하기
conv_x_train_noisy = np.clip(conv_x_train_noisy, 0.0, 1.0) # 0보다 작으면 0으로 저장, 1보다 크면 1로 저장
conv_x_test_noisy = conv_x_test + np.random.normal(
    loc=0.0, scale=1.0, size=conv_x_test.shape) * noise_factor # 노이즈를 테스트 데이터셋에 더하기
conv_x_test_noisy = np.clip(conv_x_test_noisy, 0.0, 1.0) # 0보다 작으면 0으로 저장, 1보다 크면 1로 저장




# 모델 학습
fit_hist = autoencoder.fit(conv_x_train_noisy, conv_x_train,
                           epochs=50, batch_size=256,
                           validation_data=(conv_x_test_noisy, conv_x_test)) # 노이즈 추가

# 인코딩과 디코딩
decoded_img = autoencoder.predict(conv_x_test_noisy[:10]) # autoencoder로 예측한 결과

# 결과를 시각화
n = 10
plt.figure(figsize=(20, 4))
# autoencoder에 데이터를 넣어본 결과
for i in range(n):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i])
    # x축과 y축 선을 비표시
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 1 + n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    # x축과 y축 선을 비표시
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# 손실 플롯
plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show()

# 모델 저장
autoencoder.save('./models/autoencoder_noisy.h5')