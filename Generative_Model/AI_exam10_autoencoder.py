import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.datasets import mnist

# encoder
input_img = Input(shape=(784, ))
encoded = Dense(32, activation='relu')
encoded = encoded(input_img)

# decoder
decoded = Dense(784, activation='sigmoid')
decoded = decoded(encoded)

# autoencoder 구축
autoencoder = Model(input_img, decoded)
autoencoder.summary()

# encoder 확인
encoder = Model(input_img, encoded)
encoder.summary()

# decoder 확인
encoder_input = Input(shape=(32, ))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoder_input, decoder_layer(encoder_input))
decoder.summary()

# 모델 컴파일
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# MNIST 손글씨 데이터셋을 이용한다
(x_train, _), (x_test, _) = mnist.load_data()

# 전처리
# 스케일링
x_train = x_train / 255
x_test = x_test / 255

# 한 줄로 평활화
flatted_x_train = x_train.reshape(-1, 28 * 28)
flatted_x_test = x_test.reshape(-1, 28 * 28)
print(flatted_x_train.shape)
print(flatted_x_test.shape)

# 모델 학습
fit_hist = autoencoder.fit(flatted_x_train, flatted_x_train,
                           epochs=50, batch_size=256,
                           validation_data=(flatted_x_test, flatted_x_test))

# 인코딩과 디코딩
encoded_img = encoder.predict(x_test[:10].reshape(-1, 784)) # 32차원 데이터를 784차원으로 reshaping
decoded_img = decoder.predict(encoded_img)

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