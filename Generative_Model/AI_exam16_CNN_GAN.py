import matplotlib.pyplot as plt
import numpy as np
import os
from keras.datasets import mnist
from keras.models import *
from keras.layers import *

OUT_DIR = './CNN_out'
img_shape = (28, 28, 1)
epochs = 100000
batch_size = 128
noise = 100
sample_interval = 100

(x_train, _), (_, _) = mnist.load_data()
print(x_train.shape)

# 전처리
x_train = x_train / 127.5 - 1 # 127.5로 나눈 뒤 1을 뺌 -> x_train의 범위가 -1부터 1까지가 됨
x_train = np.expand_dims(x_train, axis=3)
print(x_train.shape)

# GAN 모델 만들기 ---------------------------------
# 생성자(Generator)
generator = Sequential()
generator.add(Dense(256 * 7 * 7, input_dim=noise))
generator.add(Reshape((7, 7, 256)))
generator.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(alpha=0.01))
generator.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(alpha=0.01))
generator.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))
generator.add(Activation('tanh'))
generator.summary()

# 판별자(Discriminator)
discriminator = Sequential()
discriminator.add(Conv2D(32, kernel_size=3, strides=2, padding='same', input_shape=img_shape))
discriminator.add(LeakyReLU(alpha=0.01))
discriminator.add(Conv2D(32, kernel_size=3, strides=2, padding='same'))
discriminator.add(LeakyReLU(alpha=0.01))
discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
discriminator.add(LeakyReLU(alpha=0.01))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.summary()
discriminator.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

# GAN 모델로 합치기
gan_model = Sequential()
gan_model.add(generator)
gan_model.add(discriminator)
gan_model.summary()

gan_model.compile(loss='binary_crossentropy', optimizer='adam')
discriminator.trainable = False # 일단 discriminator의 학습을 막는다

# 정답(target) 행렬 만들기
real = np.ones((batch_size, 1)) # [1.]이 128개 들어 있는 행렬
fake = np.zeros((batch_size, 1)) # [0.]이 128개 들어 있는 행렬
print(real)
print(fake)

# 학습
for epoch in range(epochs):
    idx = np.random.randint(0, x_train.shape[0], batch_size) # 랜덤하게 학습 데이터마다 인덱스 부여하기
    real_img = x_train[idx]

    z = np.random.normal(0, 1, (batch_size, noise)) # 가짜 이미지를 128개 생성
    fake_img = generator.predict(z)

    # epoch 당 학습 빈도를 조절하지 않도록 했다.
    # if epoch % 7 == 0: # 이 때는 판별자만 학습시키기(학습 속도의 균형이 깨지지 않도록... 판별자만 너무 학습을 잘할까봐...)
    discriminator.trainable = True # 판별자가 학습이 가능하도록
    # train_on_batch() : 받은 데이터를 이용해서 1회 학습함
    d_hist_real = discriminator.train_on_batch(real_img, real)
    d_hist_fake = discriminator.train_on_batch(fake_img, fake)
    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake) # real과 fake의 각 손실값의 평균 내기

    # 노이즈를 다시 만들어서 GAN 모델에게 주다
    z = np.random.normal(0, 1, (batch_size, noise))
    discriminator.trainable = False # 판별자 학습 차단
    gan_hist = gan_model.train_on_batch(z, real) # 1회 학습

    if epoch % sample_interval == 0: # sample_interval마다 한 번씩
        print('%d [D loss: %f, acc: %.2f%%] [G loss:%f]'%(
            epoch, d_loss, d_acc * 100, gan_hist))
        row = col = 4
        z = np.random.normal(0, 1, (row * col, noise)) # row * col개의 노이즈 샘플 만들기
        fake_imgs = generator.predict(z) # 가짜 이미지 만들기
        fake_imgs = 0.5 * fake_imgs # tanh로 나온 이미지의 값 범위가 -1부터 1(간격이 2)이기 때문에 이를 -0.5부터 0.5로 줄임 -> 이미지의 밝기가 감소함
        #(사실 0.5를 곱하는 건 matplotlib가 알아서 해 주기 때문에 굳이 안 해도 된다고 교수님께서 알려 주심)
        _, axs = plt.subplots(row, col, figsize=(row, col),
                              sharex=True, sharey=True)
        count = 0
        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(fake_imgs[count, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                count += 1
        path = os.path.join(OUT_DIR, 'img-{}'.format(epoch))
        plt.savefig(path) # 가짜 이미지 저장
        plt.close()
        generator.save('./models/generator.h5')

