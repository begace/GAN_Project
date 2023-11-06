import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, LeakyReLU, BatchNormalization, Conv2D, UpSampling2D, Dropout, ZeroPadding2D
from keras.datasets import mnist
import os

from tqdm import tqdm

OUT_DIR = './DNN_out/'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

img_shape = (28, 28, 1)
epochs = 5000
batch_size = 512  # 배치 크기를 10배로 조정
noise = 100
sample_interval = 100

(X_train, _), (_, _) = mnist.load_data()
X_train = X_train / 127.5 - 1  # 데이터를 -1에서 1 사이의 값으로 정규화
X_train = np.expand_dims(X_train, axis=3)

# 생성자 모델 정의
generator = Sequential()
generator.add(Dense(128, input_dim=noise))
generator.add(LeakyReLU(alpha=0.01))
# 여기에 배치 정규화 추가
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(784, activation='tanh'))
generator.add(Reshape(img_shape))

# generator = Sequential()
# generator.add(Dense(256 * 7 * 7, input_dim=noise))
# generator.add(LeakyReLU(alpha=0.01))
# generator.add(BatchNormalization(momentum=0.8))
# generator.add(Reshape((7, 7, 256)))
# generator.add(UpSampling2D())
# generator.add(Conv2D(128, kernel_size=3, padding='same'))
# generator.add(LeakyReLU(alpha=0.01))
# generator.add(BatchNormalization(momentum=0.8))
# generator.add(UpSampling2D())
# generator.add(Conv2D(64, kernel_size=3, padding='same'))
# generator.add(LeakyReLU(alpha=0.01))
# generator.add(BatchNormalization(momentum=0.8))
# generator.add(Conv2D(1, kernel_size=3, padding='same', activation='tanh'))
generator.summary()

# 판별자 모델 정의
discriminator = Sequential()
discriminator.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
discriminator.add(LeakyReLU(alpha=0.01))
discriminator.add(Dropout(0.5))
discriminator.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
discriminator.add(ZeroPadding2D(padding=((0,1),(0,1))))
discriminator.add(BatchNormalization(momentum=0.8))
discriminator.add(LeakyReLU(alpha=0.01))
discriminator.add(Dropout(0.25))
discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
discriminator.add(LeakyReLU(alpha=0.01))
discriminator.add(Dropout(0.5))
discriminator.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
discriminator.add(LeakyReLU(alpha=0.01))
discriminator.add(Dropout(0.25))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.summary()

# 판별자 모델 컴파일
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# GAN 모델 정의
gan_model = Sequential()
gan_model.add(generator)
gan_model.add(discriminator)
gan_model.compile(loss='binary_crossentropy', optimizer='adam')

# 훈련 루프
real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in tqdm(range(epochs)):
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]

    z = np.random.normal(0, 1, (batch_size, noise))
    fake_imgs = generator.predict(z)

    d_hist_real = discriminator.train_on_batch(real_imgs, real)
    d_hist_fake = discriminator.train_on_batch(fake_imgs, fake)

    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake)

    z = np.random.normal(0, 1, (batch_size, noise))
    gan_hist = gan_model.train_on_batch(z, real)

    if epoch % sample_interval == 0:
        print('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]' % (epoch, d_loss, d_acc*100, gan_hist))
        row = col = 4
        z = np.random.normal(0, 1, (row * col, noise))
        fake_imgs = generator.predict(z)
        fake_imgs = 0.5 * fake_imgs + 0.5  # 이미지를 [0, 1] 범위로 스케일링
        _, axs = plt.subplots(row, col, figsize=(row, col), sharey=True, sharex=True)
        cnt = 0
        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(fake_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        path = os.path.join(OUT_DIR, 'mnist_%d.png' % epoch)
        plt.savefig(path)
        plt.close()
