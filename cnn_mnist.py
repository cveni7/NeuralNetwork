# 导入必要的库
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.datasets import mnist  # 使用MNIST数据集作为示例
import matplotlib.pyplot as plt


# 定义CNN模型，这里使用了一个简单的编码器-解码器结构
def get_cnn_model():
    input_img = Input(shape=(28, 28, 1))  # 输入图像的形状，这里是28x28的灰度图像
    # 编码器部分，使用卷积层提取特征，并降低图像的尺寸
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    # 解码器部分，使用转置卷积层恢复图像的尺寸，并输出去噪后的图像
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)
    # 构建模型，并返回
    model = Model(input_img, decoded)
    return model


# 加载MNIST数据集，并进行预处理
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.  # 归一化到[0,1]区间
x_test = x_test.astype('float32') / 255.
x_train = x_train[..., tf.newaxis]  # 增加一个维度，表示通道数为1
x_test = x_test[..., tf.newaxis]

# 给训练集和测试集添加高斯噪声，作为模型的输入
noise_factor = 0.2  # 噪声因子，控制噪声的强度
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)
x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)  # 将噪声图像裁剪到[0,1]区间
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)
x_train_noisy = x_train_noisy.numpy()
x_test_noisy = x_test_noisy.numpy()

# 获取CNN模型，并编译
model = get_cnn_model()
model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())  # 使用Adam优化器和均方误差损失函数

# 训练模型，使用有噪声的图像作为输入，无噪声的图像作为输出
model.fit(x_train_noisy, x_train,
          epochs=10,
          batch_size=128,
          shuffle=True,
          validation_data=(x_test_noisy, x_test))

# 使用测试集中的有噪声的图像，预测去噪后的图像
decoded_imgs = model.predict(x_test_noisy)

# 可视化原始图像、有噪声的图像和去噪后的图像
n = 10  # 显示10个样本
plt.figure(figsize=(20, 6))
for i in range(n):
    # 原始图像
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.title("original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 有噪声的图像
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.title("noisy")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 去噪后的图像
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.title("denoised")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
