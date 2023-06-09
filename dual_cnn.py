# 导入keras相关的模块
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, Multiply
from keras.layers import BatchNormalization, Activation
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
import matplotlib.pyplot as plt


# 定义噪声估计网络
def noise_estimation_network(input_img):
    # 使用三个卷积层，每个卷积层后面跟着一个批量归一化层和一个激活层
    x = Conv2D(32, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    # 输出噪声图，使用sigmoid激活函数将其限制在[0,1]区间
    noise_map = Activation('sigmoid')(x)  # 反应每个像素的噪声程度
    return noise_map


# 定义去噪网络
def denoising_network(input_img):
    # 使用一个卷积层，后面跟着一个批量归一化层和一个激活层
    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # 分成两个分支，每个分支使用四个卷积层和四个转置卷积层，交替进行特征提取和上采样
    branch1 = Conv2D(64, (3, 3), padding='same')(x)
    branch1 = BatchNormalization()(branch1)
    branch1 = Activation('relu')(branch1)
    branch1 = Conv2DTranspose(64, (3, 3), padding='same')(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = Activation('relu')(branch1)
    branch1 = Conv2D(64, (3, 3), padding='same')(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = Activation('relu')(branch1)
    branch1 = Conv2DTranspose(64, (3, 3), padding='same')(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = Activation('relu')(branch1)

    branch2 = Conv2D(64, (3, 3), padding='same')(x)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)
    branch2 = Conv2DTranspose(64, (3, 3), padding='same')(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)
    branch2 = Conv2D(64, (3, 3), padding='same')(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)
    branch2 = Conv2DTranspose(64, (3, 3), padding='same')(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)

    # 将两个分支进行拼接，得到一个更丰富的特征图
    x = Concatenate()([branch1, branch2])

    # 使用一个转置卷积层，后面跟着一个批量归一化层和一个激活层，输出去噪后的图像，使用sigmoid激活函数将其限制在[0,1]区间
    x = Conv2DTranspose(1, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    denoised_img = Activation('sigmoid')(x)
    return denoised_img


# 定义注意力模块
def attention_module(input_img, noise_map):
    # 将输入图像和噪声图进行拼接，得到一个两通道的特征图
    x = Concatenate()([input_img, noise_map])
    # 使用一个卷积层，后面跟着一个批量归一化层和一个激活层，输出一个注意力图，使用sigmoid激活函数将其限制在[0,1]区间
    x = Conv2D(1, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    attention_map = Activation('sigmoid')(x)
    # 将注意力图和输入图像进行逐元素相乘，得到一个加权的输入图像
    weighted_input = Multiply()([attention_map, input_img])
    return weighted_input


# 定义双卷积神经网络
def dual_convolutional_network():
    # 创建一个输入层，假设输入图像的形状是(28, 28, 1)，即28x28的灰度图像
    input_img = Input(shape=(28, 28, 1))
    # 使用噪声估计网络得到噪声图
    noise_map = noise_estimation_network(input_img)
    # 使用注意力模块得到加权的输入图像
    weighted_input = attention_module(input_img, noise_map)
    # 使用去噪网络得到去噪后的图像
    denoised_img = denoising_network(weighted_input)
    # 构建模型，并返回
    model = Model(input_img, denoised_img)
    return model


# 加载MNIST数据集，并进行预处理
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.  # 归一化到[0,1]区间
x_test = x_test.astype('float32') / 255.
x_train = x_train[..., tf.newaxis]  # 增加一个维度，表示通道数为1
x_test = x_test[..., tf.newaxis]

# 给训练集和测试集添加高斯噪声，作为模型的输入
noise_factor = 0.3  # 噪声因子，控制噪声的强度
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)
x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)  # 将噪声图像裁剪到[0,1]区间
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)
x_train_noisy = x_train_noisy.numpy()
x_test_noisy = x_test_noisy.numpy()

# 调用双卷积神经网络函数，得到模型对象
model = dual_convolutional_network()
# 打印模型的结构和参数
# model.summary()
# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())  # 使用Adam优化器和均方误差损失函数
# 训练模型，使用有噪声的图像作为输入，无噪声的图像作为输出
model.fit(x_train_noisy, x_train,
          epochs=10,
          batch_size=128,
          shuffle=True,
          validation_data=(x_test_noisy, x_test))
# 保存模型
model.save('my_dncnn_model')

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
