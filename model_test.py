# 导入所需的库
import cv2
import keras
import numpy as np

# 加载模型
model = keras.models.load_model('my_dncnn_model')
model.trainable = False

# 读取图片
img = cv2.imread('image4.jpg', 0)

# 图像补齐
img = cv2.resize(img, (280, 280))

# 归一化图片
img = img.astype('float32') / 255.0

# 增加一个维度，以适应模型的输入
# img = np.expand_dims(img, axis=0)
# img = np.expand_dims(img, axis=-1)

# 图片切片
slices = []
for i in range(10):
    for j in range(10):
        slice = cv2.getRectSubPix(img, (28, 28), (14 + j * 28, 14 + i * 28))
        slices.append(slice)
slices = np.array(slices)
slices = np.expand_dims(slices, axis=-1)  # shape = [100, 28, 28, 1]

# 使用模型去噪
decode_slices = model.predict(slices)

# 还原图像
rows = []
for i in range(10):
    # 定义一个空列表，用于存放每一行的图像
    row = []
    # 遍历每一列
    for j in range(10):
        # 获取当前位置的图像
        imgs = decode_slices[i * 10 + j]
        # 将当前位置的图像添加到行列表中
        row.append(imgs)
    # 使用cv2.hconcat()函数将行列表中的所有图像水平拼接起来
    row = cv2.hconcat(row)
    # 将拼接后的行添加到总列表中
    rows.append(row)

# 使用cv2.vconcat()函数将总列表中的所有行垂直拼接起来
decode_img = cv2.vconcat(rows)
# print(decode_img.shape)

# 从图片中减去噪声残差，得到去噪后的图片
# denoised = img - noise

# 去掉多余的维度
# decode_img = np.squeeze(decode_img, axis=0)
# decode_img = np.squeeze(decode_img, axis=-1)

# 反归一化图片
# decode_img = decode_img * 255.0
# decode_img = decode_img.astype('uint8')

# 显示原始图片和去噪后的图片
cv2.imshow('original image', img)
cv2.imshow('denoised image', decode_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
