#!/usr/bin/env python3
# coding=utf-8
# @Time    : 2021/12/17
# @Author  : github.com/guofei9987
import numpy as np
from numpy.linalg import svd
import copy
import cv2
from cv2 import dct, idct
from pywt import dwt2, idwt2
from .pool import AutoPool


class WaterMarkCore:
    def __init__(self, password_img=1, mode='common', processes=None):
        self.block_shape = np.array([4, 4])
        self.password_img = password_img
        self.d1, self.d2 = 36, 20  # d1/d2 越大鲁棒性越强,但输出图片的失真越大

        # init data
        self.img, self.img_YUV = None, None  # self.img 是原图，self.img_YUV 对像素做了加白偶数化
        self.ca, self.hvd, = [np.array([])] * 3, [np.array([])] * 3 # Kết quả dct cho từng kênh
        self.ca_block = [np.array([])] * 3  # 每个 channel 存一个四维 array，代表四维分块后的结果
        self.ca_part = [np.array([])] * 3  # 四维分块后，有时因不整除而少一部分，self.ca_part 是少这一部分的 self.ca

        self.wm_size, self.block_num = 0, 0  # 水印的长度，原图片可插入信息的个数
        self.pool = AutoPool(mode=mode, processes=processes)

        self.fast_mode = False
        self.alpha = None  # 用于处理透明图

    def init_block_index(self):
        self.block_num = self.ca_block_shape[0] * self.ca_block_shape[1]
        assert self.wm_size < self.block_num, IndexError(
            '最多可嵌入{}kb信息，多于水印的{}kb信息，溢出'.format(self.block_num / 1000, self.wm_size / 1000))
        # self.part_shape 是取整后的ca二维大小,用于嵌入时忽略右边和下面对不齐的细条部分。
        self.part_shape = self.ca_block_shape[:2] * self.block_shape
        self.block_index = [(i, j) for i in range(self.ca_block_shape[0]) for j in range(self.ca_block_shape[1])]

    def read_img_arr(self, img):
        # 处理透明图
        self.alpha = None
        if img.shape[2] == 4:
            if img[:, :, 3].min() < 255:
                self.alpha = img[:, :, 3]
                img = img[:, :, :3]

        # 读入图片->YUV化->加白边使像素变偶数->四维分块
        self.img = img.astype(np.float32)
        self.img_shape = self.img.shape[:2]

        # 如果不是偶数，那么补上白边，Y（明亮度）UV（颜色）
        self.img_YUV = cv2.copyMakeBorder(cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV),
                                          0, self.img.shape[0] % 2, 0, self.img.shape[1] % 2,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))

        self.ca_shape = [(i + 1) // 2 for i in self.img_shape]

        self.ca_block_shape = (self.ca_shape[0] // self.block_shape[0], self.ca_shape[1] // self.block_shape[1],
                               self.block_shape[0], self.block_shape[1])
        strides = 4 * np.array([self.ca_shape[1] * self.block_shape[0], self.block_shape[1], self.ca_shape[1], 1])

        for channel in range(3):
            self.ca[channel], self.hvd[channel] = dwt2(self.img_YUV[:, :, channel], 'haar')
            # 转为4维度
            self.ca_block[channel] = np.lib.stride_tricks.as_strided(self.ca[channel].astype(np.float32),
                                                                     self.ca_block_shape, strides)

    def read_wm(self, wm_bit):
        self.wm_bit = wm_bit
        self.wm_size = wm_bit.size

    def block_add_wm(self, arg):
        if self.fast_mode:
            return self.block_add_wm_fast(arg)
        else:
            return self.block_add_wm_slow(arg)

    def block_add_wm_slow(self, arg):
        block, shuffler, i = arg
        # dct->(flatten->加密->逆flatten)->svd->打水印->逆svd->(flatten->解密->逆flatten)->逆dct
        wm_1 = self.wm_bit[i % self.wm_size]
        block_dct = dct(block)

        # 加密（打乱顺序）
        block_dct_shuffled = block_dct.flatten()[shuffler].reshape(self.block_shape)
        u, s, v = svd(block_dct_shuffled)
        s[0] = (s[0] // self.d1 + 1 / 4 + 1 / 2 * wm_1) * self.d1
        if self.d2:
            s[1] = (s[1] // self.d2 + 1 / 4 + 1 / 2 * wm_1) * self.d2

        block_dct_flatten = np.dot(u, np.dot(np.diag(s), v)).flatten()
        block_dct_flatten[shuffler] = block_dct_flatten.copy()
        return idct(block_dct_flatten.reshape(self.block_shape))

    def block_add_wm_fast(self, arg):
        # dct->svd->hình mờ->svd ngược->dct ngược
        block, shuffler, i = arg
        # Nhúng lần lượt các bit 1 hoặc 0 vào ảnh
        wm_1 = self.wm_bit[i % self.wm_size]

        u, s, v = svd(dct(block))
        s[0] = (s[0] // self.d1 + 1 / 4 + 1 / 2 * wm_1) * self.d1

        return idct(np.dot(u, np.dot(np.diag(s), v)))

    def embed(self):
        self.init_block_index()

        embed_ca = copy.deepcopy(self.ca)
        embed_YUV = [np.array([])] * 3

        self.idx_shuffle = random_strategy1(self.password_img, self.block_num,
                                            self.block_shape[0] * self.block_shape[1])
        for channel in range(3):
            # Nhúng thông tin thủy vân vào ảnh
            tmp = self.pool.map(self.block_add_wm,
                                [(self.ca_block[channel][self.block_index[i]], self.idx_shuffle[i], i)
                                 for i in range(self.block_num)])

            for i in range(self.block_num):
                self.ca_block[channel][self.block_index[i]] = tmp[i]

            # Chuyển đổi khối 4D trở lại 2D
            self.ca_part[channel] = np.concatenate(np.concatenate(self.ca_block[channel], 1), 1)
            # Khi chia thành các khối 4 chiều, các dải dài bên phải và phía dưới không thể chia được sẽ được giữ lại, 
            # phần còn lại là phần chính, được thay thế bằng dữ liệu miền tần số sau khi nhúng.
            embed_ca[channel][:self.part_shape[0], :self.part_shape[1]] = self.ca_part[channel]
            # Trở lại phép biến đổi nghịch đảo
            embed_YUV[channel] = idwt2((embed_ca[channel], self.hvd[channel]), "haar")

        # Hợp nhất 3 kênh
        embed_img_YUV = np.stack(embed_YUV, axis=2)
        # Nếu nó không phải là số nguyên bằng 2 thì đường viền màu trắng đã được thêm vào trước đó và nó sẽ bị xóa ở đây.
        embed_img_YUV = embed_img_YUV[:self.img_shape[0], :self.img_shape[1]]
        embed_img = cv2.cvtColor(embed_img_YUV, cv2.COLOR_YUV2BGR)
        embed_img = np.clip(embed_img, a_min=0, a_max=255)

        if self.alpha is not None:
            embed_img = cv2.merge([embed_img.astype(np.uint8), self.alpha])
        return embed_img

    def block_get_wm(self, args):
        if self.fast_mode:
            return self.block_get_wm_fast(args)
        else:
            return self.block_get_wm_slow(args)

    def block_get_wm_slow(self, args):
        block, shuffler = args
        # dct->flatten->加密->逆flatten->svd->解水印
        block_dct_shuffled = dct(block).flatten()[shuffler].reshape(self.block_shape)

        u, s, v = svd(block_dct_shuffled)
        wm = (s[0] % self.d1 > self.d1 / 2) * 1
        if self.d2:
            tmp = (s[1] % self.d2 > self.d2 / 2) * 1
            wm = (wm * 3 + tmp * 1) / 4
        return wm

    def block_get_wm_fast(self, args):
        block, shuffler = args
        # dct->svd->解水印
        u, s, v = svd(dct(block))
        wm = (s[0] % self.d1 > self.d1 / 2) * 1

        return wm

    def extract_raw(self, img):
        # Trích xuất 1 bit thông tin từ mỗi khối
        self.read_img_arr(img=img)
        self.init_block_index()

        wm_block_bit = np.zeros(shape=(3, self.block_num))  # 3 kênh, hình mờ có chiều dài được trích xuất theo khối, tất cả đều được ghi lại

        self.idx_shuffle = random_strategy1(seed=self.password_img,
                                            size=self.block_num,
                                            block_shape=self.block_shape[0] * self.block_shape[1],  # 16
                                            )
        for channel in range(3):
            wm_block_bit[channel, :] = self.pool.map(self.block_get_wm,
                                                     [(self.ca_block[channel][self.block_index[i]], self.idx_shuffle[i])
                                                      for i in range(self.block_num)])
        return wm_block_bit

    def extract_avg(self, wm_block_bit):
        # Nhúng vòng lặp trung bình + 3 kênh
        wm_avg = np.zeros(shape=self.wm_size)
        for i in range(self.wm_size):
            wm_avg[i] = wm_block_bit[:, i::self.wm_size].mean()
        return wm_avg

    def extract(self, img, wm_shape):
        self.wm_size = np.array(wm_shape).prod()

        #Trích xuất các bit được nhúng trong mỗi khối:
        wm_block_bit = self.extract_raw(img=img)
        # Làm trung bình:
        wm_avg = self.extract_avg(wm_block_bit)
        return wm_avg

    def extract_with_kmeans(self, img, wm_shape):
        wm_avg = self.extract(img=img, wm_shape=wm_shape)

        return one_dim_kmeans(wm_avg)


def one_dim_kmeans(inputs):
    threshold = 0
    e_tol = 10 ** (-6)
    center = [inputs.min(), inputs.max()]  #1. Khởi tạo điểm trung tâm
    for i in range(300):
        threshold = (center[0] + center[1]) / 2
        is_class01 = inputs > threshold  #2. Kiểm tra khoảng cách giữa tất cả các điểm và k điểm này và phân loại từng điểm đến tâm gần nhất
        center = [inputs[~is_class01].mean(), inputs[is_class01].mean()] #3. Tìm lại điểm trung tâm
        if np.abs((center[0] + center[1]) / 2 - threshold) < e_tol: #4. Điều kiện dừng
            threshold = (center[0] + center[1]) / 2
            break

    is_class01 = inputs > threshold
    return is_class01


def random_strategy1(seed, size, block_shape):
    return np.random.RandomState(seed) \
        .random(size=(size, block_shape)) \
        .argsort(axis=1)


def random_strategy2(seed, size, block_shape):
    one_line = np.random.RandomState(seed) \
        .random(size=(1, block_shape)) \
        .argsort(axis=1)

    return np.repeat(one_line, repeats=size, axis=0)
