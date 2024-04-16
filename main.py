# Blind watermark based on DWT-DCT-SVD.

import cv2
from blind_watermark import WaterMark
import os

os.chdir(os.path.dirname(__file__))

def embed_watermark_mode_string(secret_text, path_img_input, path_img_output):
    bwm1 = WaterMark(password_img=1, password_wm=1)
    bwm1.read_img(path_img_input)
    wm = secret_text
    # Đọc nó dưới dạng một bức ảnh sáng hoặc chứa bit 1 hoặc 0
    bwm1.read_wm(wm, mode='str')
    bwm1.embed(path_img_output)
    print(bwm1.wm_bit)
    len_wm = len(bwm1.wm_bit)
    print('Put down the length of wm_bit {len_wm}'.format(len_wm=len_wm))
    return len_wm
    
def extract_watermark_mode_string(len_wm, path_img_output):
    # Khởi tạo thủy vân
    bwm1 = WaterMark(password_img=1, password_wm=1)
    # len_wm: chỉ định kích thước hoặc độ dài của watermark cần được trích xuất từ hình ảnh
    # len_wm cung cấp thông tin cần thiết để hàm trích xuất có thể xác định và tái tạo chính xác nội dung của watermark từ hình ảnh.
    wm_extract = bwm1.extract(path_img_output, wm_shape=len_wm, mode='str')
    print(f"Output: {wm_extract}")
    
def embed_watermark_mode_img(path_watermark_img, path_img_input, path_img_output):
    bwm1 = WaterMark(password_wm=1, password_img=1)
    bwm1.read_img(path_img_input)
    bwm1.read_wm(path_watermark_img, mode="img")
    bwm1.embed(path_img_output, compression_ratio=0)
    
def extract_watermark_mode_img(path_embedded_img, path_img_extract, path_watermark_img):
    # Khởi tạo thủy vân
    bwm1 = WaterMark(password_img=1, password_wm=1)
    # Mục đích của đoạn code này là để lấy kích thước của ảnh watermark (chiều cao và chiều rộng): (128, 128)
    wm_shape = cv2.imread(path_watermark_img, flags=cv2.IMREAD_GRAYSCALE).shape
    print(3333, wm_shape)
    wm_extract = bwm1.extract(path_embedded_img, wm_shape=wm_shape, out_wm_name=path_img_extract, mode='img')
    print(f"Output: {wm_extract}")



def main():
    # Mode Str
    PATH_IMAGE_INPUT = "assets/image_test.jpg"
    PATH_IMAGE_OUTPUT = "output/embedded.jpg"
    SECRET_TEXT = "This is secret key"
    # Step 1:
    len_wm = embed_watermark_mode_string(SECRET_TEXT, PATH_IMAGE_INPUT, PATH_IMAGE_OUTPUT)
    print(2222, len_wm)
    
    # Step 2
    # len_wm = 143
    # extract_watermark_mode_string(len_wm, PATH_IMAGE_OUTPUT)
    
    # Mode Img
    # PATH_IMAGE_INPUT = "assets/ori_img.jpeg"
    # PATH_IMAGE_EMBEDDED = "output/embedded_mode_img.png"
    # PATH_IMAGE_WATERMARK = "assets/watermark.png"
    # PATH_IMAGE_EXTRACT = "output/watermark_extract.png"
    
    # Step 1
    # embed_watermark_mode_img(PATH_IMAGE_WATERMARK, PATH_IMAGE_INPUT, PATH_IMAGE_EMBEDDED)
    # Step 2
    # extract_watermark_mode_img(PATH_IMAGE_EMBEDDED, PATH_IMAGE_EXTRACT, PATH_IMAGE_WATERMARK)
    
    
    return 1

if __name__ == "__main__":
    main()