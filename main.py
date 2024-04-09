# Blind watermark based on DWT-DCT-SVD.


from blind_watermark import WaterMark

def embed_watermark(secret_text, path_img_input, path_img_output):
    bwm1 = WaterMark(password_img=1, password_wm=1)
    bwm1.read_img(path_img_input)
    wm = secret_text
    bwm1.read_wm(wm, mode='str')
    bwm1.embed(path_img_output)
    print(bwm1.wm_bit)
    len_wm = len(bwm1.wm_bit)
    print('Put down the length of wm_bit {len_wm}'.format(len_wm=len_wm))
    return len_wm
    
def extract_watermark(len_wm, path_img_output):
    # Khởi tạo thủy vân
    bwm1 = WaterMark(password_img=1, password_wm=1)
    wm_extract = bwm1.extract(path_img_output, wm_shape=len_wm, mode='str')
    print(f"Output: {wm_extract}")

def main():
    PATH_IMAGE_INPUT = "assets/image_test.jpg"
    PATH_IMAGE_OUTPUT = "output/embedded.png"
    SECRET_TEXT = "This is secret key"
    # Step 1:
    len_wm = embed_watermark(SECRET_TEXT, PATH_IMAGE_INPUT, PATH_IMAGE_OUTPUT)
    print(2222, len_wm)
    
    # Step 2
    # len_wm = 143
    # extract_watermark(len_wm, PATH_IMAGE_OUTPUT)
    return 1

if __name__ == "__main__":
    main()