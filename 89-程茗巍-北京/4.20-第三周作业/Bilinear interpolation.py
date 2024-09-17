import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def Bilinear(file, new_width, new_height):
    im = Image.open(file)

    #获取图像高度、宽度和每个通道数据
    width, height = im.size
    r, g, b = im.split()
    r_data, g_data, b_data = r.getdata(), g.getdata(), b.getdata()
    r_data = np.array(r_data).reshape(width, height)
    g_data = np.array(g_data).reshape(width, height)
    b_data = np.array(b_data).reshape(width, height)
    # scale_factor = max(new_width/width, new_height/height)

    #生成新三通道图像的大小
    new_img_r = np.zeros((new_width, new_height),dtype=np.uint8)
    new_img_g = np.zeros((new_width, new_height),dtype=np.uint8)
    new_img_b = np.zeros((new_width, new_height),dtype=np.uint8)

    for i in range(new_width):
        for j in range(new_height):
            x = (j + 0.5) / new_width * width - 0.5
            y = (i + 0.5) / new_height * height - 0.5
            #将目标图像坐标映射到原图坐标系下
            # x = x/scale_factor
            # y = y/scale_factor


            #找到周围四个像素点坐标和颜色
            x0 = int(x)
            y0 = int(y)
            x1, y1 = x0 + 1, y0 + 1
            r00, g00, b00 = r_data[x0, y0], g_data[x0, y0], b_data[x0, y0]
            r01, g01, b01 = r_data[x0, y1], g_data[x0, y1], b_data[x0, y1]
            r10, g10, b10 = r_data[x1, y0], g_data[x1, y0], b_data[x1, y0]
            r11, g11, b11 = r_data[x1, y1], g_data[x1, y1], b_data[x1, y1]

            #根据插值计算像素点颜色
            a = x - x0
            b = y - y0
            r_new = int((1 - a) * (1 - b) * r00 + a * (1 - b) * r10 + (1 - a) * b * r01 + a * b * r11)
            g_new = int((1 - a) * (1 - b) * g00 + a * (1 - b) * g10 + (1 - a) * b * g01 + a * b * g11)
            b_new = int((1 - a) * (1 - b) * b00 + a * (1 - b) * b10 + (1 - a) * b * b01 + a * b * b11)
            new_img_r[j, i] = r_new
            new_img_g[j, i] = g_new
            new_img_b[j, i] = b_new

    new_img_r = Image.fromarray(new_img_r)
    new_img_g = Image.fromarray(new_img_g)
    new_img_b = Image.fromarray(new_img_b)
    new_img = Image.merge('RGB',[new_img_r, new_img_g, new_img_b])

    return new_img



if __name__ =='__main__':
    filename = 'lenna.png'
    im = Image.open(filename)

    new_width = 256
    new_height = 256
    bilinear_im = Bilinear(filename,new_width,new_height)
    # 显示原始图像和处理后的图像
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(im)
    ax[1].imshow(bilinear_im)
    plt.show()