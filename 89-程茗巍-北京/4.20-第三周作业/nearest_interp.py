import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def nearest_interp(file,new_width, new_height):
    #打开图像
    im = Image.open(file)

    #获取图像高度、宽度和每个通道数据
    width, height = im.size
    r, g, b = im.split()
    r_data, g_data, b_data = r.getdata(), g.getdata(), b.getdata()
    r_data = np.array(r_data).reshape(width, height)
    g_data = np.array(g_data).reshape(width, height)
    b_data = np.array(b_data).reshape(width, height)

    #生成新三通道图像的大小
    new_img_r = np.zeros((new_width, new_height),dtype=np.uint8)
    new_img_g = np.zeros((new_width, new_height),dtype=np.uint8)
    new_img_b = np.zeros((new_width, new_height),dtype=np.uint8)


    for i in range(new_width):
        for j in range(new_height):
            x = int(i / new_width * width + 0.5)
            y = int(j / new_height * height + 0.5)
            new_img_r[i, j] = r_data[x, y]
            new_img_g[i, j] = g_data[x, y]
            new_img_b[i, j] = b_data[x, y]

    new_img_r = Image.fromarray(new_img_r)
    new_img_g = Image.fromarray(new_img_g)
    new_img_b = Image.fromarray(new_img_b)
    new_img = Image.merge('RGB',[new_img_r, new_img_g, new_img_b])

    return new_img

if __name__ =='__main__':
    filename = 'lenna.png'
    im = Image.open(filename)

    new_width = 768
    new_height = 768
    channel = 3
    nearest_interp_im = nearest_interp(filename, new_width, new_height)
    # 显示原始图像和处理后的图像
    fig, ax = plt.subplots(1, 2)
    print(ax)
    ax[0].imshow(im)
    ax[1].imshow(nearest_interp_im)
    plt.show()