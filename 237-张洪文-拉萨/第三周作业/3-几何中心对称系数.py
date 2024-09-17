

"""
1、按比例对应公式：
src_x = dst_x * (src_width / dst_width)
src_y = dst_y * (src_height / dst_height)

2、几何中心重合公式：
src_x + 0.5 = (dst_x + 0.5) * (src_width / dst_width)
src_y + 0.5 = (dst_y + 0.5) * (src_height / dst_height)

3、几何中心公式：最小坐标和最大坐标的平均值 (min_coordinate + max_coordinate) / 2
center_x = (min_x + max_x) / 2
center_y = (min_y + max_y) / 2

针对以上的公式我们可以进行代入, 设:
src_width = 4
src_height = 4
dst_width = 8
dst_height = 8

由几何中心公式可得, 源图像和目标图像得几何中心（4x4的图像，最小坐标为(0, 0)，最大坐标为(3, 3)）：
src_center = ((0+3)/2, (0+3)/2) = (1.5, 1.5)
dst_center = ((0+7)/2, (0+7)/2) = (3.5, 3.5)

将得到的几何中心代入几何中心重合公式中：
1.5 + 0.5 = (3.5 + 0.5) * (4 / 8)
2 = 4 * 1/2
2 = 2
可得几何中心重合公式成立，系数 0.5 无误.

"""
