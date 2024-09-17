import cv2

"""
差值哈希
"""


def reduce_hash(img):
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    n_img = cv2.resize(g_img, (9, 8))
    ls = list()
    for i in range(8):
        for j in range(9):
            if j == 8:
                break
            if n_img[i, j + 1] > n_img[i, j]:
                ls.append(1)
            else:
                ls.append(0)
    return ls


"""
均值哈希
"""


def avg_hash(img):
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    n_img = cv2.resize(g_img, (8, 8))
    s = 0
    ls = list()
    for i in range(8):
        for j in range(8):
            s += n_img[i, j]
    avg = s / 64
    for i in range(8):
        for j in range(8):
            if n_img[i, j] > avg:
                ls.append(1)
            else:
                ls.append(0)
    return ls


def compare(h1, h2):
    if len(h1) != len(h2):
        raise ValueError("len must equal")
    diff = 0
    for i in range(len(h1)):
        if h1[i] != h2[i]:
            diff += 1
    return diff


if __name__ == "__main__":
    img = cv2.imread("lenna.png")
    noise_img = cv2.imread("lenna_noise.png")
    hash1 = reduce_hash(img)
    hash2 = reduce_hash(noise_img)
    print("差值哈希比较结果：", compare(hash1, hash2))
    hash1 = avg_hash(img)
    hash2 = avg_hash(noise_img)
    print("均值哈希比较结果：", compare(hash1, hash2))
