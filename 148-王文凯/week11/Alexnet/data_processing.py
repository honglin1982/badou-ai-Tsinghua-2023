import os
import numpy as np
import cv2


def get_file(file_dir):
    images = []
    folders = []
    for root, sub_folders, files in os.walk(file_dir):
        for file in files:
            images.append(os.path.join(root, file))
        for folder in sub_folders:
            folders.append(os.path.join(root, folder))

    labels = []
    for one_folder in folders:
        num_img = len(os.listdir(one_folder))  # 统计one_folder下包含多少个文件
        label = one_folder.split('\\')[-1]
        # print(label)
        if label == 'cats':
            labels = np.append(labels, num_img * [0])
        else:
            labels = np.append(labels, num_img * [1])

    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    images_list = []
    for image in image_list:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images_list.append(img)
    images_list = np.array(images_list)
    label_list = list(temp[:, 1])
    label_list = np.array([int(i.split('.')[0]) for i in label_list])
    return images_list, label_list


# if __name__ == "__main__":
#     image_list, label_list = get_file('./data/image/train')
#     print(image_list.shape)
