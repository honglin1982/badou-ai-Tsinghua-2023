import random

import numpy as np
import cv2


class DiyGenerator:
    def __init__(self, path, image_name_format, total_data_num, valid_rate=0.1, batch_size=128, target_size=(227, 227)):
        assert 0 < valid_rate < 1
        self.base_path = path
        self.image_name_format = image_name_format
        self.total_data_num = total_data_num
        self.valid_rate = valid_rate
        self.batch_size = batch_size
        self.target_size = target_size
        self._gen_init()

    def _gen_init(self):
        train_num = int(self.total_data_num * (1 - self.valid_rate))

        index_list = np.arange(self.total_data_num)
        np.random.shuffle(index_list)

        train_index_list = index_list[:train_num]
        valid_index_list = index_list[train_num:]

        self.train_gen = self._gen_load(train_index_list, distorted=True)
        self.valid_gen = self._gen_load(valid_index_list, distorted=False)

    def _gen_load(self, index_list, distorted=False):
        i = 0
        num = len(index_list)
        while 1:
            x = []
            y = []
            for b in range(self.batch_size):
                if i == 0:
                    np.random.shuffle(index_list)
                img = cv2.imread(self.base_path + "/" + self.image_name_format(index_list[i]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.target_size)
                if distorted:
                    img = self._img_distorted(img)
                img = img / 255
                x.append(img)
                y.append(1 if index_list[i] > 12499 else 0)
                i = (i + 1) % num
            x = np.array(x)
            y = np.array(y)
            yield x, y

    def _img_distorted(self, img):
        # 随机调整亮度
        img = np.uint8(np.clip(((0.9 + np.random.random() / 5) * img), 0, 255))
        return img

    def get_train_gen(self):
        return self.train_gen

    def get_test_gen(self):
        return self.valid_gen


if __name__ == "__main__":
    name_format = lambda index: "{}.{}.jpg".format("dog", index - 12499) if index > 12499 \
        else "{}.{}.jpg".format("cat", index)
    diyGenerator = DiyGenerator("./image/train", name_format, 25000, valid_rate=0.1)
