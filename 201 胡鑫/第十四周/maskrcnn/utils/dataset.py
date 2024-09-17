import numpy as np
import skimage
import logging
import skimage.color
import skimage.io
import skimage.transform


# ----------------------------------------------------------#
#  Dataset
# ----------------------------------------------------------#
class Dataset(object):
    # 数据集训练的基本格式
    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # 背景作为第一分类
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    @property
    def image_ids(self):
        return self._image_ids

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # 用于增加新的分类
        for info in self.class_info:
            if info['source'] == source and info['id'] == class_id:
                return
        self.class_info.append({"source": source, "id": class_id, "name": class_name})

    def add_image(self, source, image_id, path, **kwargs):
        # 用于增加用于训练的图片
        image_info = {"id": image_id, "source": source, "path": path}
        image_info.update(**kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        return ""

    def prepare(self, class_map=None):
        # 准备数据
        def clean_name(name):
            """返回对象名称的较短版本，以便更清晰地显示。"""
            return ",".join(name.split(",")[:1])
        # 分多少类
        self.num_classes = len(self.class_info)
        # 种类的id
        self.class_ids = np.arange(self.num_classes)
        # 简称，用于显示
        self.class_names = [clean_name(c["name"]) for c in self.class_info]

        # 计算一共有多少图片
        self.num_images = len(self.image_info)

        # 图片的id
        self._image_ids = np.arange(self.num_images)

        # 从源类和图象id到内部id的映射
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # 建立sources
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}

        # 在数据集上循环
        for source in self.sources:
            self.source_class_ids[source] = []
            # 查找属于此数据集的类
            for i, info in enumerate(self.class_info):
                # 在所有数据集中包括BG类
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def load_image(self, image_id):
        """载入图片"""
        image = skimage.io.imread(self.image_info[image_id]['path'])
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        if image.shape[-1] == 4:
            image = image[..., :3]

        return image

    def load_mask(self, image_id):
        """载入语义分割部分"""
        logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids



