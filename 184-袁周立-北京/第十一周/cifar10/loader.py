import os
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def parse_data_dict(data_dict_list):
    data_concat = []
    label_concat = []
    for data_dict in data_dict_list:
        data = data_dict[b"data"].reshape((-1, 3, 32, 32)) / 255   # n,c,h,w
        data_concat.append(np.transpose(data, (0, 2, 3, 1)))   # n,h,w,c
        label_concat.append(data_dict[b"labels"])
    trade_data = np.concatenate(data_concat, axis=0)
    trade_label = np.concatenate(label_concat, axis=0)
    return trade_data, trade_label


def load_data(data_path):
    train_data_files = [os.path.join(data_path, "data_batch_{}".format(i)) for i in range(1, 6)]
    train_data_dict_list = [unpickle(e) for e in train_data_files]
    train_data, train_label = parse_data_dict(train_data_dict_list)

    test_data_dict = unpickle(os.path.join(data_path, "test_batch"))
    test_data, test_label = parse_data_dict([test_data_dict])

    return train_data, train_label, test_data, test_label


def load_meta(meta_path):
    label_names = unpickle(meta_path)[b"label_names"]
    label_names = [str(e, encoding="utf-8") for e in label_names]
    meta_dict = {i: v for i, v in enumerate(label_names)}
    return meta_dict


if __name__ == "__main__":
    load_data("./cifar-10-batches-py")
    load_meta("./cifar-10-batches-py/batches.meta")