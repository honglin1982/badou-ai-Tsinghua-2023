import numpy as np
import matplotlib.pyplot as plt

'''
用keras实现简单神经网络
从零开始实现神经网络

注：
    以下代码loss稳步下降
    预测时候的loss也比较低
    但效果很差
    
    原因：
        使用sigmoid和交叉熵做loss，导致最后每一个类别的预测概率都很高，但交叉熵算出来的loss却很低
        所以还是得softmax来配合交叉熵
'''


class DIY_Model:
    def __init__(self, input_size, hidden_size, output_size, lr):
        self.wih = np.random.rand(input_size, hidden_size) - 0.5
        self.who = np.random.rand(hidden_size, output_size) - 0.5
        self.lr = lr
        self.activation = self.__diy_sigmoid
        self.cross_entropy = self.__diy_cross_entropy

    def __diy_sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    def __diy_cross_entropy(self, input, y_true):
        self.delta = 1e-7
        return -np.sum(y_true * np.log(input + self.delta), axis=-1)

    def train(self, input, y_true):  # (1, input_size)

        hidden = np.dot(input, self.wih)  # (1, hidden_size)
        hidden_activate = self.activation(hidden)
        output = np.dot(hidden_activate, self.who)  # (1, output_size)
        output_activate = self.activation(output)
        loss = self.cross_entropy(output_activate, y_true)
        print("loss: ", loss)

        # 偏导如下
        loss_to_output_activate = -y_true / (output_activate + self.delta)  # (1, output_size)
        output_activate_to_output = output_activate * (1 - output_activate)  # (1, output_size)
        output_to_who = hidden_activate  # (hidden_size, 1)

        gradient_who = (loss_to_output_activate * output_activate_to_output).reshape(-1, 1) @ output_to_who.reshape(1,
                                                                                                                    -1)
        gradient_who = gradient_who.T  # (hidden_size, output_size)

        output_to_hidden_activate = self.who  # (hidden_size, output_size)
        hidden_activate_to_hidden = hidden_activate * (1 - hidden_activate)  # (1, hidden_size)
        hidden_to_wih = input  # (1, input_size)

        gradient1 = hidden_to_wih.reshape(-1, 1) @ hidden_activate_to_hidden.reshape(1, -1)  # (input_size, hidden_size)
        gradient2 = (loss_to_output_activate * output_activate_to_output) * output_to_hidden_activate  # (hidden_size, output_size)
        gradient2 = gradient2.T  # (output_size, hidden_size)
        gradient_wih = np.zeros(gradient1.shape)
        for index in range(gradient2.shape[0]):
            gradient_wih += gradient1 * gradient2[index]

        self.wih = self.wih - gradient_wih * self.lr
        self.who = self.who - gradient_who * self.lr

        return loss

    def query(self, input, y_true=None):
        hidden = np.dot(input, self.wih)
        hidden_activate = self.activation(hidden)
        output = np.dot(hidden_activate, self.who)
        output_activate = self.activation(output)
        print("预测概率矩阵")
        print(output_activate)
        if y_true is not None:
            if y_true.ndim == 1:
                new_y_true = np.zeros((y_true.shape[0], 10))
                for index, t in enumerate(list(y_true)):
                    new_y_true[index][int(t)] = 1
                y_true = new_y_true
            print("预测时候的loss")
            print(self.cross_entropy(output_activate, y_true))
        return output_activate


train_data_file = open("dataset/mnist_train.csv", "r")
train_data_list = train_data_file.readlines()
train_data_file.close()

train_data = np.zeros((len(train_data_list), 28 * 28))
train_label = np.zeros((len(train_data_list), 10))
for i, e in enumerate(train_data_list):
    data = e.split(',')
    train_data[i] = np.array(data[1:])
    train_label[i][int(data[0])] = 1
train_data = train_data / 255.0 * 0.99 + 0.01

input_size = train_data.shape[1]
hidden_size = 200
output_size = 10
epoch = 40

model = DIY_Model(input_size, hidden_size, output_size, lr=0.001)
avg_loss = []
all_loss = []
for e in range(epoch):
    print("epoch: ", e)
    losses = []
    for i in range(train_data.shape[0]):
        loss = model.train(train_data[i], train_label[i])
        losses.append(loss)
        all_loss.append(loss)
    avg_loss.append(np.mean(losses))
    print()

plt.subplot(121)
plt.plot([i for i in range(epoch)], avg_loss)
plt.title("avg_loss")
plt.subplot(122)
plt.plot([i for i in range(len(all_loss))], all_loss)
plt.title("all_loss")

plt.show()

# 测试如下
test_data_file = open("dataset/mnist_test.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

test_data = np.zeros((len(test_data_list), 28 * 28))
test_label = np.zeros(len(test_data_list))
for i, e in enumerate(test_data_list):
    data = e.split(',')
    test_data[i] = np.array(data[1:])
    test_label[i] = int(data[0])
test_data = test_data / 255.0 * 0.99 + 0.01

predict_result = model.query(test_data, test_label)
correct_num = 0
for i in range(len(predict_result)):
    predict = int(np.argmax(predict_result[i]))
    true_label = test_label[i]
    if predict == true_label:
        correct_num += 1
print("一共预测%d个，正确%d个，正确率%.2f" % (len(predict_result), correct_num, correct_num / len(predict_result)))
