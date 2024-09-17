import numpy as np
import matplotlib.pyplot as plt
import scipy.special



class NeuraNetWork:
    def __init__(self,inputnodes,hiddenodes,outputnodes,learate):
        '''
        初始化网络，设置输入层，隐藏层，输出层节点数
        :param inputnodes:
        :param hiddenodes:
        :param outputnodes:
        :param learate:
        '''
        self.innode = inputnodes
        self.hnode = hiddenodes
        self.onode = outputnodes

        #设置学习率
        self.lr = learate

        '''随机初始化权重矩阵'''
        self.wih = np.random.rand(self.hnode,self.innode) - 0.5
        self.woh = np.random.rand(self.onode,self.hnode) - 0.5

        '''设置激活函数  此处对应设置为sigmod激活函数'''
        self.activate_fun = lambda x:scipy.special.expit(x)
        pass

    def train(self,input_list,target_list):

        inputs = np.array(input_list,ndmin=2).T
        targets =np.array(target_list,ndmin=2).T

        '''根据输入和权重矩阵，计算隐藏层输入'''
        hidden_inputs = np.dot(self.wih,inputs)
        '''根据激活函数，计算隐藏层输出'''
        hidden_outputs = self.activate_fun(hidden_inputs)
        '''根据输出层权重，计算输出层输入'''
        final_inputs = np.dot(self.woh,hidden_outputs)
        '''根据激活函数，计算输出层输出'''
        final_outputs = self.activate_fun(final_inputs)

        '''计算输出层和隐藏层误差
        输出层误差即所见
        隐藏层误差中，final_outputs*(1-final_outputs)为激活函数的导数'''
        output_errs = targets - final_outputs
        hidden_errs = np.dot(self.woh.T,output_errs*final_outputs*(1-final_outputs))
        '''反向更新权重参数'''
        self.woh += self.lr * np.dot((output_errs*final_outputs*(1-final_outputs)),np.transpose(hidden_outputs))

        self.wih += self.lr * np.dot((hidden_errs*hidden_outputs*(1-hidden_outputs)),np.transpose(inputs))
        pass

    def query(self,inputs):
        '''输入数据计算得到中间层的数据'''
        hidden_inputs = np.dot(self.wih,inputs)
        '''将中间层数据送入激活函数'''
        hidden_output = self.activate_fun(hidden_inputs)
        '''将激活函数处理后的数据和输出层权重计算'''
        final_intput = np.dot(self.woh,hidden_output)
        '''将输出层得到的数据送入激活函数'''
        final_output = self.activate_fun(final_intput)
        '''返回最外层神经元经过激活函数处理后的信号量'''
        print(final_output)
        return final_output
        pass


'''初始化网络'''
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
network = NeuraNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

'''导入训练数据'''
training_data_file = open("/Users/aragaki/artificial/pythonProject/week09/dataset/mnist_train.csv")

training_data_list = training_data_file.readlines()
training_data_file.close()
'''设定循环次数'''
epoch = 100

for i in range(epoch):
    for record in training_data_list:
        '''数据依靠‘，’来分割'''
        all_values = record.split(',')
        '''对输入数据进行归一化'''
        inputs = (np.asfarray(all_values[1:])) / 255 * 0.99 + 0.01
        '''设置标签对应的独热编码'''
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        '''使用模型对此进行训练'''
        network.train(inputs, targets)

test_data_file = open("/Users/aragaki/artificial/pythonProject/week09/dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    '''数据中第一位为标签，即为对应的数字'''
    correct_number = int(all_values[0])
    print("该图片对应的数字为：", correct_number)
    '''对输入数据进行归一化'''
    inputs = np.asfarray(all_values[1:]) / 255 * 0.99 + 0.01
    outputs = network.query(inputs)
    '''处理输出，对输出结果进行查询，最大的下标即为网络推理结果'''
    label = np.argmax(outputs)
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)

scores_arr = np.asarray(scores)
print("performance  = ", scores_arr.sum() / scores_arr.size)
