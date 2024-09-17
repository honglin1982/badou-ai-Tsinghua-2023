import numpy
# 特殊函数模块
import scipy.special

class NeuralNetwork():
    def __init__(self,inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.outputnodes = outputnodes
        self.lr = learningrate

        self.wih = (numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes)))
        self.who = (numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes)))
        # sigmax函数
        self.activate_function = lambda x:scipy.special.expit(x)

    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activate_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activate_function(final_inputs)

        # 计算误差
        output_error = targets-final_outputs
        hidden_error = numpy.dot(self.who.T,output_error*final_outputs*(1-final_outputs))

        self.who += self.lr * numpy.dot(output_error*final_outputs*(1-final_outputs),
                                        numpy.transpose(hidden_outputs))

        self.wih += self.lr * numpy.dot(hidden_error * hidden_outputs*(1-hidden_outputs),
                                        numpy.transpose(inputs))



input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetwork()
training_data_file = open("dataset/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 5

for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = numpy.asfarray(all_values[1:])/255.0*0.99+0.1
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])]=0.99
        n.train(inputs,targets)

test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores=[]
for record in test_data_list:
    all_values =record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字：",correct_number)

    inputs = (numpy.asfarray(all_values[1:]))/255.0*0.99+0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    print("网络认为图片的数字是：",label)
    if label ==correct_number:
        scores.append(1)
    else:
        scores.append(0)

print(scores)
scores_array = numpy.asarray(scores)
print("perfermance=",scores_array.sum()/scores_array.size)

