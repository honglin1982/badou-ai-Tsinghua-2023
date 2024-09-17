import torch
import torch.optim as optim
import torch.nn.functional as func
import torchvision
import torchvision.transforms as transforms

# 定义一个可以实现手写数字识别的简单神经网络生成类 返回一个基于pytorch的神经网络模型
class mnist_network(torch.nn.Module):
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super(mnist_network, self).__init__()
        self.ins = input_nodes
        self.input_to_hidden = torch.nn.Linear(input_nodes, hidden_nodes)
        self.hidden_to_hidden = torch.nn.Linear(hidden_nodes, hidden_nodes)
        self.hidden_to_output = torch.nn.Linear(hidden_nodes, output_nodes)

    def forward(self, x):
        x = x.view(-1, self.ins)
        x = func.relu(self.input_to_hidden(x))
        x = func.relu(self.hidden_to_hidden(x))
        x = func.softmax(self.hidden_to_output(x), dim=1)
        return x

class mnist_model:
    def __init__(self, network, learning_rates):
        self.net = network
        self.cost = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=learning_rates)

    def train(self, train_loader, epoch=5):
        for _ in range(epoch):
            running_loss = 0.0
            for i, cur_data in enumerate(train_loader, 0):
                cur_input, cur_label = cur_data
                self.optimizer.zero_grad()
                cur_predict = self.net(cur_input)
                loss = self.cost(cur_predict, cur_label)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if _ % 100 == 0:
                    print('[epoch %d], %.2f %% loss: %.3f' %
                          (_ + 1, (i + 1) * 1. / len(train_loader) * 100, running_loss / 100))
                    running_loss = 0.0

    def query(self, test_loader):
        correct, total = 0, 0
        with torch.no_grad():
            for cur_data in test_loader:
                cur_img, cur_label = cur_data
                predict_label = torch.argmax(self.net(cur_img), 1)
                total += cur_label.size(0)
                correct += (predict_label == cur_label).sum().item()
                print('网络推测结果：', predict_label)
                print('实际结果：', cur_label)
        print('模型推理准确率：%.2f %%' % (100 * correct / total))

def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0, ], [1, ])])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True, num_workers=2)
    return train_loader, test_loader

def main():
    train_loader, test_loader = mnist_load_data()

    network = mnist_network(input_nodes=784, hidden_nodes=512, output_nodes=10)
    model = mnist_model(network, learning_rates=0.001)
    model.train(train_loader=train_loader)
    model.query(test_loader)

if __name__ == '__main__':
    main()
