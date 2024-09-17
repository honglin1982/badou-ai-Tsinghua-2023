from sklearn.decomposition import PCA
import numpy as np

# PCA-接口
X = np.array([[10, 15, 29], [15, 46, 13], [23, 21, 30], [11, 9, 35], [42, 45, 11],
              [9, 48, 5], [11, 21, 14], [8, 5, 15], [11, 12, 21], [21, 20, 25]])  # 导入数据，维度为3
pca = PCA(n_components=2)  # 降到2维
pca.fit(X)  # 训练
newX = pca.fit_transform(X)  # 降维后的数据
print("原始数据(10行3列，10个样例，每个样例3个特征)：\n", X)
print("各维度输出贡献率 ：")
print(pca.explained_variance_ratio_)  # 输出贡献率
print("PCA降维后的数据 ：")
print(newX)  # 输出降维后的数据
print("\n")


# -------------------------------------------------------------------------------------------------
# PCA-自实现
class My_PCA(object):
    def __init__(self, X, K):
        self.X = X  # 样本矩阵X
        self.K = K  # K阶降维矩阵的K值
        self.centrX = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()  # Z=XU求得

    def _centralized(self):
        # 矩阵X的中心化，去均值
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 样本集的特征均值
        print('样本集的特征均值:\n', mean)
        centrX = self.X - mean  ##样本集的中心化
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX

    def _cov(self):
        ns = np.shape(self.centrX)[0]  # 矩阵横坐标
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def _U(self):
        a, b = np.linalg.eig(self.C)
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 给出特征值降序的topK的索引序列
        ind = np.argsort(-1 * a)
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    def _Z(self):
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__ == '__main__':
    K = np.shape(X)[1] - 1  # 降到几维
    pca = My_PCA(X, K)


