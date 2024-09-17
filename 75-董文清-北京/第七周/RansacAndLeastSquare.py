import numpy as np
import pylab
import scipy as sp
import scipy.linalg as sl

class LinearLeastSquareModel:

    def __init__(self, inputColumns, outputColumns, debug=False):
        self.inputColumns = inputColumns
        self.outputColumns = outputColumns
        self.debug = debug
    def fit(self, data):

        #按行谁徐堆叠数据
        X = np.vstack([data[:,i] for i in self.inputColumns]).T
        Y = np.vstack([data[:,i] for i in self.outputColumns]).T
        vecter, resids, rank, s = sl.lstsq(X, Y) #残差和
        return vecter
    def getError(self, data, model):

        X = np.vstack([data[:,i] for i in self.inputColumns]).T
        Y = np.vstack([data[:,i] for i in self.outputColumns]).T
        Yfit = sp.dot(X, model) #计算Y值 Yfit = k * X + model.b
        errPerRow = np.sum((Y - Yfit) ** 2, axis=1) #按行求方差
        return errPerRow

def RanSac(data, model, n, k, t, d, debug=False, returnAll=False):
    #输入  data - 样本点 model - 假设模型  n - 生成模型所需的最少样本点 k - 最大迭代次数  t - 阈值:作为判断点满足模型的条件

    loops = 0
    bestFit = None
    minErr = np.inf #默认值先设置为正无穷大
    bestInlineIdxs = None
    while loops < k :

        maybeIdxs, remainIdxs = randomPartition(n, data.shape[0])
        inPoints = data[maybeIdxs,:] #获取内群点数据坐标[xi,yi]
        outPoints = data[remainIdxs]
        maybeModel = model.fit(inPoints) #拟合模型
        testErr = model.getError(outPoints, maybeModel) #计算误差：方差最小
        alsoIdxs = remainIdxs[testErr < t]              #按阈值判定其他剩余点是否符合模型
        alsoPoints = data[alsoIdxs,:]
        if debug :
            #debug模式 打印相关参数
            print('testErr.min()', testErr.min())
            print('testErr.max()', testErr.max())
            print('numpy.mean(testErr)', np.mean(testErr.min()))
            print('loops %d:len(alsoPoints) = %d' % (loops, len(alsoPoints)))
        if len(alsoPoints) > d:
            #符合好模型条件

            betterData = np.concatenate((inPoints, alsoPoints)) #合并坐标点数据
            betterModel = model.fit(betterData)
            betterErr = model.getError(betterData, betterModel)
            thisErr = np.mean(betterErr)
            if thisErr < minErr:
                bestFit = betterModel
                minErr = thisErr
                bestInlineIdxs = np.concatenate((maybeIdxs, alsoIdxs))
        loops += 1
    if bestFit is None:
        raise ValueError("Not found fit acceptance criteria")
    if returnAll:
        return bestFit, {'inliners':bestInlineIdxs}
    else:
        return bestFit

def randomPartition(n, data):

    allIdxs = np.arange(data)
    np.random.shuffle(allIdxs)
    return allIdxs[:n], allIdxs[n:]

# 样本数为 500，输入变量个数和输出变量个数都为1
def test(samples=500, input=1, output=1, outLiners = 100):

    Xexact = 20 * np.random.random((samples, input))         #随机生成 500 * 1 的 0 ~ 20 范围内的横坐标
    perfectFit = 60 * np.random.normal(size=(input, output)) #求随机斜率
    Yexact = sp.dot(Xexact, perfectFit) # y = k * x
    #加入高斯噪声的横纵坐标
    Xnoisy = Xexact + np.random.normal(size=Xexact.shape)
    Ynoisy = Yexact + np.random.normal(size=Yexact.shape)
    #添加局外点
    allIdxs = np.arange(Xnoisy.shape[0])
    np.random.shuffle(allIdxs) #随机打乱顺序
    outLinersIdxs = allIdxs[:outLiners]
    Xnoisy[outLinersIdxs] = 20 * np.random.random((outLiners, input))         #加入噪声和局外点的Xi
    Ynoisy[outLinersIdxs] = 50 * np.random.normal(size=(outLiners, output)) #加入噪声和局外点的Yi

    allData = np.hstack((Xnoisy, Ynoisy)) #形式[[Xi, Yi]...] shape:(500,2)
    inputColumns = range(input)
    outputColumns = [input + i for i in range(output)] #数组最后一列 1
    debug = False
    model = LinearLeastSquareModel(inputColumns, outputColumns, debug) #用最小二乘生成已知模型

    linearFit, resids, rank, s = sp.linalg.lstsq(allData[:,inputColumns], allData[:, outputColumns])
    ransacFit, ransacData = RanSac(allData, model, 50, 1000, 7e3, 300, debug, True)

    sortIdxs = np.argsort(Xexact[:,0])
    Xcol0Sorted = Xexact[sortIdxs]
    pylab.plot(Xnoisy[:,0], Ynoisy[:,0], 'k.', label='data') #散点图
    pylab.plot(Xnoisy[ransacData['inliners'], 0], Ynoisy[ransacData['inliners'], 0], 'bx', label='Ransac data')
    pylab.plot(Xcol0Sorted[:, 0], np.dot(Xcol0Sorted, ransacFit)[:,0],  label='Ransac fit')
    pylab.plot(Xcol0Sorted[:, 0], np.dot(Xcol0Sorted, perfectFit)[:,0], label='Exact system')
    pylab.plot(Xcol0Sorted[:, 0], np.dot(Xcol0Sorted, linearFit)[:,0],  label='Linear fit')
    pylab.legend()
    pylab.show()


if __name__ == '__main__':

    test()