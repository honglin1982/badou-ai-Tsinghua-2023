import numpy as np
import scipy as sp
import scipy.linalg as sl
import pylab

def ransac(data,model,n,k,t,d,debug=False,return_all=False):
    """
    输入:
        data - 样本点
        model - 假设模型:事先自己确定
        n - 生成模型所需的最少样本点
        k - 最大迭代次数
        t - 阈值:作为判断点满足模型的条件
        d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
    输出:
        bestfit - 最优拟合解（返回nil,如果未找到）
    
    iterations = 0 #迭代次数
    bestfit = nil #后面更新
    besterr = something really large #后期更新besterr = thiserr
    while iterations < k 
    {
        maybeinliers = 从样本中随机选取n个,不一定全是局内点,甚至全部为局外点
        maybemodel = n个maybeinliers 拟合出来的可能符合要求的模型
        alsoinliers = emptyset #满足误差要求的样本点,开始置空
        for (每一个不是maybeinliers的样本点)
        {
            if 满足maybemodel即error < t
                将点加入alsoinliers 
        }
        if (alsoinliers样本点数目 > d) 
        {
            %有了较好的模型,测试模型符合度
            bettermodel = 利用所有的maybeinliers 和 alsoinliers 重新生成更好的模型
            thiserr = 所有的maybeinliers 和 alsoinliers 样本点的误差度量
            if thiserr < besterr
            {
                bestfit = bettermodel
                besterr = thiserr
            }
        }
        iterations++
    }
    return bestfit
    """
    iteration = 0 #迭代次数初始化
    bestfit = None
    besterr = np.inf #设置默认值
    best_inlier_idxs = None
    while iteration < k:
        print('迭代次数：',iteration)
        maybe_idxs,test_idxs = random_partition(n,data.shape[0])
        print('test_idxs',test_idxs)
        maybe_inliers = data[maybe_idxs,:] #获取size(maybe_idxs)行数据(Xi,Yi)
        test_points = data[test_idxs]      #若干行(Xi,Yi)数据点
        maybemodel = model.fit(maybe_inliers) #拟合内群模型
        test_err = model.get_err(test_points,maybemodel) #计算误差:平方和最小
        print('test_err\n',test_err<t)
        also_idxs = test_idxs[test_err<t]
        also_inliers = data[also_idxs,:]

        if debug:
            print('test_err.min()',test_err.min())
            print('test_err.max()',test_err.max())
            print('test_err.mean()',np.mean(test_err))
            print('iteration:%dlen(also_inliers)%d'%(iteration,len(also_inliers)))

        if len(also_inliers) >d :
            betterdata = np.concatenate((maybe_inliers,also_inliers)) #内群和符合内群模型的外群集合
            bestmodel = model.fit(betterdata)  
            better_errs = model.get_err(betterdata,bestmodel)
            this_err = np.mean(better_errs) #平均误差作为新的误差
            if this_err < besterr:
                bestfit = bestmodel
                besterr = this_err
                best_inliers_idxs = np.concatenate((maybe_idxs,also_idxs)) #更新局内点,将新点加入
        iteration += 1
    if bestfit == None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit,{'inliers':best_inliers_idxs}
    else:
        return bestfit

def random_partition(n,n_data):
    """return n random rows of data and the other len(data) - n rows"""
    all_idxs = np.arange(n_data) #获取数据所有索引
    np.random.shuffle(all_idxs)  #打乱索引顺序
    idxs1 = all_idxs[:n]         
    idxs2 = all_idxs[n:]
    return idxs1,idxs2



class LinearLeastSquareModel():
    #类的初始化
    def __init__(self,input_columns,output_columns,debug=False) :
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug
    def fit(self,data):
        #np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = np.vstack([data[:,i] for i in self.input_columns]).T #第一列Xi-->行Xi
        B = np.vstack([data[:,i] for i in self.output_columns]).T #第二列Yi-->行Yi
        x,resids,rank,s = sl.lstsq(A,B) #residues:残差和
        return x
    def get_err(self,data,model):
        A = np.vstack([data[:,i] for i in self.input_columns]).T #第一列Xi-->行Xi
        B = np.vstack([data[:,i] for i in self.output_columns]).T #第二列Yi-->行Yi
        B_fit = sp.dot(A,model)  #计算的外群y值,B_fit = model.k*A + model.b
        err_per_points = np.sum((B-B_fit)**2,axis=1) #sum squared error per row 
        return err_per_points


def test():
    #样本数500
    n_samples = 500
    #输入变量个数1
    n_inputs = 1
    #输出变量个数1
    n_outputs = 1

    #生成n_samples行，n_inputs列的0-20之间的数据
    A_exact = 20*np.random.random((n_samples,n_inputs))
    #创建一个斜率，随机线性度
    perfect_fit = 60*np.random.normal(size=(n_inputs,n_outputs))
    #y=k*x 内群
    B_exact = np.dot(A_exact,perfect_fit)
    # pylab.plot(A_exact,'k.')
    # pylab.plot(perfect_fit,'k.')
    # pylab.plot(B_exact,'k.')
    # pylab.show()

    

    #加入高斯噪声
    A_noisy = A_exact + np.random.normal(size=A_exact.shape) #500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape) #500 * 1行向量,代表Yi
    if 1:
        #添加局外点：
        n_outliters = 100
        all_idxs = np.arange(A_exact.shape[0]) #获取索引0-499
        np.random.shuffle(all_idxs) #打乱索引顺序
        outlier_idxs = all_idxs[:n_outliters] #0-500范围内100个随机局外点
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliters,n_inputs))#加入噪声和局外点的Xi
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliters,n_outputs))#加入噪声和局外点的Xi
    #set model
    all_data = np.hstack((A_noisy,B_noisy)) #500行2列的([Xi,Yi])
    print(all_data)
    input_columns = range(n_inputs) #数组的第一列x:0
    output_columns = [n_inputs+i for i in range(n_outputs)] #数组最后一列y:1
    debug = False
    print('input_columns:\n',input_columns)
    print('output_columns:\n',output_columns)
    model = LinearLeastSquareModel(input_columns,output_columns,debug=debug) #类的实例化:用最小二乘生成已知模型,初始化
    print('all_data[:,input_columns]\n',all_data[:,input_columns])
    print('all_data[:,output_columns]\n',all_data[:,output_columns])

    #最小二乘法
    linear_fit,resids,rank,s = sp.linalg.lstsq(all_data[:,input_columns],all_data[:,output_columns])
    #ransac 算法
    ransac_fit, ransac_data = ransac(all_data,model,50, 1000, 7e3, 300,debug=debug,return_all=True)

    if 1:
        import pylab
 
        sort_idxs = np.argsort(A_exact[:,0])
        A_col0_sorted = A_exact[sort_idxs] #秩为2的数组
 
        if 1:
            pylab.plot( A_noisy[:,0], B_noisy[:,0], 'k.', label = 'data' ) #散点图
            pylab.plot( A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label = "RANSAC data" )
        else:
            pylab.plot( A_noisy[non_outlier_idxs,0], B_noisy[non_outlier_idxs,0], 'k.', label='noisy data' )
            pylab.plot( A_noisy[outlier_idxs,0], B_noisy[outlier_idxs,0], 'r.', label='outlier data' )
 
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,ransac_fit)[:,0],
                    label='RANSAC fit' )
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,perfect_fit)[:,0],
                    label='exact system' )
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,linear_fit)[:,0],
                    label='linear fit' )
        pylab.legend()
        pylab.show()
 





    



if __name__ == '__main__':
    test()