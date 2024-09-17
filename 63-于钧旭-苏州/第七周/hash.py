import cv2
import glob

#均值哈希算法
def aHash(img):
    img = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str=''
    avg=gray.mean()
    for i in range(8):
        for j in range(8):
            hash_str += '1' if gray[i,j]>avg else '0'
    return hash_str
#差值哈希算法
def dHash(img):
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str=''
    #每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if   gray[i,j]>gray[i,j+1]:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str

#Hash值对比
def cmpHash(hash1,hash2):
    n=0
    if len(hash1)!=len(hash2):
        return -1
    for i in range(len(hash1)):
        #不相等则n计数+1，n最终为相似度
        if hash1[i]!=hash2[i]:
            n=n+1
    return n

def main():
    img0 = cv2.imread('img/lenna.png')
    hash1 = aHash(img0)
    hash2 = aHash(img0)
    print('img/lenna.png',hash1,hash2,sep='\t')
    img_list = glob.glob('img/lenna*.jpg')
    for img_path in img_list:
        img = cv2.imread(img_path)
        hash_1=aHash(img)
        hash_2=dHash(img)
        n_1 = cmpHash(hash1, hash_1)
        n_2 = cmpHash(hash2, hash_2)
        print(img_path,hash_1,hash_2,n_1,n_2,sep='\t')

if __name__ == '__main__':
    main()