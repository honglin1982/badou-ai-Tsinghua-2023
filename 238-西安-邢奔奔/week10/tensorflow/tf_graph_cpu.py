import tensorflow as tf

'''手动指派设备可以使用with tf.device来创建一个设备环境，这个环境下的opertatin都统一运行在环境制定的设备上'''
with tf.device('/cpu:0'):
    a = tf.constant([1,2,3,4,5,6],shape=[2,3],name='a')
    b = tf.constant([1,2,3,4,5,6],shape=[3,2],name='b')
    c = tf.matmul(a,b,name='mul')
    print("tensor:",c)
# with tf.Session() as sess:
#     resu = sess.run(c)
#     print(resu)
'''可以在创建的session中把参数allow_soft_placement设置为True，这样tensorflow会自动选择一个存在并且支持的设备来运行op'''
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
print(sess.run(c))
sess.close()