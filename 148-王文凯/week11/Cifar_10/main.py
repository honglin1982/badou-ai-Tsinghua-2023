import cifar_data_get
import network_model
import numpy as np
import math
import time
import tensorflow as tf

data_dir = "cifar_data/cifar-10-batches-bin"
batch_size = 100
num_examples_for_eval = 10000

def main():
    data_train, label_train = cifar_data_get.inputs(data_dir=data_dir, batch_size=batch_size, distorted=True)
    data_test, label_test = cifar_data_get.inputs(data_dir=data_dir, batch_size=batch_size)
    model = network_model.ConvolutionNeuralNetworkForCifar(height=24, width=24, channels=3, batch_size=batch_size)
    with tf.Session() as sess:
        sess.run(model.init_op)
        tf.train.start_queue_runners()

        for step in range(5001):
            start_time = time.time()
            image_batch, label_batch = sess.run([data_train, label_train])
            _, loss_value = sess.run([model.train_op, model.loss], feed_dict={
                model.x: image_batch, model.y_: label_batch})
            duration = time.time() - start_time

            if step % 100 == 0:
                examples_per_sec = batch_size / duration
                sec_per_batch = float(duration)
                print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)" % (
                    step, loss_value, examples_per_sec, sec_per_batch))

        num_batch = int(math.ceil(num_examples_for_eval / batch_size))  # math.ceil()函数用于求整
        true_count = 0
        total_sample_count = num_batch * batch_size

        # 在一个for循环里面统计所有预测正确的样例个数
        for _ in range(num_batch):
            image_batch, label_batch = sess.run([data_test, label_test])
            predictions = sess.run([model.top_k_op], feed_dict={model.x: image_batch, model.y_: label_batch})
            true_count += np.sum(predictions)

        # 打印正确率信息
        print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))


if __name__ == "__main__":
    main()
