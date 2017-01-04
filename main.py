#! -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from Model import Model
from make_fig import get_batch    

if __name__ == u'__main__':

    file_name = 'mnist_test.csv'
    #file_name = 'mnist_mini.csv'
    dump_dir = 'sample_result'
    if os.path.exists(dump_dir) == False:
        os.mkdir(dump_dir)
        
    # parameter
    batch_size = 10
    z_dim = 20
    epoch_num = 10
    
    # make model
    model = Model(z_dim, batch_size)
    model.set_model()
    
    num_one_epoch = sum(1 for _ in open(file_name)) //batch_size
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(epoch_num):
            loss = 0
            with open(file_name, 'r') as f_obj:
                for step in range(num_one_epoch):
                    batch_labels, batch_figs = get_batch(f_obj, batch_size)
                    model.training(sess, batch_figs)
                    
                    if step%1000 == 0:
                        print(epoch)
                        tmp = model.generate(sess, [[0] * z_dim])
                        for l in tmp:
                            print(l)
                        print("---------")

        print("-- end train--");sys.stdout.flush()
        z1 = model.encode(sess, [batch_figs[1]])[0]
        z2 = model.encode(sess, [batch_figs[8]])[0]
        diff = [z1[i] - z2[i] for i in range(z_dim)]
        
        for i in range(20):
            z = [z2[_] + diff[_] * i * 0.05 for _ in range(z_dim)]
            tmp = model.generate(sess, [z])
            plt.imshow(tmp, cmap = plt.cm.gray)
            #plt.show()
            plt.savefig(os.path.join(dump_dir, "fig{}.png".format(i)))
