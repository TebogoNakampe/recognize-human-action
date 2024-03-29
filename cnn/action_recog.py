#hello rha
import tensorflow as tf
import numpy as np
import random

def run(data, batch_size, epoch):
    train_x = data[0]
    train_y = data[1]
    test_x = data[2]
    test_y = data[3]
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)

    train_len = len(train_x)
    num_out = len(train_y[0])

    train_y = train_y.reshape(-1, num_out)
    test_y = test_y.reshape(-1, num_out)
#    X -> fix
#    Y -> can be changed
    X = tf.placeholder(tf.float32, [None, 60, 80, 3])
    Y = tf.placeholder(tf.float32, [None, num_out])
    keep_prob = tf.placeholder(tf.float32)

#    no padding -> edge == trash data
    W1 = tf.Variable(tf.random_normal([3, 3, 3, 6], stddev=0.01))
    C1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(C1)
    P1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    D1 = tf.nn.dropout(P1, keep_prob=keep_prob)

#    print(P1)
#    P1 -> (?, 30, 40, 6)


    W2 = tf.Variable(tf.random_normal([3, 3, 6, 12], stddev=0.01))
    C2 = tf.nn.conv2d(D1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(C2)
    P2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    D2 = tf.nn.dropout(P2, keep_prob=keep_prob)

#    print(P2)
#    P2 -> (?, 15, 20, 12)


    #W3 = tf.Variable(tf.random_normal([30*40*12, 30*40*12], stddev=0.01))
    W3 = tf.get_variable("W3", shape=[15*20*12, 15*20*12], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([15*20*12]))
    L3 = tf.reshape(D2, [-1, 15*20*12])
    L3 = tf.nn.relu(tf.matmul(L3, W3)+b3)
    D3 = tf.nn.dropout(L3, keep_prob=keep_prob)


    #W4 = tf.Variable(tf.random_normal([30*40*12, num_out], stddev=0.01))
    W4 = tf.get_variable("W4", shape=[15*20*12, num_out], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([num_out]))
    model = tf.matmul(D3, W4) + b4
  

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        total_batch = int(train_len/batch_size)
        if(total_batch == 0):
            total_batch = 1

        for e in range(epoch):
            total_cost = 0

            j = 0
            for i in range(total_batch):
                if(j+batch_size > len(train_x)):
                    batch_x = train_x[j:]
                    batch_y = train_y[j:]
                else:
                    batch_x = train_x[j:j+batch_size]
                    batch_y = train_y[j:j+batch_size]
                    j = j+batch_size

                batch_y = batch_y.reshape(-1, num_out)
    #            print(type(batch_x))
    #            print(batch_x.shape)
    #            print(type(batch_y))
    #            print(batch_y.shape)
                _, cost_val = sess.run([optimizer, cost], feed_dict={X:batch_x, Y:batch_y, keep_prob:1})

                total_cost = total_cost + cost_val

            print('Epoch:', '%d' % (e + 1), 'Average cost =', '{:.3f}'.format(total_cost / total_batch))

        print("complete")

        is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        #print('accuracy: ', sess.run(accuracy, feed_dict={X: test_x, Y: test_y}))
        print("\naccuracy: ", "{:.2f}".format(accuracy.eval(feed_dict={X:test_x, Y:test_y, keep_prob:1})*100), "%")

        print("Label: ", sess.run(tf.argmax(test_y, 1)))
        print("Prediction: ", sess.run(tf.argmax(model, 1), feed_dict={X: test_x, keep_prob:1}))
