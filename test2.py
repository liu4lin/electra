import numpy as np
import tensorflow as tf
import random

Vs = 1000
Bs = 3
Lmax = 20
Ns = 4 #number of positions to interrupt

lens_np = np.random.uniform(Lmax/2, Lmax, Bs).astype(int)
inputs_list = []
for i in range(Bs):
    lens_np[i] = lens_np[i]
    one = tf.constant(np.concatenate((np.random.uniform(0, Vs, lens_np[i]),
                          [0] * (Lmax - lens_np[i]))).astype(np.int32))
    #one = tf.concat([tf.random.uniform([lens_np[i]], 0, Vs, tf.int32),
    #                 tf.constant([0] * (Lmax - lens_np[i]))], 0)
    inputs_list.append(tf.expand_dims(one, 0))

inputs_tf = tf.concat(inputs_list, 0)

mask = 1 - tf.cast(tf.equal(inputs_tf, 0), tf.int32)
lens_tf = tf.reduce_sum(mask, 1)

labels_list = []
for i in range(Bs):
    positions = tf.range(Lmax)
    substitues = tf.random.uniform([1], 1, lens_tf[i] - 1, tf.int32)
    labels = tf.cast(tf.equal(positions, substitues), tf.int32)
    labels_list.append(tf.expand_dims(labels,0))

labels_tf = tf.concat(labels_list, 0)

splits_list = []
for i in range(Bs):
    #one = tf.constant(np.random.uniform(0, lens_np[i], (Ns * 2)).astype(int))
    one = tf.random.uniform([Ns * 4], 1, lens_tf[i], tf.int32)
    one, _ = tf.unique(one)
    one = tf.cond(tf.less(tf.shape(one)[0], Ns * 2),
                 lambda: tf.expand_dims(tf.range(Ns * 2)[1::2], 0),
                 lambda: tf.sort(tf.reshape(one[: Ns * 2], [1, Ns * 2]))[:, ::2])
    splits_list.append(one)

splits_tf = tf.concat(splits_list, 0)

splits_up = tf.concat([splits_tf, tf.expand_dims(tf.constant([Lmax] * Bs, tf.int32), 1)], 1)
splits_lo = tf.concat([tf.expand_dims(tf.constant([0] * Bs, tf.int32), 1), splits_tf], 1)
size_splits = splits_up - splits_lo


new_labels_list = []
new_inputs_list = []
for i in range(Bs):
    inputs_splits = tf.split(inputs_tf[i, :], size_splits[i, :])
    labels_splits = tf.split(labels_tf[i, :], size_splits[i, :])
    one_inputs = []
    one_labels = []
    size_split = len(inputs_splits)
    rand_check = 1 #random.randint(0,1)
    for j in range(size_split):
        inputs = inputs_splits[j]
        labels = labels_splits[j] #label 1 for substistution
        if j < size_split -1: #exclude the last split
            if j % 2 == rand_check: #label 2 for insertion
                labels = tf.concat([labels, tf.constant([2])], 0)
                inputs = tf.concat([inputs, tf.constant([10000])], 0)
            else: #label 3 for deletion
                labels = tf.concat([labels[:-2], tf.constant([3])], 0)
                inputs = inputs[:-1]
        one_labels.append(labels)
        one_inputs.append(inputs)
    one_inputs_tf = tf.concat(one_inputs, 0)
    one_labels_tf = tf.concat(one_labels, 0)
    one_inputs_tf = tf.cond(tf.less(lens_tf[i], Ns * 2), lambda: inputs_tf[i, :], lambda: one_inputs_tf)
    one_labels_tf = tf.cond(tf.less(lens_tf[i], Ns * 2), lambda: labels_tf[i, :], lambda: one_labels_tf)
    new_inputs_list.append(tf.expand_dims(one_inputs_tf, 0))
    new_labels_list.append(tf.expand_dims(one_labels_tf, 0))

sess = tf.Session()
print(sess.run(new_inputs_list))

print(sess.run(new_labels_list))

print('ending')
