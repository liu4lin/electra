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
    one = tf.cond(tf.less(tf.shape(one)[0], Ns * 2 + 1),
                 lambda: tf.expand_dims(tf.range(1, Ns * 2 + 2), 0),
                 lambda: tf.sort(tf.reshape(one[: Ns * 2 + 1], [1, Ns * 2 + 1])))
    splits_list.append(one[:, 2::2])

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
    rand_check = random.randint(0,1)
    inputs_end = inputs_splits[-1]
    labels_end = labels_splits[-1]
    for j in range(size_split-1):
        inputs = inputs_splits[j]
        labels = labels_splits[j] #label 1 for substistution
        rand_op = random.randint(0, 2)
        if rand_op == 0: #label 2 for insertion
            labels_hold = tf.concat([labels, tf.constant([2])], 0)
            #if bilm is None: #noise
            #insert_tok = tf.random.uniform([1], 0, Vs, tf.int32)
            #else: #2-gram prediction
            #    insert_tok = tf.expand_dims(bilm[inputs[-1], random.randint(0, nlms-1)], 0)
            insert_tok = tf.constant([9999], tf.int32)
            inputs_hold = tf.concat([inputs, insert_tok], 0)
            inputs = tf.cond(tf.less_equal(2, tf.shape(inputs_end)[0]), lambda: inputs_hold, lambda: inputs)
            labels = tf.cond(tf.less_equal(2, tf.shape(inputs_end)[0]), lambda: labels_hold, lambda: labels)
            inputs_end = tf.cond(tf.less_equal(2, tf.shape(inputs_end)[0]), lambda: inputs_end[:-1],
                                 lambda: inputs_end)
            labels_end = tf.cond(tf.less_equal(2, tf.shape(inputs_end)[0]), lambda: labels_end[:-1],
                                 lambda: labels_end)
        elif rand_op == 1: #label 3 for deletion
            labels = tf.concat([labels[:-2], tf.constant([3])], 0)
            inputs = inputs[:-1]
            inputs_end = tf.concat([inputs_end, tf.constant([0])], 0)
            labels_end = tf.concat([labels_end, tf.constant([0])], 0)
        else: #label 4 for swap
            labels = tf.concat([labels[:-1], tf.constant([4])], 0)
            inputs = tf.concat([inputs[:-2], [inputs[-1]], [inputs[-2]]], 0)
        one_labels.append(labels)
        one_inputs.append(inputs)
    one_inputs.append(inputs_end)
    one_labels.append(labels_end)
    one_inputs_tf = tf.concat(one_inputs, 0)
    one_labels_tf = tf.concat(one_labels, 0)
    one_inputs_tf = tf.cond(tf.less(lens_tf[i], Ns * 2 + 1), lambda: inputs_tf[i, :], lambda: one_inputs_tf)
    one_labels_tf = tf.cond(tf.less(lens_tf[i], Ns * 2 + 1), lambda: labels_tf[i, :], lambda: one_labels_tf)
    new_inputs_list.append(tf.expand_dims(one_inputs_tf, 0))
    new_labels_list.append(tf.expand_dims(one_labels_tf, 0))

new_inputs_tf = tf.concat(new_inputs_list, 0)
new_labels_tf = tf.concat(new_labels_list, 0)

new_input_mask = tf.cast(tf.not_equal(new_inputs_tf, 0), tf.int32)

sess = tf.Session()
print(sess.run(splits_tf))
a, b, c, d, e = sess.run([inputs_tf, new_inputs_tf, new_labels_tf, mask, new_input_mask])
print(a)
print(b)
print(c)
print(d)
print(e)
print('ending')
