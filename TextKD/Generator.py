import os, sys
sys.path.append(os.getcwd())

import time

import numpy as np
import tensorflow as tf

import language_helpers
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import tflib.plot
from keras.utils import to_categorical

import copy
import pickle



BATCH_SIZE = 64
SEQ_LEN = 33
DIM = 256
LAMBDA = 10
############

try:
    with open("../dictionary.pickle", 'rb') as f:
        charmap = pickle.load(f)

    with open("../inverted_dictionary.pickle", 'rb') as f:
        inv_charmap = pickle.load(f)
except Exception as e:
    print(e)
    exit()

#####################################################################

def softmax(logits):
    return tf.reshape(
        tf.nn.softmax(
            tf.reshape(logits, [-1, len(charmap)])
        ),
        tf.shape(logits)
    )

def make_noise(shape):
    return tf.random_normal(shape)

def ResBlock(name, inputs):
    output = inputs
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.1', DIM, DIM, 5, output)
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.2', DIM, DIM, 5, output)
    return inputs + (0.3*output)

def Generator(n_samples, prev_outputs=None):
    output = make_noise(shape=[n_samples, 128])
    output = lib.ops.linear.Linear('Generator.Input', 128, SEQ_LEN*DIM, output)
    output = tf.reshape(output, [-1, DIM, SEQ_LEN])
    output = ResBlock('Generator.1', output)
    output = ResBlock('Generator.2', output)
    output = ResBlock('Generator.3', output)
    output = ResBlock('Generator.4', output)
    output = ResBlock('Generator.5', output)
    output = lib.ops.conv1d.Conv1D('Generator.Output', DIM, len(charmap), 1, output)
    output = tf.transpose(output, [0, 2, 1])
    output = softmax(output)
    return output

fake_inputs = Generator(BATCH_SIZE)
fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)



###########################



saver = tf.train.Saver() 
with tf.Session() as session:


    ckpt = tf.train.get_checkpoint_state("./model")
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        session.run(tf.initialize_all_variables())
        print("No model found")


    def generate_samples():
        samples,fake_inputs_discretez = session.run([fake_inputs,fake_inputs_discrete])
        samples = np.argmax(samples, axis=2)
        decoded_samples = []
        for i in range(len(samples)):
            decoded = []
            for j in range(len(samples[i])):
                decoded.append(inv_charmap[samples[i][j]])
            decoded_samples.append(tuple(decoded))
        return decoded_samples


    samples = []
    for i in range(10):
        sample=generate_samples()
        #print(sample)
        samples.extend(sample)



    with open("TESTING.txt", 'w') as f:
        for s in samples:
            s = "".join(s)
            f.write(s + "\n")