
import sys, os, time, shutil, random
import tensorflow as tf
import numpy as np

from model import build_network
from instance_loader import InstanceLoader
from train import run_batch, summarize_epoch

if __name__ == '__main__':



    GNN = build_network(d)

    config = tf.ConfigProto( device_count = {'GPU':0})
    with tf.Session(config=config) as sess:

        print("Initializing global variables ... ", flush=True)
        sess.run( tf.global_variables_initializer() )

        (sess,vars(args)['checkpoint'])

        n_instances = len(loader.filenames)
        stats = { k:np.zeros(n_instances) for k in ['loss','acc','sat','pred','TP','FP','TN','FN'] }

        for (batch_i, batch) in enumerate(loader.get_batches(1, target_cost_dev)):
            stats['loss'][batch_i], stats['acc'][batch_i], stats['sat'][batch_i], stats['pred'][batch_i], stats['TP'][batch_i], stats['FP'][batch_i], stats['TN'][batch_i], stats['FN'][batch_i] = run_batch(sess, GNN, batch, batch_i, 0, time_steps, train=False, verbose=True)

        summarize_epoch(0,stats['loss'],stats['acc'],stats['sat'],stats['pred'],train=False)

