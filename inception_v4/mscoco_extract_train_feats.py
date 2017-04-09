#! encoding: UTF-8

import numpy as np
import os
import glob
import time

import tensorflow as tf

import urllib2
#import matplotlib.pyplot as plt

from datasets import imagenet
from nets import inception
from preprocessing import inception_preprocessing

import ipdb
checkpoints_dir = './checkpoints'
if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)

slim = tf.contrib.slim

image_size = inception.inception_v4.default_image_size

mscoco_train_images_path = '/data1/ailab_view/dataset/mscoco/train2014'
mscoco_val_images_path = '/data1/ailab_view/dataset/mscoco/val2014'
mscoco_test_images_path = '/data1/ailab_view/dataset/mscoco/test2014'

train_images_feats_save_path = '/data1/ailab_view/image_caption/inception/official_splits/train_feats_v4'
val_images_feats_save_path = '/data1/ailab_view/image_caption/inception/official_splits/val_feats_v4'
test_images_feats_save_path = '/data1/ailab_view/image_caption/inception/official_splits/test_feats_v4'

images_lists = sorted(glob.glob(mscoco_train_images_path + '/*.jpg'))

tf_image = tf.placeholder(tf.string, None)
#tf_image = tf.placeholder(tf.int32, [None, None, 3])
image = tf.image.decode_jpeg(tf_image, channels=3)

processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
processed_images  = tf.expand_dims(processed_image, 0)

with slim.arg_scope(inception.inception_v4_arg_scope()):
    tf_features = inception.inception_v4(processed_images, num_classes=1001, is_training=False, create_aux_logits=False)

init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'inception_v4.ckpt'), slim.get_model_variables('InceptionV4'))   
        
with tf.Session() as sess:    

    init_fn(sess)

    for idx, image_path in enumerate(images_lists):
        start_time = time.time()
        
        image_name = os.path.basename(image_path)
        
        url = 'file://' + image_path
        image_string = urllib2.urlopen(url).read()
        
        #image_raw_data = tf.gfile.FastGFile(image_path, 'r').read()
        #image = tf.image.decode_jpeg(image_raw_data, channels=3)
        #image = sess.run(image)
        #print(image.shape)
        #ipdb.set_trace()
        
        features = sess.run(tf_features, feed_dict={tf_image: image_string})
        
        features = np.reshape(features, [-1, 1536])
        
        feature_save_path = train_images_feats_save_path + '/' + image_name + '.npy'
        np.save(feature_save_path, features)
        
        print('{}  {}  time cost: {:.5f}'.format(idx, image_name, time.time()-start_time))

