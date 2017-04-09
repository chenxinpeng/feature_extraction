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


images = sorted(glob.glob(mscoco_train_images_path + '/*.jpg'))


with tf.Graph().as_default():
    sess = tf.Session()

    for idx, image_path in enumerate(images):
        start_time = time.time()        
        image_name = os.path.basename(image_path)

        # read image, first way
        url = 'file://' + image_path
        image_string = urllib2.urlopen(url).read()
        
        # read image, second way
        #filename_queue = tf.train.string_input_producer(['./tmp/EnglishCockerSpaniel_simon.jpg'])
        #reader = tf.WholeFileReader()
        #_, image_string = reader.read(filename_queue)
        
        image = tf.image.decode_jpeg(image_string, channels=3)
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_images  = tf.expand_dims(processed_image, 0)
        
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception.inception_v4_arg_scope()):
            features_2 = inception.inception_v4(processed_images, num_classes=1001, is_training=False, create_aux_logits=False)
        
        #probabilities = tf.nn.softmax(logits)
        
        if idx == 0:
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'inception_v4.ckpt'),
                                                     slim.get_model_variables('InceptionV4'))   
        
            init_fn(sess)

        #np_image, probabilities, features_2 = sess.run([image, probabilities, features_2])
        np_image, features = sess.run([image, features_2])

        feature_save_path = train_images_feats_save_path + '/' + image_name + '.npy'

        np.save(feature_save_path, features)

        print('{}  {}  time cost: {:.3f}'.format(idx, image_name, time.time()-start_time))


