# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:01:27 2019

@author: kesha
"""
import os
import sys
import string
from datetime import datetime as dt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def gen_rand_string_data(data_count,                        
                         min_char_count = 3, 
                         max_char_count = 8,
                         max_char = 16,
                         x_pos = 'side',
                         img_size = (32,256,1),
                         font = [cv2.FONT_HERSHEY_SIMPLEX], 
                         font_scale = np.arange(0.7, 1, 0.1), 
                         thickness = range(1, 3, 1)):
  '''
  random string data generation
  ''' 
  #start_time=dt.now() 
  images = []
  labels = []
  color = (255,255,255)
  count = 0
  char_list = list(string.ascii_letters) \
              + list(string.digits) \
              + list(' ')     
  while(1):
    
    for fs in font_scale:
      for thick in thickness:
        for f in font:
          img = np.zeros(img_size, np.uint8)
          char_count = np.random.randint(min_char_count, \
                                         (max_char_count + 1))
          rand_str = ''.join(np.random.choice(char_list, \
                                              char_count))
          #generate image data
          text_size = cv2.getTextSize(rand_str, f, fs, thick)[0]  
          if(x_pos == 'side'):
            org_x = 0
          else:
            org_x = (img_size[1] - text_size[0])//2         
          org_y = (img_size[0] +  text_size[1])//2
          cv2.putText(img, rand_str, (org_x, org_y), f, fs, \
                      color, thick, cv2.LINE_AA)
          
          label = list(rand_str) + [' '] \
          * (max_char - len(rand_str))
          for i,t in enumerate(label):
            label[i] = char_list.index(t)
            
          label = np.uint8(label)
          images.append(img)
          labels.append(label)        
          count +=1
          if count == data_count:
            break
        else: continue
        break
      else: continue
      break
    else: continue
    break  
  #end_time = dt.now()  
  #print("time taken to generate data", end_time - start_time)          
  return images, labels

def _bytes_feature(value):
  return tf.train.Feature \
         (bytes_list=tf.train.BytesList(value=[value]))
def write_tfrecords(all_features, all_labels, file):
  '''
  write data to a tfrecords file
  '''
  #start_time=dt.now()
  writer = tf.python_io.TFRecordWriter(file)
  for features, labels in zip(all_features, all_labels):
      feature = {'labels': _bytes_feature(tf.compat.as_bytes \
                           (np.array(labels).tostring())),
                 'images': _bytes_feature(tf.compat.as_bytes \
                           (np.array(features).tostring()))}
      example = tf.train.Example(features=tf.train.Features \
                                 (feature=feature))
      writer.write(example.SerializeToString())    
  writer.close()
  #end_time = dt.now()  
  #print("time taken to write data", end_time - start_time)
  
Folder_path = './tfrecordoutput/'
keyword = '3to8'
train_filename = folder_path + 'train_' + keyword + '_'
test_filename = folder_path + 'test_' + keyword + '_'
file_count = 2
img_size = [32,256,1]
max_char = 8
class_count = 63
batch_size = 32
num_of_threads=16
min_after_dequeue=5000
capacity=min_after_dequeue+(num_of_threads+1)*batch_size
with tf.Graph().as_default():
  image_batch, label_batch=minibatch(batch_size, train_filename \
                   , file_count, img_size, max_char, class_count)
  init=tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord) 
    for i in range(5):
      image_b, label_b= sess.run([image_batch, label_batch])
      if(i==0):
        print('data type of image:', type(image_b[0][0,0,0]))
        print('data type of label:', type(label_b[0][0,0]))
        print("shape of image_batch:", image_b.shape)
        print('shape of label_out:', label_b.shape)
      plt.imshow(np.reshape(image_b[0],[32,256]), cmap = 'gray')
      plt.show()
      print(sess.run(tf.transpose(label_b[0])))
    coord.request_stop()
    coord.join(threads)
  
folder_path = './tfrecordoutput/'
file_count = 2
train_data_count = 8192
test_data_count = 2048
print('total train data =', file_count * train_data_count)
print('total test data =', file_count * test_data_count)
keyword = '3to8'
for i in range(file_count):
  index = i+1
  train_filename =folder_path+"train_"+keyword+"_%d.tfrecords"%index
  test_filename =folder_path+"test_"+keyword+"_%d.tfrecords"%index
  print('generating train file number %d'%(i+1))
  images, labels = gen_rand_string_data(train_data_count)
  write_tfrecords(images, labels, train_filename)                     
  print('train file number %d generated'%(i+1))
  print('generating test file number %d'%(i+1))
  images, labels, gen_rand_string_data(test_data_count)
  write_tfrecords(images, labels, test_filename)
  print('test file number %d generated'%(i+1))
