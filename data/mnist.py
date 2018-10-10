#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""tf.data.Dataset interface to the MNIST dataset."""

import os
import tensorflow as tf


def download(directory, filename):
  filepath = os.path.join(directory, filename)
  if tf.gfile.Exists(filepath):
    return filepath
  raise IOError('Not found: ' + filepath)


def dataset(directory, images_file, labels_file):
  images_file = download(directory, images_file)
  labels_file = download(directory, labels_file)

  def decode_image(image):
    # Normalize from [0, 255] to [0.0, 1.0]
    image = tf.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [28,28,1])
    return tf.to_float(tf.less(tf.random_uniform([28,28,1]), (image / 255.0)))

  def decode_label(label):
    label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
    label = tf.reshape(label, [])  # label is a scalar
    return tf.to_int32(label)

  images = tf.data.FixedLengthRecordDataset(images_file, 28 * 28, header_bytes=16).map(decode_image)
  labels = tf.data.FixedLengthRecordDataset(labels_file, 1, header_bytes=8).map(decode_label)
  return images, labels


def train():
  """tf.data.Dataset object for MNIST training data."""
  directory = 'data/mnist'
  return dataset(directory, 'train-images.idx3-ubyte', 'train-labels.idx1-ubyte')


def test():
  """tf.data.Dataset object for MNIST test data."""
  directory = 'data/mnist'
  return dataset(directory, 't10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')