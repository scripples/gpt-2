
import re
import time

from google.cloud import storage # sudo pip3 install google-cloud-storage

import tensorflow as tf

import functools


def op_scope(fn, name=None):
    if name is None:
        name = fn.__name__
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        with tf.name_scope(fn.__name__):
            return fn(*args, **kwargs)
    return _fn


class State:
  pass


if 'state' not in globals():
  state = State()
  state.client = None


def gs_filesize(filename):
  """tf.string.length unfortunately fails for files larger than 2GB due to its result being a 32-bit integer. Punt by asking gsutil for the filesize."""
  import subprocess
  result = int(subprocess.run(['gsutil', 'du', '-s', filename], stdout=subprocess.PIPE, check=True).stdout.split()[0])
  if result <= 0:
    raise FileNotFoundError("Blob path does not exist or is zero length: {!r}".format(filename))
  return result


@op_scope
def tf_file_contents(filename):
  size = gs_filesize(filename)
  data = tf.raw_ops.ReadFile(filename=filename);
  return data, size


@op_scope
def tf_file_data(filename, out_dtype=None):
  data, size = tf_file_contents(filename)
  if out_dtype == tf.string:
    out_dtype = None
  if out_dtype is not None:
    if size % out_dtype.size != 0:
      raise ValueError("Size of file isn't divisible by dtype size. File size: {!r} dtype size: {!r} dtype: {!r}".format(size, out_dtype.size, out_dtype))
    data = tf.io.decode_raw(data, out_dtype);
    data.set_shape((size // out_dtype.size,));
  return data, size


@op_scope
def tf_file_variable(filename, dtype=None, **kws):
  data, size = tf_file_data(filename, out_dtype=dtype)
  v = tf.Variable(data, dtype=dtype, **kws)
  return v


@op_scope
def tf_tok16_variable(filename, **kws):
  #use_resource = kws.pop('use_resource', True)
  use_resource = kws.pop('use_resource', False)
  trainable = kws.pop('trainable', False)
  collections = kws.pop('collections', ['local_variables'])
  dtype = kws.pop('dtype', tf.int32)
  #return tf_file_variable(filename, dtype=dtype, collections=collections, trainable=trainable, use_resource=use_resource, **kws)
  data, size = tf_file_data(filename, out_dtype=tf.uint16)
  data1 = tf.cast(data, dtype)
  v = tf.Variable(data1, dtype=dtype, collections=collections, trainable=trainable, use_resource=use_resource, **kws)
  return v


@op_scope
def sample_text(x, amount, batch_size=None):
  if batch_size is not None:
    features, labels = [], []
    for i in range(batch_size):
      features1, labels1 = sample_text(x, amount)
      features.append(features1)
      labels.append(labels1)
    features = tf.stack(features)
    labels = tf.stack(labels)
    return features, labels
  s = tf.size(x, out_type=tf.dtypes.int64)
  r = tf.random.uniform([], maxval=s-(amount+1), dtype=tf.dtypes.int64)
  r1 = tf.range(r, r+amount)
  r2 = tf.range(r+1, (r+1)+amount)
  r1 = tf.reshape(r1, [amount]) # Somehow, this makes the compiler happy
  r2 = tf.reshape(r2, [amount]) # TPUs want constant sized input, and these reshapes makes it recognize the shape of the input
  vals1 = tf.gather(x, r1)
  vals2 = tf.gather(x, r2)
  vals1 = tf.cast(vals1, tf.dtypes.int32)
  vals2 = tf.cast(vals2, tf.dtypes.int32)
  features, labels = vals1, vals2
  return features, labels


@op_scope
def sample_tok16(x, amount, batch_size=1):
  s = tf.size(x, out_type=tf.dtypes.int64)
  r = tf.random.uniform([batch_size, 1], maxval=s-(amount+1), dtype=tf.dtypes.int64)
  #r1 = tf.tile(tf.expand_dims(tf.range(amount, dtype=tf.dtypes.int64), 0), [batch_size, 1])
  r1 = tf.stack([tf.range(amount, dtype=tf.int64) for _ in range(batch_size)])
  r2 = r1 + r
  return tf.gather(x, r2)


from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.python.tpu import tpu; tpu_ops = tpu.tpu_ops;


@op_scope
def tf_tpu_cpu(f, *args, **kws):
  return tpu.outside_compilation(f, *args, **kws)


@op_scope
def sample_tok16_cpu(x, amount, batch_size=1):
  return tpu.outside_compilation(sample_tok16, x, amount, batch_size=batch_size)


from tensorflow.core.protobuf import config_pb2
from functools import partial

def tf_session_run_timeout(session, timeout_in_ms=10000):
  return partial(session.run, options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms))


def tf_foo():
  print('foo3')


def is_cloud_path(x):
  return ':/' in x and x.index(':') < x.index('/')


def path_parse(x):
  if '/' not in x:
    return '', '', x
  root = ''
  if is_cloud_path(x):
    root, x = x.split(':/', 1)
    root += ':/'
  else:
    while x.startswith('/'):
      root += '/'
      x = x[1:]
  dirname = ''
  if '/' in x:
    dirname, x = x.rsplit('/', 1)
    dirname += '/'
  return root, dirname, x


def gs_client(client=None):
  if client is not None:
    return client
  if state.client is None:
    state.client = storage.Client()
  return state.client


def gs_path_parse(path):
  root, dirname, basename = path_parse(path)
  if root != 'gs:/':
    raise ValueError("expected path like gs://foo/bar, got {!r}".format(path))
  assert dirname.startswith('/')
  if dirname == '/':
    bucket = basename
    basename = ''
  else:
    bucket, dirname = dirname.lstrip('/').split('/', 1)
  blob = dirname.rstrip('/') + '/' + basename.lstrip('/')
  blob = blob.lstrip('/') # remove leading slash from blob path
  return bucket, blob


def gs_blob(path, client=None):
  bucket_name, blob_path = gs_path_parse(path)
  client = gs_client(client)
  bucket = client.get_bucket(bucket_name)
  blob = bucket.get_blob(blob_path)
  return blob


def gs_size(path):
  fp = gs_blob(path)
  return fp.size

