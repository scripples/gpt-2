import os
import functools

import tensorflow as tf
from tensorflow.python.framework.ops import _DefaultStack
from tensorflow.python.util import tf_contextlib


def variable_scope(*args, **kws):
  return tf.variable_scope(*args, **kws, use_resource=True)


def op_scope(fn, name=None):
    if name is None:
        name = fn.__name__
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        with tf.name_scope(fn.__name__):
            return fn(*args, **kwargs)
    return _fn



class DefaultValue(_DefaultStack):
  def __init__(self, value):
    super(DefaultValue, self).__init__()
    self.stack.append(value)

  @property
  def global_value(self):
    return self.stack[0]

  @global_value.setter
  def global_value(self, value):
    self.stack[0] = value

  @property
  def value(self):
    return self.stack[-1]

  def set(self, value):
    self.stack[-1] = value

  def overlay(self, value):
    return self.get_controller(value)


_highway_size = DefaultValue(int(os.environ.get('NUM_CORES', '1')))
_current_lane = DefaultValue(0)


@tf_contextlib.contextmanager
def switch_lane(lane):
  highway_size = get_highway_size()
  if lane >= highway_size:
    raise ValueError("Tried to switch lane to {}, but highway size is {}".format(lane, highway_size))
  with _current_lane.overlay(lane):
    with tf.device(device_for_core()):
      with tf.name_scope('lane__%d_of_%d' % (lane, highway_size)):
        yield


def swerve_across_highway():
  highway_size = get_highway_size()
  for lane in range(highway_size):
    with switch_lane(lane):
      yield lane


def get_current_lane():
  return _current_lane.value


def get_highway_size():
  return _highway_size.value



def device_for_core(task=0, core=None, job=None):
  if core is None:
    core = get_current_lane()
  # #return "/job:%s/task:%d/device:TPU_REPLICATED_CORE:%d" % (job, task, core)
  # return None
  if 'TPU_NAME' in os.environ or 'COLAB_TPU_ADDR' in os.environ: # TODO: check whether current session has TPU devices
    if job is None:
      job = "worker"
    return "/job:%s/task:%d/device:TPU_REPLICATED_CORE:%d" % (job, task, core)
  elif 'NUM_CORES' in os.environ:
    if job is None:
      job = "localhost"
    return "/job:%s/task:%d/device:CPU:%d" % (job, task, core)


def alist(x):
  return isinstance(x, list)


def at(l, index=None):
  if isinstance(l, list):
    if index is None:
      index = get_current_lane()
    return l[index]
  return l


def pmap(fun, *args, **kws):
  #assert alist(h)
  highway_size = get_highway_size()
  ops = []
  for lane in swerve_across_highway():
    argv = [at(h) for h in args]
    argkw = {k: at(v) for k, v in kws.items()}
    result = fun(*argv, **argkw)
    ops.append(result)
  return ops


identity = functools.partial(pmap, tf.identity)


@op_scope
def static_shape(h):
  return at(h).shape


@op_scope
def col_parallel(h, dtype, scope, input_size, output_size):
  assert alist(h)
  highway_size = get_highway_size()
  output_size_per_partition = divide(output_size, highway_size)
  w, b = init_weight_bias_parallel(dtype, scope, [input_size, output_size_per_partition], axis=-1, use_bias=True)
  # h0 = []
  # for lane in swerve_across_highway():
  #   x = tf.matmul(at(h), at(w))
  #   x = tf.add(x, at(b))
  #   h0.append(x)
  def fork(h, w, b):
    x = tf.matmul(h, w)
    x = tf.add(x, b)
    return x
  return pmap(fork, h, w, b)


@op_scope
def row_parallel(h, dtype, scope, input_size, output_size):
  assert alist(h)
  highway_size = get_highway_size()
  input_size_per_partition = divide(input_size, highway_size)
  #w, b = init_weight_bias_parallel(dtype, scope, [input_size_per_partition, output_size], axis=-2, use_bias='full')
  w, b = init_weight_bias_parallel(dtype, scope, [input_size_per_partition, output_size], axis=-2, use_bias='mirror')
  h1 = pmap(tf.matmul, h, w)
  def fork(b):
    #h2 = tf.stack(h1, name='row_parallel_stack')
    #h3 = tf.reduce_sum(h2, axis=0, name='row_parallel_reduce_sum')
    h3 = tf.add_n(h1, name='row_parallel_reduce_sum')
    h4 = tf.add(h3, b, name='row_parallel_add_bias')
    return h4
  return pmap(fork, b)


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def init_weight_bias_parallel(dtype, scope, shape, axis, use_bias=True, weight_name='w', bias_name='b'):
  weight = []
  bias = [] if use_bias else None
  highway_size = get_highway_size()
  with variable_scope(scope, dtype=dtype):
    if highway_size <= 1 and False:
      weight = init_variable(weight_name, shape, initializer=normal_initializer(dtype=dtype))
      if use_bias:
        bias = init_variable(bias_name, shape[0], initializer=constant_initializer(dtype=dtype))
    else:
      for lane in swerve_across_highway():
        w = init_variable(weight_name + '__slice__%d_of_%d_of_%d' % (axis, lane, highway_size), shape, initializer=normal_initializer(dtype=dtype))
        weight.append(w)
        if use_bias and use_bias != 'full':
          if use_bias == 'mirror':
            b = init_variable(bias_name + '__mirror__%d_of_%d' % (lane, highway_size), shape[-1], initializer=constant_initializer(dtype=dtype))
          else:
            assert use_bias is True
            b = init_variable(bias_name + '__slice__%d_of_%d_of_%d' % (axis, lane, highway_size), shape[-1], initializer=constant_initializer(dtype=dtype))
          bias.append(b)
      if use_bias == 'full':
        bias = init_variable(bias_name, shape[-1], initializer=constant_initializer(dtype=dtype))
  return weight, bias


# def init_weight_bias_parallel(dtype, scope, input_size, output_size, axis, use_bias=True, weight_name='w', bias_name='b'):
#   weight = []
#   bias = [] if use_bias else None
#   world_size = get_model_parallel_world_size()
#   with variable_scope(scope, dtype=dtype):
#     if world_size <= 1 and False:
#       weight = init_variable(weight_name, [input_size, output_size], initializer=normal_initializer(dtype=dtype))
#       if use_bias:
#         bias = init_variable(bias_name, [output_size], initializer=constant_initializer(dtype=dtype))
#     else:
#       for core in range(world_size):
#         with tf.device(device_for_core(core=core)):
#           w = init_variable(weight_name + '__slice__%d_of_%d_of_%d' % (axis, core, world_size), [input_size, output_size], initializer=normal_initializer(dtype=dtype))
#           weight.append(w)
#           if use_bias and use_bias != 'full':
#             b = init_variable(bias_name + '__slice__%d_of_%d_of_%d' % (axis, core, world_size), [output_size], initializer=constant_initializer(dtype=dtype))
#             bias.append(b)
#       if use_bias == 'full':
#         bias = init_variable(bias_name, [output_size], initializer=constant_initializer(dtype=dtype))
#   return weight, bias


@op_scope
def get_variable(name):
    name = os.path.join(tf.get_variable_scope().name, name)
    vs = tf.trainable_variables()
    for x in vs:
        if x.name.startswith(name + ':'):
            return x


@op_scope
def init_variable(name, shape, initializer):
  v = get_variable(name)
  if v is not None:
    return v
  return tf.get_variable(name, shape=shape, initializer=initializer, use_resource=True)


@op_scope
def init_variable_mirrored(name, shape, initializer):
  highway_size = get_highway_size()
  name = name.split('__mirror__')[0]
  vs = []
  for lane in swerve_across_highway():
    v = init_variable(name + '__mirror__%d_of_%d' % (lane, highway_size), shape, initializer=initializer)
    vs.append(v)
  return vs


@op_scope
def normal_initializer(dtype, stddev=0.02):
  return tf.random_normal_initializer(stddev=stddev, dtype=dtype)


@op_scope
def constant_initializer(dtype, value=0):
  return tf.constant_initializer(value=value, dtype=dtype)
