import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams
import os
import math
import tflex
from collections import OrderedDict
from tensorflow.python.framework import ops as tf_ops
import pdb
import functools


class Context():
  def __init__(self):
    self.should_break = False

  def set_trace(self):
    if self.should_break:
      pdb.set_trace()


api = Context()

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




def default_hparams():
    return HParams(
        n_vocab=50257,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
        res_dropout=0.1,
        attn_dropout=0.1,
        dtype=tf.float32
    )

 #_cores = None
 #
 #def get_core(i, session=None):
 #  global _cores
 #  if session is None:
 #    session = tf.get_default_session()
 #  if _cores is None:
 #    _cores = session.list_devices()[2:]
 #  n = len(_cores)
 #  if n <= 0:
 #    return None
 #  result = _cores[i % n].name
 #  if 'GPT2_VERBOSE' in os.environ:
 #    print(result)
 #  return result

def get_cores(session=None):
  if session is None:
    session = tf.get_default_session()
  cores = session.list_devices()[2:2+8]
  cores = cores[::-1]
  return cores

def get_core(i, session=None):
  cores = get_cores(session=session)
  if len(cores) > 0:
    return cores[i % len(cores)].name


@op_scope
def get_variable(name):
    name = os.path.join(tf.get_variable_scope().name, name)
    vs = tf.trainable_variables()
    for x in vs:
        if x.name.startswith(name + ':'):
            return x

@op_scope
def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

@op_scope
def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

@op_scope
def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

@op_scope
def norm(x, scope, *, axis=-1, epsilon=1e-5, hparams=None):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    dtype = hparams.dtype if hparams else tf.float32
    with variable_scope(scope, dtype=dtype):
        n_state = x.shape[-1].value
        g = init_variable('g', [n_state], tf.constant_initializer(1, dtype=dtype))
        b = init_variable('b', [n_state], tf.constant_initializer(0, dtype=dtype))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x

@op_scope
def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, u, v = shape_list(x)
    m = u * v
    return tf.reshape(x, start + [n, m//n])

@op_scope
def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

@op_scope
def normal_initializer(dtype, stddev=0.02):
  return tf.random_normal_initializer(stddev=stddev, dtype=dtype)

@op_scope
def constant_initializer(dtype, value=0):
  return tf.constant_initializer(value=value, dtype=dtype)

@op_scope
def init_variable(name, shape, initializer):
  v = get_variable(name)
  if v is not None:
    return v
  return tf.get_variable(name, shape=shape, initializer=initializer, use_resource=True)

@op_scope
def conv1d(x, scope, nf, *, hparams=None):
    dtype = hparams.dtype if hparams else tf.float32
    with variable_scope(scope, dtype=dtype):
        *start, nx = shape_list(x)
        w = init_variable('w', [nx, nf], initializer=normal_initializer(dtype=dtype))
        b = init_variable('b', [nf], initializer=constant_initializer(dtype=dtype))
        lhs = tf.reshape(x, [-1, nx])
        rhs = tf.reshape(w, [-1, nf])
        if False: # noticeable slowdown https://i.imgur.com/95VAycJ.png
          lhs_n = tf.split(lhs, 8, axis=1)
          rhs_n = tf.split(rhs, 8, axis=0)
          ops = []
          for i in range(8):
            with tf.device(get_core(i)):
              ops.append(tf.matmul(lhs_n[i], rhs_n[i]))
          W = tf.reduce_sum(ops, axis=0)
        else:
          W = tf.matmul(lhs, rhs)
        lhs1 = W+b
        rhs1 = start+[nf]
        c = tf.reshape(lhs1, rhs1)
        #if api.should_break: pdb.set_trace()
        return c

@op_scope
def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


@op_scope
def attn(x, scope, n_state, *, past, hparams, batch_size, seq_length):
    assert x.shape.ndims == 2  # Should be [batch*sequence, features]
    assert n_state % hparams.n_head == 0
    *start, hidden_size = shape_list(x)
    num_attention_heads = hparams.n_head
    assert(hidden_size % num_attention_heads == 0)
    size_per_head = hidden_size // num_attention_heads

    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    @op_scope
    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        x = tf.reshape(x, [batch_size, seq_length, num_attention_heads, size_per_head])
        x = split_states(x, hparams.n_head)
        return tf.transpose(x, [0, 2, 1, 3])

    @op_scope
    def merge_heads(x):
        # Reverse of split_heads
        x = tf.transpose(x, [0, 2, 1, 3])
        x = merge_states(x)
        x = tf.reshape(x, [batch_size * seq_length, num_attention_heads * size_per_head])
        return x

    @op_scope
    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(65500 if w.dtype == tf.float16 else 1e10, w.dtype)*(1-b)
        return w

    @op_scope
    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        w = dropout(w, hparams.attn_dropout)
        a = tf.matmul(w, v)
        return a

    dtype = hparams.dtype if hparams else tf.float32
    with variable_scope(scope, dtype=dtype):
        c = conv1d(x, 'c_attn', n_state*3, hparams=hparams)
        q, k, v = map(split_heads, tf.split(c, 3, axis=-1))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, hparams=hparams)
        a = dropout(a, hparams.res_dropout)
        return a, present


@op_scope
def mlp(x, scope, n_state, *, hparams):
    dtype = hparams.dtype if hparams else tf.float32
    with variable_scope(scope, dtype=dtype):
        nx = x.shape[-1].value
        h0 = conv1d(x, 'c_fc', n_state, hparams=hparams)
        h1 = gelu(h0)
        h2 = conv1d(h1, 'c_proj', nx, hparams=hparams)
        h2 = dropout(h2, hparams.res_dropout)
        #if api.should_break: pdb.set_trace()
        return h2

# @op_scope
# def attn_parallel(x, scope, n_state, *, past, hparams, batch_size, seq_length):
#     assert x.shape.ndims == 2  # Should be [batch*sequence, features]
#     assert n_state % hparams.n_head == 0
#     *start, hidden_size = shape_list(x)
#     num_attention_heads = hparams.n_head
#     assert(hidden_size % num_attention_heads == 0)
#     world_size = get_model_parallel_world_size()
#     size_per_head = hidden_size // num_attention_heads
#     ensure_divisibility(size_per_head, world_size)
#     size_per_head //= world_size

#     if past is not None:
#         assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

#     @op_scope
#     def split_heads(x):
#         # From [batch, sequence, features] to [batch, heads, sequence, features]
#         x = tf.reshape(x, [batch_size, seq_length, num_attention_heads, size_per_head])
#         x = split_states(x, hparams.n_head)
#         return tf.transpose(x, [0, 2, 1, 3])

#     @op_scope
#     def merge_heads(x):
#         # Reverse of split_heads
#         x = tf.transpose(x, [0, 2, 1, 3])
#         x = merge_states(x)
#         x = tf.reshape(x, [batch_size * seq_length, num_attention_heads * size_per_head])
#         return x

#     @op_scope
#     def mask_attn_weights(w):
#         # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
#         _, _, nd, ns = shape_list(w)
#         b = attention_mask(nd, ns, dtype=w.dtype)
#         b = tf.reshape(b, [1, 1, nd, ns])
#         w = w*b - tf.cast(65500 if w.dtype == tf.float16 else 1e10, w.dtype)*(1-b)
#         return w

#     @op_scope
#     def multihead_attn(q, k, v):
#         # q, k, v have shape [batch, heads, sequence, features]
#         w = tf.matmul(q, k, transpose_b=True)
#         w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

#         w = mask_attn_weights(w)
#         w = softmax(w)
#         w = dropout(w, hparams.attn_dropout)
#         a = tf.matmul(w, v)
#         return a

#     dtype = hparams.dtype if hparams else tf.float32
#     with variable_scope(scope, dtype=dtype):
#         nx = x.shape[-1].value
#         x0 = [x for _ in range(world_size)]
#         #c = conv1d(x, 'c_attn', n_state*3, hparams=hparams)
#         ensure_divisibility(n_state*3, world_size)
#         c_w0, c_b0 = init_weight_bias_parallel(dtype, 'c_attn', nx, n_state*3 // world_size, axis=-1, use_bias=True)
#         p_w0, p_b0 = init_weight_bias_parallel(dtype, 'c_proj', n_state*3 // world_size, axis=-1, use_bias=True)
#         #h0 = [tf.matmul(x, w, name='c_attn_h0__slice__%d_of_%d' % (i, world_size)) + b for i, x, w, b in zip(range(world_size), x0, c_w0, c_b0)]
#         ops = []
#         present = []
#         for i, x, w, b in zip(range(world_size), x0, c_w0, c_b0):
#           with tf.device(device_for_tpu_core(core=i)):
#             c = tf.matmul(x, w, name='c_attn__slice__%d_of_%d' % (i, world_size)) + b
#             q, k, v = map(split_heads, tf.split(c, 3, axis=-1))
#             present.append(tf.stack([k, v], axis=1))
#             if past is not None:
#                 pk, pv = tf.unstack(past, axis=1)
#                 k = tf.concat([pk, k], axis=-2)
#                 v = tf.concat([pv, v], axis=-2)
#             import pdb; pdb.set_trace()
#             a = multihead_attn(q, k, v)
#             a = merge_heads(a)
#             a = conv1d(a, 'c_proj', n_state, hparams=hparams)
#             a = dropout(a, hparams.res_dropout)
#             ops.append(a)
#         import pdb; pdb.set_trace()
#         return ops, present




@op_scope
def attn_parallel(x, scope, n_state, *, past, hparams, batch_size, seq_length):
    assert x.shape.ndims == 2  # Should be [batch*sequence, features]
    assert n_state % hparams.n_head == 0
    *start, hidden_size = shape_list(x)
    num_attention_heads = hparams.n_head
    assert(hidden_size % num_attention_heads == 0)
    size_per_head = hidden_size // num_attention_heads
    world_size = get_model_parallel_world_size()

    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    @op_scope
    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        x = tf.reshape(x, [batch_size, seq_length, num_attention_heads, size_per_head])
        x = split_states(x, hparams.n_head)
        return tf.transpose(x, [0, 2, 1, 3])

    @op_scope
    def merge_heads(x):
        # Reverse of split_heads
        x = tf.transpose(x, [0, 2, 1, 3])
        x = merge_states(x)
        x = tf.reshape(x, [batch_size * seq_length, num_attention_heads * size_per_head])
        return x

    @op_scope
    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(65500 if w.dtype == tf.float16 else 1e10, w.dtype)*(1-b)
        return w

    @op_scope
    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        w = dropout(w, hparams.attn_dropout)
        a = tf.matmul(w, v)
        return a

    dtype = hparams.dtype if hparams else tf.float32
    with variable_scope(scope, dtype=dtype):
        nx = x.shape[-1].value
        #c = conv1d(x, 'c_attn', n_state*3, hparams=hparams)
        #q0, k0, v0 = tf.split(c, 3, axis=-1)
        #q00 = conv1d(x, 'c_attn...0_of_3', n_state, hparams=hparams)
        #k00 = conv1d(x, 'c_attn...1_of_3', n_state, hparams=hparams)
        #v00 = conv1d(x, 'c_attn...2_of_3', n_state, hparams=hparams)
        x0 = [x for _ in range(world_size)]
        q0 = col_parallel(x0, dtype, 'c_attn...0_of_3', nx, n_state)
        k0 = col_parallel(x0, dtype, 'c_attn...1_of_3', nx, n_state)
        v0 = col_parallel(x0, dtype, 'c_attn...2_of_3', nx, n_state)
        q0 = tf.concat(q0, axis=-1)
        k0 = tf.concat(k0, axis=-1)
        v0 = tf.concat(v0, axis=-1)
        #import pdb; pdb.set_trace()
        q, k, v = map(split_heads, [q0, k0, v0])
        #import pdb; pdb.set_trace()
        #if api.should_break: pdb.set_trace()
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a0 = multihead_attn(q, k, v)
        a1 = merge_heads(a0)
        a = split_tensor_along_last_dim(a1, world_size)
        a2 = row_parallel(a, dtype, 'c_proj', n_state, n_state)
        a3 = dropout(a2, hparams.res_dropout)
        return a3, present


def col_parallel(h, dtype, scope, input_size, output_size):
  world_size = get_model_parallel_world_size()
  ensure_divisibility(output_size, world_size)
  output_size //= world_size
  w, b = init_weight_bias_parallel(dtype, scope, input_size, output_size, axis=-1, use_bias=True)
  h0 = [tf.matmul(h, w, name='h2__slice__%d_of_%d' % (i, world_size)) + b for i, h, w, b in zip(range(world_size), h, w, b)]
  return h0


def row_parallel(h, dtype, scope, input_size, output_size):
  world_size = get_model_parallel_world_size()
  ensure_divisibility(input_size, world_size)
  input_size //= world_size
  w, b = init_weight_bias_parallel(dtype, scope, input_size, output_size, axis=-2, use_bias='full')
  h1 = [tf.matmul(h, w, name='h2__slice__%d_of_%d' % (i, world_size)) for i, h, w in zip(range(world_size), h, w)]
  h2 = tf.reduce_sum(tf.stack(h1, name='h1_stack'), axis=0, name='h_reduce_sum')
  h3 = tf.add(h2, b, name='h_add_bias')
  return h3


@op_scope
def mlp_parallel(x, scope, n_state, *, hparams):
  def func(x):
    def grad(dy, variables=None):
      api.custom_grads.append([dy, variables, x])
      #import pdb; pdb.set_trace()
      return dy
    world_size = get_model_parallel_world_size()
    dtype = hparams.dtype if hparams else tf.float32
    with variable_scope(scope, dtype=dtype):
        nx = x.shape[-1].value
        ensure_divisibility(n_state, world_size)
        ensure_divisibility(nx, world_size)

        #x0 = clone_tensor_to_each_device(x)

        x0 = [x for _ in range(world_size)]

        # w0, b0 = init_weight_bias_parallel(dtype, 'c_fc', nx, n_state // world_size, axis=-1, use_bias=True)
        # h0 = [tf.matmul(x, w, name='h0__slice__%d_of_%d' % (i, world_size)) + b for i, x, w, b in zip(range(world_size), x0, w0, b0)]

        h0 = col_parallel(x0, dtype, 'c_fc', nx, n_state)

        h1 = [gelu(h) for h in h0]

        # w2, b2 = init_weight_bias_parallel(dtype, 'c_proj', n_state // world_size, nx, axis=-2, use_bias='full')
        # h2 = [tf.matmul(h, w, name='h2__slice__%d_of_%d' % (i, world_size)) for i, h, w in zip(range(world_size), h1, w2)]
        # h = tf.reduce_sum(tf.stack(h2, name='h2_stack'), axis=0, name='h_reduce_sum')
        # h = tf.add(h, b2, name='h_add_bias')
        h = row_parallel(h1, dtype, 'c_proj', n_state, nx)
        h = dropout(h, hparams.res_dropout)
        #if api.should_break: pdb.set_trace()
    return h, grad
  return func(x)[0]


@op_scope
def dropout(x, pdrop=0.1, train=True):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, rate=pdrop)
    return x


@op_scope
def block(x, scope, *, past, hparams, attn, **attn_kws):
    dtype = hparams.dtype if hparams else tf.float32
    with variable_scope(scope, dtype=dtype):
        nx = x.shape[-1].value
        x0 = norm(x, 'ln_1', hparams=hparams)
        #a, present = attn(x0, 'attn', nx, past=past, hparams=hparams, **attn_kws)
        a, present = attn_parallel(x0, 'attn', nx, past=past, hparams=hparams, **attn_kws)
        x = x + a
        x1 = norm(x, 'ln_2', hparams=hparams)
        #m = mlp(x1, 'mlp', nx*4, hparams=hparams)
        m = mlp_parallel(x1, 'mlp', nx*4, hparams=hparams)
        #m = GPT2MLP(hidden_size=nx, hparams=hparams)(x1)
        #m = GPT2ParallelMLP(hidden_size=nx, hparams=hparams)(x1)
        #import pdb; pdb.set_trace()
        x = x + m
        return x, present

@op_scope
def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

@op_scope
def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

@op_scope
def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


@op_scope
def model(hparams, X, past=None, scope='model', reuse=tf.AUTO_REUSE, checkpoint=False):
    dtype = hparams.dtype if hparams else tf.float32
    with variable_scope(scope, reuse=reuse, dtype=dtype):
        results = {}
        batch, sequence = shape_list(X)

        wpe = init_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01, dtype=dtype))
        wte = init_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02, dtype=dtype))
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        ## We keep the representation as a 2D tensor to avoid re-shaping it back and
        ## forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
        ## the GPU/CPU but may not be free on the TPU, so we want to minimize them to
        ## help the optimizer.
        batch_size, seq_length, hidden_size = shape_list(h)
        api.should_break = past is not None
        #if api.should_break: pdb.set_trace()
        h = tf.reshape(h, [batch_size * seq_length, hidden_size])

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        #every = int(math.sqrt(hparams.n_layer))
        every = 1
        #tf.add_to_collection('checkpoints', h)
        #if api.should_break: pdb.set_trace()
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams,
                attn=attn, batch_size=batch, seq_length=sequence)
            #if layer == 10:
            if checkpoint and layer % every == 0:
                tf.add_to_collection('checkpoints', h)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f', hparams=hparams)

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        if hparams.dtype != tf.float32:
          logits = tf.cast(logits, tf.float32)
        results['logits'] = logits
        return results

from tensorflow.python.ops import gradients
import memory_saving_gradients

class Shard(object):
  pass

def randomize(context, hparams, p):
    if p > 0:
        mask = tf.random.uniform(shape=tf.shape(context)) < p
        noise = tf.random.uniform(shape=tf.shape(context), minval=0, maxval=hparams.n_vocab, dtype=tf.int32)
        return tf.where(mask, noise, context)
    else:
        return context

from contextlib import contextmanager

@contextmanager
def nullcontext(enter_result=None):
    yield enter_result

def bfloat16context(hparams):
  if hparams.dtype == tf.bfloat16:
    return tf.contrib.tpu.bfloat16_scope()
  else:
    return nullcontext()

def shape_to_list(shape):
    """Convert a Tensorflow shape to a list of ints."""
    return [dim.value for dim in shape]

def is_tf_expression(x):
    """Check whether the input is a valid Tensorflow expression, i.e., Tensorflow Tensor, Variable, or Operation."""
    return isinstance(x, (tf.Tensor, tf.Variable, tf.Operation))

def absolute_name_scope(scope):
    """Forcefully enter the specified name scope, ignoring any surrounding scopes."""
    return tf.name_scope(scope + "/")


def absolute_variable_scope(scope, **kwargs):
    """Forcefully enter the specified variable scope, ignoring any surrounding scopes."""
    return tf.variable_scope(tf.VariableScope(name=scope, **kwargs), auxiliary_name_scope=False)


@op_scope
def all_sum_plain(g, colocate=True, *args, **kws):
  r = []
  for i in range(len(g)):
    if colocate:
      with tf_ops.colocate_with(g[i]):
        r.append(tf.add_n(g))
    else:
      r.append(tf.add_n(g))
  return r

try:
    # TensorFlow 1.13
    from tensorflow.python.ops import nccl_ops
except:
    # Older TensorFlow versions
    import tensorflow.contrib.nccl as nccl_ops

def all_sum_gpu(g, *args, **kws):
  return nccl_ops.all_sum(g, *args, **kws)

from tensorflow.python.tpu.ops import tpu_ops

#def all_sum_tpu(g, *args, **kws):
#  g = tpu_ops.cross_replica_sum(g, *args, **kws)
#  return [g[i] for i in range(shape_list(g)[0])]

def all_sum_tpu(g, colocate=True, *args, **kws):
  #import pdb
  #pdb.set_trace()
  #r = tf.reduce_sum(g)
  #r = tf.reduce_sum(tf.stack(g), axis=0, keepdims=True)
  #r = tpu_ops.cross_replica_sum(g, *args, **kws)
  #r = [r[i] for i in range(shape_list(r)[0])]
  return all_sum_plain(g, colocate=colocate, *args, **kws)

def all_sum(cores, g, colocate=True, *args, **kws):
  if any(['TPU' in x for x in cores]):
    return all_sum_tpu(g, colocate=colocate, *args, **kws)
  elif any(['GPU' in x for x in cores]):
    return all_sum_gpu(g, *args, **kws)
  else:
    return all_sum_cpu(g, *args, **kws)

def init_uninitialized_vars(target_vars = None, run = lambda ops: [False] * len(ops)):
    """Initialize all tf.Variables that have not already been initialized.

    Equivalent to the following, but more efficient and does not bloat the tf graph:
    tf.variables_initializer(tf.report_uninitialized_variables()).run()
    """
    #assert_tf_initialized()
    if target_vars is None:
        target_vars = tf.global_variables()

    test_vars = []
    test_ops = []

    with tf.control_dependencies(None):  # ignore surrounding control_dependencies
        for var in target_vars:
            assert is_tf_expression(var)

            try:
                tf.get_default_graph().get_tensor_by_name(var.name.replace(":0", "/IsVariableInitialized:0"))
            except KeyError:
                # Op does not exist => variable may be uninitialized.
                test_vars.append(var)

                with absolute_name_scope(var.name.split(":")[0]):
                    test_ops.append(tf.is_variable_initialized(var))

    init_vars = [var for var, inited in zip(test_vars, run(test_ops)) if not inited]
    return [var.initializer for var in init_vars]

def shard(batch_size, hparams, learning_rate=0.0001, optimizer='sgd', noise=0.0, only_train_transformer_layers=False, colocate_gradients_with_ops=False, colocate_sum=False, use_memory_saving_gradients=False, ungate_gradients=False, global_step=None, graph=None, scope='model', skip_cores=4, max_cores=4, length=None, sample_ctx=None, encoder=None, temperature=1, top_k=0, top_p=0.0, devices=None, *args, **kws):
    if graph is None:
        graph = tf.get_default_graph()
    if length is None:
        length = hparams.n_ctx
    if sample_ctx is None:
        sample_ctx = length
    results = {}
    results['shards'] = {}
    #results = {}
    #results['present'] = []
    #results['logits'] = []
    _dev_grads = results['grads'] = OrderedDict()
    with graph.as_default():
      #batch_size, *rest = shape_list(X)
      if devices is None:
        devices = get_cores()
      else:
        devices = devices[2:2+8]
      cores = [x.name for x in devices[skip_cores:]]
      num_cores = len(cores)
      if max_cores < 0:
        num_cores = 1
      elif max_cores is not None:
        if num_cores > max_cores:
          num_cores = max_cores
      assert batch_size % num_cores == 0
      assert num_cores > 0
      cores = cores[:num_cores]
      if max_cores < 0:
        cores = [None]
      #if num_cores <= 0:
      #  return model(hparams, X, scope=scope, *args, **kws)
      print('Sharding across %d cores' % len(cores))
      assert(batch_size % num_cores == 0)
      #contexts = tf.split(X, num_cores, axis=0)
      def make_shard(i):
        with graph.as_default():
          return make_shard_1(i)
      def make_shard_1(i):
        core = cores[i]
        prefix = 'core%04d' % i
        #context = contexts[i]
        #context = tf.placeholder(tf.int32, [batch_size // num_cores, None])
        #context_in = randomize(context, hparams, noise)
        with tf.device(core), bfloat16context(hparams), variable_scope(prefix, reuse=tf.AUTO_REUSE):
          context = tf.Variable(tf.zeros(shape=[batch_size // num_cores, sample_ctx], name="context", dtype=tf.int32), dtype=tf.int32, shape=[batch_size // num_cores, sample_ctx], trainable=False)
          #context_set = tf.placeholder(tf.int32, [batch_size // num_cores, None])
          #feed_op = tf.assign(context, context_set)
          context_in = randomize(context, hparams, noise)
          output = model(hparams=hparams, X=context_in, scope=scope, checkpoint=use_memory_saving_gradients, *args, **kws)
          #if hparams.dtype == tf.bfloat16:
          #  output['logits'] = tf.cast(output['logits'], tf.float32)
          infer = None
          if encoder:
            infer = sample.sample_sequence(
                hparams=hparams, length=length,
                start_token=encoder.encoder['<|endoftext|>'],
                batch_size=batch_size,
                temperature=temperature, top_k=top_k, top_p=top_p
            )[:, 1:]

          loss_batch = tf.nn.sparse_softmax_cross_entropy_with_logits(
                  labels=context[:, 1:], logits=output['logits'][:, :-1])
          loss = tf.reduce_mean(loss_batch)
          #if hparams.dtype != tf.float32:
          #    loss = tf.cast(loss, tf.float32)

        path = scope
        if prefix is not None:
          path = prefix + '/' + path
        global_vars = [v for v in tf.global_variables() if v.name.startswith(path + '/')]
        all_vars = [v for v in tf.trainable_variables() if v.name.startswith(path + '/')]
        def should_train_variable(v):
          if only_train_transformer_layers:
            if '/h' not in v.name and '/ln_f' not in v.name:
              return False
            #for i in range(1):
            #  if ('/h%01d/' % i) in v.name:
            #    return False
            #  if ('/h%02d/' % i) in v.name:
            #    return False
          print(v)
          return True
        train_vars = [v for v in all_vars if should_train_variable(v)]

        parameter_count = sum([np.prod(v.shape.as_list()) for v in train_vars])
        print("Shard %d is using %d parameters (%.2fM) (scope='%s/')" % (i, parameter_count, parameter_count/(1024.0*1024.0), path))

        with tf.device(core), bfloat16context(hparams), variable_scope(prefix, reuse=tf.AUTO_REUSE):
          if optimizer == 'adam':
              opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
          elif optimizer == 'sgd':
              opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
          elif optimizer == 'ada':
            params = {}
            params["decay_type"] = "adam"
            #params["beta1"] = 0.9
            params["beta1"] = 0.0
            params["beta2"] = 0.999
            lr = learning_rate
            if params["decay_type"] == "adam":
                decay_rate = adafactor_decay_rate_adam(params["beta2"])
            elif params["decay_type"] == "pow":
                decay_rate = adafactor_decay_rate_pow(params["decay_exponent"])
            else:
                raise ValueError("unknown optimizer_adafactor_decay_type")

            if not "weight_decay" in params.keys():
                opt = AdafactorOptimizer(
                    learning_rate=lr,
                    decay_rate=decay_rate,
                    beta1=params["beta1"],
                    name="Adafactor")
            else:
                AdafactorWOptimizer = tf.contrib.opt.extend_with_decoupled_weight_decay(AdafactorOptimizer)

                opt = AdafactorWOptimizer(
                    weight_decay=params["weight_decay"] * lr,
                    learning_rate=lr,
                    decay_rate=decay_rate,
                    beta1=params["beta1"],
                    name="AdafactorW")
          elif optimizer == 'ada':
              import tensor2tensor.utils.optimize
              from tensor2tensor.utils import hparam
              import tensor2tensor.models.research
              from tensor2tensor.utils import registry
              ada_hparams = registry.hparams('afx_mimic_adam')
              ada_hparams.optimizer_adafactor_beta1 = 0.0
              ada_hparams.optimizer_adafactor_factored = True
              opt = tensor2tensor.utils.optimize.adafactor(learning_rate=learning_rate, hparams=ada_hparams)
          elif optimizer == None:
            pass
          else:
              exit('Bad optimizer:', optimizer)
          _dev_grads[core] = []
          r = Shard()
          r.prefix = prefix
          r.scope = scope
          r.context = context
          r.context_in = context_in
          #r.context_set = context_set
          #r.feed_op = feed_op
          r.device = core
          r.output = output
          r.infer = infer
          if optimizer is not None:
            #opt_apply = opt.minimize(loss, var_list=train_vars, global_step=global_step, colocate_gradients_with_ops=colocate_gradients_with_ops)
            gate_gradients=None
            if ungate_gradients:
              gate_gradients=tf.train.Optimizer.GATE_NONE
            if use_memory_saving_gradients:
              #grads = memory_saving_gradients.gradients(loss, train_vars, colocate_gradients_with_ops=colocate_gradients_with_ops, checkpoints='memory')
              #grads = memory_saving_gradients.gradients_memory if i == 0 else memory_saving_gradients.gradients_speed
              #grads = memory_saving_gradients.gradients_speed if i == 0 else memory_saving_gradients.gradients_speed
              grads = memory_saving_gradients.gradients
              grads = grads(loss, train_vars, colocate_gradients_with_ops=colocate_gradients_with_ops, gate_gradients=gate_gradients)
            else:
              grads = gradients.gradients(loss, train_vars, colocate_gradients_with_ops=colocate_gradients_with_ops, gate_gradients=gate_gradients)
            grads = list(zip(grads, train_vars))
            grads = [(g, v) if g is not None else (tf.zeros_like(v), v) for g, v in grads]  # replace disconnected gradients with zeros
            _dev_grads[core].append(grads)
            #opt_apply = opt.apply_gradients(grads, global_step=global_step)
            #fit = tf.tuple([loss], control_inputs=[opt_apply])
            r.loss_batch = loss_batch
            r.loss = loss
            r.opt = opt
            #r.opt_apply = opt_apply
            #r.fit = fit
          r.global_vars = global_vars
          r.all_vars = all_vars
          r.train_vars = train_vars
          r.global_vars = global_vars
          r.parameter_count = parameter_count
          results['shards'][i] = r
        #results['present'].append(r['present'])
        #results['logits'].append(r['logits'])
      #for thread in tflex.parallelize([i for i in range(num_cores)], make_shard):
      #  thread.join()
      for i in range(num_cores):
        make_shard(i)
      #import pdb
      #pdb.set_trace()
      results['shards'] = [v for i, v in sorted(list(results['shards'].items()))]

      total_grads = sum(len(grads) for grads in _dev_grads.values())
      assert len(cores) >= 1 and total_grads >= 1
      use_loss_scaling = False

      # Cast gradients to FP32 and calculate partial sum within each device.
      dev_grads = OrderedDict()  # device => [(grad, var), ...]

      for dev_idx, dev in enumerate(cores):
          with tf.name_scope("ProcessGrads%d" % dev_idx), tf.device(dev):
              sums = []

              for gv in zip(*_dev_grads[dev]):
                  assert all(v is gv[0][1] for g, v in gv)
                  g = [tf.cast(g, tf.float32) for g, v in gv]
                  g = g[0] if len(g) == 1 else tf.add_n(g)
                  sums.append((g, gv[0][1]))

              dev_grads[dev] = sums

      trainable_vars = results['shards'][0].train_vars
      _grad_shapes = [shape_to_list(var.shape) for var in trainable_vars]

      # Sum gradients across cores.
      if len(cores) > 1:
          with tf.name_scope("SumAcrossGPUs"), tf.device(None):
              for var_idx, grad_shape in enumerate(_grad_shapes):
                  g = [dev_grads[dev][var_idx][0] for dev in cores]

                  if np.prod(grad_shape):  # nccl does not support zero-sized tensors
                    g = all_sum(cores, g, colocate=colocate_sum)

                  for dev, gg in zip(cores, g):
                      dev_grads[dev][var_idx] = (gg, dev_grads[dev][var_idx][1])

      # Apply updates separately on each device.
      ops = []
      for dev_idx, (dev, grads) in enumerate(dev_grads.items()):
          with tf.name_scope("ApplyGrads%d" % dev_idx), tf.device(dev):
              # Scale gradients as needed.
              if use_loss_scaling or total_grads > 1:
                  with tf.name_scope("Scale"):
                      coef = tf.constant(np.float32(1.0 / total_grads), name="coef")
                      #coef = undo_loss_scaling(coef)
                      grads = [(g * coef, v) for g, v in grads]

              # Check for overflows.
              #with tf.name_scope("CheckOverflow"):
              #    grad_ok = tf.reduce_all(tf.stack([tf.reduce_all(tf.is_finite(g)) for g, v in grads]))

              # Update weights and adjust loss scaling.
              with tf.name_scope("UpdateWeights"):
                  # pylint: disable=cell-var-from-loop
                  #opt = _dev_opt[dev]
                  opt = results['shards'][dev_idx].opt
                  #ls_var = get_loss_scaling_var(dev)
                  #ls_var = None

                  #if not use_loss_scaling:
                  #    ops.append(tf.cond(grad_ok, lambda: opt.apply_gradients(grads), tf.no_op))
                  #else:
                  #    ops.append(tf.cond(grad_ok,
                  #                       lambda: tf.group(tf.assign_add(ls_var, loss_scaling_inc), opt.apply_gradients(grads)),
                  #                       lambda: tf.group(tf.assign_sub(ls_var, loss_scaling_dec))))
                  ops.append(opt.apply_gradients(grads))
      opt_apply = tf.group(*ops, name="TrainingOp")

      #present = tf.concat(results['present'], axis=0)
      #logits = tf.concat(results['logits'], axis=0)
      #import pdb
      #pdb.set_trace()
      #results['present'] = present
      #results['logits'] = logits
      inputs = [x.context for x in results['shards']]
      results['inputs'] = inputs
      def get_feed_dict(batch, session=None, options=None):
        session = session or tf.get_default_session()
        r = {}
        j = batch_size // num_cores
        parts = tflex.tuples(j, batch)
        assert(batch_size % num_cores == 0)
        shards = results['shards']
        def load(i, verbose=False):
          context = inputs[i]
          tokens = parts[i]
          #r[context] = tokens
          shard = shards[i]
          with tf.device(shard.device):
            if verbose:
              print('Loading context', i)
            #session.run(shard.feed_op, feed_dict={shard.context_set: tokens}, options=options)
            tflex.load(shard.context, tokens, session=session, timeout_in_ms=15000)
            #session.run(shard.feed_op, feed_dict={shard.context_set: tokens}, options=options)
            if verbose:
              print('Loaded context', i)
        for thread in tflex.parallelize([i for i in range(len(inputs))], load):
          thread.join()
        #for i, context in enumerate(inputs):
        #  tokens = parts[i]
        #  #r[context] = tokens
        #  shard = shards[i]
        #  with tf.device(shard.device):
        #    print('Loading context')
        #    #session.run(shard.feed_op, feed_dict={shard.context_set: tokens}, options=options)
        #    tflex.load(shard.context, tokens, session=session, timeout_in_ms=15000)
        #    #session.run(shard.feed_op, feed_dict={shard.context_set: tokens}, options=options)
        #    print('Loaded')
        return r
      results['feed'] = get_feed_dict
      shards = results['shards']
      opt_losses = [x.loss for x in shards]
      opt_loss = tf.reduce_mean(opt_losses)
      #the_vars = [x.global_vars for x in shards]
      the_vars = [x.train_vars for x in shards]
      gather_ops = []
      variable_count = len(the_vars[0])
      shard_count = len(the_vars)
      #opt_apply = tf.tuple([x.fit for x in shards])
      #opt_apply = tf.group([x.opt_apply for x in shards])
      #opt_apply = tf.group([shards[i].opt_apply for i in range(1,shard_count)])
      #opt_apply = tf.group([x.opt_apply for x in shards[1:]])
      #dev_grads = OrderedDict() # device => [(val, var), ...]
      #devices = [x.device for x in shards]
      #for dev_idx, dev in enumerate(devices):
      #  with tf.name_scope("GatherWeights%d" % dev_idx), tf.device(dev):
      #    #sums = []
      #    #for gv in zip(*self._dev_grads[dev]):
      #    #   assert all(v is gv[0][1] for g, v in gv)
      #    #   g = [tf.cast(g, tf.float32) for g, v in gv]
      #    #   g = g[0] if len(g) == 1 else tf.add_n(g)
      #    #   sums.append((g, gv[0][1]))
      #    #dev_grads[dev] = sums
      #    vs = []
      #    for variables in zip(*the_vars):
      #    for variables in the_vars[dev_idx]:
      #        g = [tf.cast(g, tf.float32) for g in variables]
      #        g = g[0] if len(g) == 1 else tf.add_n(g)
      #        sums.append((g, gv[0][1]))


      ##    dev_grads[dev] = sums
      pivot = 0
      #for j in range(variable_count):
      #  with tf.device(shards[pivot].device):
      #    gather_ops.append(tf.assign(the_vars[pivot][j], tf.reduce_mean([the_vars[i][j] for i in range(shard_count)], axis=0), use_locking=True))
      #with tf.control_dependencies(gather_ops):
      #  opt_gather = tf.no_op()
      #broadcast_ops = []
      #for i in range(shard_count):
      #  if i != pivot:
      #    with tf.device(shards[i].device):
      #      for j in range(variable_count):
      #        broadcast_ops.append(tf.assign(the_vars[i][j], the_vars[pivot][j], use_locking=True))
      #  ##op1 = tf.group([tf.assign(the_vars[i][j], op0) for i in range(0,shard_count)])
      #  ##op0 = tf.reduce_sum([(the_vars[i][j] - the_vars[0][j]) for i in range(shard_count)], axis=0) / (shard_count/2) + the_vars[0][j]
      #  ##op0 = tf.reduce_mean([(the_vars[i][j] - the_vars[0][j]) for i in range(shard_count)], axis=0) * interp_rate + the_vars[0][j]
      #  ##op0 = tf.reduce_sum([(the_vars[i][j] - the_vars[0][j]) for i in range(1,shard_count)], axis=0) / (shard_count - 1) * interp_rate + the_vars[0][j]
      #  ##op0 = tf.reduce_mean([(the_vars[i][j] - the_vars[0][j]) for i in range(1,shard_count)], axis=0) * interp_rate + the_vars[0][j]
      #  #op1 = tf.group([tf.assign(the_vars[i][j], op0) for i in range(0,shard_count)])
      #  #gather_ops.append(op1)
      #with tf.control_dependencies(broadcast_ops):
      #  opt_broadcast = tf.no_op()
      ##opt_gather = tf.group(gather_ops)
      ##opt_gather = tf.group(gather_ops)
      opt_gather = tf.no_op()
      opt_broadcast = tf.no_op()
      all_vars = [x.all_vars for x in shards]
      reset_ops = []
      reset_var = {}
      variable_count = len(all_vars[pivot])
      shard_count = len(all_vars)
      for j in range(variable_count):
        reset_op = tf.group([tf.assign(all_vars[i][j], all_vars[pivot][j]) for i in range(shard_count) if i != pivot])
        reset_var[all_vars[pivot][j].name] = reset_op
        reset_var[all_vars[pivot][j]] = reset_op
        reset_ops.append(reset_op)
      #opt_reset = tf.group(reset_ops)
      #def init():
      #  init_op = tf.variables_initializer(shards[0].global_vars)
      #  with tf.control_dependencies([init_op]):
      #    return tf.group(reset_ops)
      #opt_init = init()
      #with tf.control_dependencies(init_uninitialized_vars(shards[0].all_vars)):
      #  ops = [reset_var[v] for v in shards[0].all_vars]
      #init_ops = init_uninitialized_vars(shards[0].all_vars)
      #opt_init = tf.group(init_ops, name="Initialize")
      opt_init = tf.global_variables_initializer()
      #opt_train = tf.tuple([x.loss for x in shards], control_inputs=[x.opt_apply for x in shards])
      #opt_train = tf.tuple([x.loss for x in shards], control_inputs=[x.opt_apply for x in shards[1:]])
      opt_train = tf.tuple([x.loss for x in shards], control_inputs=[opt_apply])
      the = tflex.Namespace()
      results['the'] = the
      the.opt_losses = opt_losses
      the.opt_loss = opt_loss
      the.opt_apply = opt_apply
      the.opt_gather = opt_gather
      the.opt_broadcast = opt_broadcast
      the.opt_train = opt_train
      the.reset_ops = reset_ops
      the.reset_var = reset_var
      the.opt_init = opt_init
      the.vars = the_vars
      the.all_vars = all_vars
      return results



# Model parallel group that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None


# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None


def initialize_model_parallel(model_parallel_size_, topology=None):
    """
    Initialize model data parallel groups.

    Arguments:
        model_parallel_size: number of GPUs used to parallelize model.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model. The present function will
    create 4 model parallel groups and 2 data parallel grous as:
        4 model parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 data parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # if torch.distributed.get_rank() == 0:
    #     print('> initializing model parallel with size {}'.format(
    #         model_parallel_size_))
    # # Get world size and rank. Ensure some consistencies.
    # assert torch.distributed.is_initialized()
    # world_size = torch.distributed.get_world_size()
    # model_parallel_size = min(model_parallel_size_, world_size)
    # ensure_divisibility(world_size, model_parallel_size)
    # rank = torch.distributed.get_rank()
    #
    # # Build the data parallel groups.
    # global _DATA_PARALLEL_GROUP
    # assert _DATA_PARALLEL_GROUP is None, \
    #     'data parallel group is already initialized'
    # for i in range(model_parallel_size):
    #     ranks = range(i, world_size, model_parallel_size)
    #     group = torch.distributed.new_group(ranks)
    #     if i == (rank % model_parallel_size):
    #         _DATA_PARALLEL_GROUP = group
    #
    # # Build the model parallel groups.
    # global _MODEL_PARALLEL_GROUP
    # assert _MODEL_PARALLEL_GROUP is None, \
    #     'model parallel group is already initialized'
    # for i in range(world_size // model_parallel_size):
    #     ranks = range(i * model_parallel_size,
    #                   (i + 1) * model_parallel_size)
    #     group = torch.distributed.new_group(ranks)
    #     if i == (rank // model_parallel_size):
    #         _MODEL_PARALLEL_GROUP = group
    global _MODEL_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP
    if topology is None:
      topology = tflex_tpu_topology.get_topology()
    num_cores = np.prod(topology.mesh_shape)
    ensure_divisibility(num_cores, model_parallel_size_)
    #_MODEL_PARALLEL_GROUP = num_cores // model_parallel_size_
    #_DATA_PARALLEL_GROUP = model_parallel_size_
    _MODEL_PARALLEL_GROUP = tflex_tpu_device_assignment.spatial_partition(topology, model_parallel_size_)
    _DATA_PARALLEL_GROUP = _MODEL_PARALLEL_GROUP



def get_data_parallel_group():
  return None


def get_model_parallel_group():
  return None


def torch_distributed_get_world_size(group=None):
  #return 1
  return 4
  #return 2


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    # assert _MODEL_PARALLEL_GROUP is not None, \
    #     'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    # assert _DATA_PARALLEL_GROUP is not None, \
    #     'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    return torch_distributed_get_world_size(group=get_model_parallel_group())




def device_for_tpu_core(task=0, core=0, job="worker"):
  #return "/job:%s/task:%d/device:TPU_REPLICATED_CORE:%d" % (job, task, core)
  return None


@op_scope
def clone_tensor_to_each_device(tensor):
  assert not isinstance(tensor, list)
  group = get_model_parallel_group()
  world_size = torch_distributed_get_world_size(group=group)
  # Bypass the function if we are using only 1 GPU.
  if world_size == 1:
      return [tensor]
  ops = []
  for core in range(world_size):
    with tf.device(device_for_tpu_core(core=core)):
      x = tf.identity(tensor)
      ops.append(x)
  return ops


@op_scope
def split_tensor_along_last_dim(tensor, num_partitions, contiguous_split_chunks=False):
  assert not isinstance(tensor, list)
  pieces = tf.split(tensor, num_partitions, axis=-1)
  ops = []
  for core in range(num_partitions):
    with tf.device(device_for_tpu_core(core=core)):
      x = tf.identity(pieces[core])
      ops.append(x)
  return ops


@op_scope
def _reduce(input_):
    """All-reduce the the input tensor across model parallel group."""
    group = get_model_parallel_group()
    # Bypass the function if we are using only 1 GPU.
    if torch_distributed_get_world_size(group=group) == 1:
        return input_
    ## All-reduce.
    #torch_distributed_all_reduce(input_, group=group)
    output = all_sum_plain(input_)
    # world_size = torch_distributed_get_world_size(group=group)
    # ops = []
    # for core in range(world_size):
    #   with tf.colocate_with(input_[core]):
    #     ops.append(tf.add_n([x for x in input_]))
    # return ops
    return output


@op_scope
def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    group = get_model_parallel_group()
    # Bypass the function if we are using only 1 GPU.
    if torch_distributed_get_world_size(group=group) == 1:
        return input_
    # Split along last dimension.
    world_size = torch_distributed_get_world_size(group=group)
    input_list = split_tensor_along_last_dim(input_, world_size)
    # # Note: torch.split does not create contiguous tensors by default.
    # rank = torch_distributed_get_rank(group=group)
    # output = input_list[rank].contiguous()
    output = input_list
    return output


@op_scope
def _gather(input_):
    """Gather tensors and concatinate along the last dimension."""
    group = get_model_parallel_group()
    # Bypass the function if we are using only 1 GPU.
    if torch_distributed_get_world_size(group=group) == 1:
        return input_
    # # Size and dimension.
    # last_dim = input_.dim() - 1
    # rank = torch.distributed.get_rank(group=group)
    # world_size = torch.distributed.get_world_size(group=group)
    #
    # tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    # tensor_list[rank] = input_
    # torch.distributed.all_gather(tensor_list, input_, group=group)
    #
    # # Note: torch.cat already creates a contiguous tensor.
    # output = torch.cat(tensor_list, dim=last_dim).contiguous()
    output = [tf.concat(input_, axis=-1)]
    return output


# dev = tflex_tpu_device_assignment.spatial_partition(topology, get_model_parallel_world_size())
# op = tpu_ops.shard(lambda: _reduce(_split(tf.ones([16]))), device_assignment=dev)

from six import with_metaclass

from functools import partial


class BackwardCFunction:#(_C._FunctionBase, _ContextMethodMixin, _HookMixin):
  _is_legacy = False

  def _apply(self, input_, **kwargs):
    #return self._forward_cls.backward(self, *args)
    self.props = kwargs
    forward = partial(self._forward_cls.forward, self)
    backward = partial(self._forward_cls.backward, self)
    @tf.custom_gradient
    def func(x):
      def grad(dy, variables=None):
        self.grad_variables = variables
        return backward(dy)
      return forward(x), grad
    return func(input_)

  def apply(self, input_):
    if isinstance(input_, (tuple, list)):
      #return [self._apply(x, boxed=input_, index=i, is_list=True) for i, x in enumerate(input_)]
      #return [self._apply(x, boxed=input_, index=i, is_list=True) for i, x in enumerate(input_)]
      ops = []
      self.reduce = False
      for i, x in enumerate(input_):
        with tf.device(device_for_tpu_core(core=i)):
          ops.append(self._apply(x, boxed=input_, index=i, is_list=True))
      if self.reduce:
        #pdb.set_trace()
        ops = _reduce(ops)[0]
      return ops
    return self._apply(input_, boxed=[input_], index=0, is_list=False)


class FunctionMeta(type):
  """Function metaclass.

  This metaclass sets up the following properties:
      _is_legacy: True if forward is not defined as a static method.
      _backward_cls: The Function class corresponding to the differentiated
          version of this function (which is generated on the fly by this
          metaclass).
  """
  def __init__(cls, name, bases, attrs):
    for super_cls in cls.mro():
      forward = super_cls.__dict__.get('forward')
      if forward is not None:
        has_static_forward = isinstance(forward, staticmethod) or isinstance(forward, classmethod)
        break
    cls._is_legacy = not has_static_forward
    # old-style functions
    if not has_static_forward:
      return super(FunctionMeta, cls).__init__(name, bases, attrs)
    backward_fn = type(name + 'Backward', (BackwardCFunction,), {'_forward_cls': cls})
    cls._backward_cls = backward_fn
    return super(FunctionMeta, cls).__init__(name, bases, attrs)


class _Function(with_metaclass(FunctionMeta)):#, _C._FunctionBase, _ContextMethodMixin, _HookMixin)):
  @classmethod
  def apply(cls, *args, **kwargs): # real signature unknown
    return cls._backward_cls().apply(*args, **kwargs)
  @staticmethod
  def forward(ctx, input_):
    return input_
  @staticmethod
  def backward(ctx, grad_output):
    return grad_output


# class _FooFunction(_Function):
#     @staticmethod
#     def forward(ctx, input_):
#         return input_
#     @staticmethod
#     def backward(ctx, grad_output):
#         return tf.zeros_like(grad_output)


#_FooFunction.apply(tf.ones([16]))
# x = tf.constant(100.); y = _FooFunction.apply(x); dy =  tf.gradients(y, x); 


# class _CopyToModelParallelRegion(_Function):
#   """Pass the input to the model parallel region."""
#   @staticmethod
#   def forward(ctx, input_):
#     return input_
#   @staticmethod
#   def backward(ctx, grad_output):
#     return _reduce(grad_output)


# class _ReduceFromModelParallelRegion(_Function):
#   """All-redcue the input from the model parallel region."""
#   @staticmethod
#   def forward(ctx, input_):
#     return _reduce(input_)
#   @staticmethod
#   def backward(ctx, grad_output):
#     return grad_output


class _CopyToModelParallelRegion(_Function):
  """Pass the input to the model parallel region."""
  @staticmethod
  def forward(ctx, input_):
    return clone_tensor_to_each_device(input_)
  @staticmethod
  def backward(ctx, grad_output):
    return _reduce(grad_output)[0]


class _ReduceFromModelParallelRegion(_Function):
  """All-reduce the input from the model parallel region."""
  @staticmethod
  def forward(ctx, input_):
    return _reduce(input_)[0]
  @staticmethod
  def backward(ctx, grad_output):
    return clone_tensor_to_each_device(grad_output)



class _ScatterToModelParallelRegion(_Function):
  """Split the input and keep only the corresponding chuck to the rank."""
  @staticmethod
  def forward(ctx, input_):
    return _split(input_)
  @staticmethod
  def backward(ctx, grad_output):
    return _gather(grad_output)


class _GatherFromModelParallelRegion(_Function):
  """Gather the input from model parallel region and concatinate."""
  @staticmethod
  def forward(ctx, input_):
    return _gather(input_)
  @staticmethod
  def backward(ctx, grad_output):
    return _split(grad_output)


# -----------------
# Helper functions.
# -----------------


@op_scope
def copy_to_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


@op_scope
def reduce_from_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


@op_scope
def scatter_to_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)


@op_scope
def gather_from_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)


# copy_to_model_parallel_region(tf.ones([16]))













class _Module:
  def __init__(self):
    self.model_parallel = False
  def forward(self, x):
    return x
  def backward(self, dy):
    return dy
  def __call__(self, x):
    class _Call(_Function):
      @staticmethod
      def forward(ctx, *args):
        self.ctx = ctx
        return self.forward(*args)
      @staticmethod
      def backward(ctx, *args):
        self.ctx = ctx
        if self.model_parallel:
          import pdb; pdb.set_trace()
        return self.backward(*args)
    if self.model_parallel:
      #import pdb; pdb.set_trace()
      x = copy_to_model_parallel_region(x)
    return _Call.apply(x)
    



class GPT2MLP(_Module):
  def __init__(self, hidden_size, hparams):
    super(GPT2MLP, self).__init__()
    self.hparams = hparams
    self.hidden_size = hidden_size
  def forward(self, input_):
    return mlp(input_, 'mlp', self.hidden_size*4, hparams=self.hparams)


# class GPT2ParallelMLP(_Module):
#   def __init__(self, hidden_size, hparams, init_method=None):
#     super(GPT2ParallelMLP, self).__init__()
#     self.hparams = hparams
#     # Project to 4h.
#     self.dense_h_to_4h = ColumnParallelLinear(hidden_size, 4*hidden_size,
#                                               gather_output=False,
#                                               init_method=init_method)
#     # Project back to h.
#     self.dense_4h_to_h = RowParallelLinear(
#         4*hidden_size,
#         hidden_size,
#         input_is_parallel=True,
#         init_method=output_layer_init_method)
    
#   def forward(self, hidden_states):
#     # [b, s, 4hp]
#     intermediate_parallel = self.dense_h_to_4h(hidden_states)
#     #intermediate_parallel = gelu(intermediate_parallel)

#     # [b, s, h]
#     output = self.dense_4h_to_h(intermediate_parallel)
#     #output = self.dropout(output)
#     return output
    



def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def init_weight_bias_parallel(dtype, scope, input_size, output_size, axis, use_bias=True):
  weight = []
  bias = [] if use_bias else None
  world_size = get_model_parallel_world_size()
  with variable_scope(scope, dtype=dtype):
    if world_size <= 1 and False:
      weight = init_variable('w', [input_size, output_size], initializer=normal_initializer(dtype=dtype))
      if use_bias:
        bias = init_variable('b', [output_size], initializer=constant_initializer(dtype=dtype))
    else:
      for core in range(world_size):
        with tf.device(device_for_tpu_core(core=core)):
          w = init_variable('w__slice__%d_of_%d_of_%d' % (axis, core, world_size), [input_size, output_size], initializer=normal_initializer(dtype=dtype))
          weight.append(w)
          if use_bias and use_bias != 'full':
            b = init_variable('b__slice__%d_of_%d_of_%d' % (axis, core, world_size), [output_size], initializer=constant_initializer(dtype=dtype))
            bias.append(b)
      if use_bias == 'full':
        bias = init_variable('b', [output_size], initializer=constant_initializer(dtype=dtype))
  return weight, bias


class ColumnParallelLinear(_Module):
  def __init__(self, scope, input_size, output_size, hparams, bias=True, gather_output=True,
               #init_method=init.xavier_normal_, stride=1,
               init_method=None, stride=1,
               keep_master_weight_for_test=False, activation=None):
    super(ColumnParallelLinear, self).__init__()

    self.model_parallel = True

    # Keep input parameters
    self.hparams = hparams
    self.activation = activation
    self.input_size = input_size
    self.output_size = output_size
    self.gather_output = gather_output
    # Divide the weight matrix along the last dimension.
    world_size = get_model_parallel_world_size()
    self.output_size_per_partition = divide(output_size, world_size)
    dtype = hparams.dtype if hparams else tf.float32
    # with variable_scope(scope, dtype=dtype):
    #   self.weight = init_variable('w', [input_size, output_size], initializer=normal_initializer(dtype=dtype))
    #   self.bias = init_variable('b', [output_size], initializer=constant_initializer(dtype=dtype))
    self.weight, self.bias = init_weight_bias_parallel(dtype, scope, input_size, self.output_size_per_partition, axis=-1, use_bias=bias)

    # Parameters.
    # Note: torch.nn.functional.linear performs XA^T + b and as a result
    # we allocate the transpose.
    # self.weight = Parameter(torch.Tensor(self.output_size_per_partition,
    #                                      self.input_size))
      
    # self.weight.model_parallel = True
    # if bias:
    #     self.bias = Parameter(torch.Tensor(self.output_size_per_partition))
    #     self.bias.model_parallel = True
    #     # Always initialize bias to zero.
    #     with torch.no_grad():
    #         self.bias.zero_()
    # else:
    #     self.register_parameter('bias', None)

    # # Initialize weight.
    # self.master_weight = _initialize_affine_weight(
    #     self.weight, self.output_size, self.input_size,
    #     self.output_size_per_partition, 0, init_method,
    #     stride=stride, return_master_weight=keep_master_weight_for_test)

    # # Initialize weight.
    # self.master_weight = _initialize_affine_weight(
    #     self.weight, self.output_size, self.input_size,
    #     self.input_size_per_partition, 1, init_method,
    #     stride=stride, return_master_weight=keep_master_weight_for_test)


  def forward(self, input_):
      #lhs = tf.reshape(input_, [-1, self.input_size])
      #rhs = tf.reshape(self.w, [-1, self.output_size])
      # if False: # noticeable slowdown https://i.imgur.com/95VAycJ.png
      #   lhs_n = tf.split(lhs, 8, axis=1)
      #   rhs_n = tf.split(rhs, 8, axis=0)
      #   ops = []
      #   for i in range(8):
      #     with tf.device(get_core(i)):
      #       ops.append(tf.matmul(lhs_n[i], rhs_n[i]))
      #   W = tf.reduce_sum(ops, axis=0)
      # else:
      #   W = tf.matmul(lhs, rhs)
      # Set up backprop all-reduce.

      # #Set up backprop all-reduce.
      # import pdb; pdb.set_trace()
      # input_parallel = copy_to_model_parallel_region(input_)
      # # Matrix multiply.
      # output_parallel = F_linear(input_parallel, self.weight, self.bias)
      # if self.gather_output:
      #     # All-gather across the partitions.
      #     output = gather_from_model_parallel_region(output_parallel)
      # else:
      #     output = output_parallel

      # # Matrix multiply.
      # output_parallel = F_linear(input_parallel, self.weight, self.bias)
      # assert not self.gather_output
      # output = output_parallel

      x = input_
      hparams = self.hparams
      dtype = hparams.dtype if hparams else tf.float32
      with variable_scope('mlp', dtype=dtype):
        h0 = conv1d(x, 'c_fc', self.output_size, hparams=hparams)
        h1 = gelu(h0)
        import pdb; pdb.set_trace()
        return h1

      input_parallel = input_
      weight = self.weight[self.ctx.props['index']]
      bias = self.bias[self.ctx.props['index']]
      #import pdb; pdb.set_trace()
      output_parallel = tf.matmul(input_parallel, weight) + bias
      output = output_parallel
      if self.activation is not None:
        output = self.activation(output)

      return output



class RowParallelLinear(_Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """
    def __init__(self, scope, input_size, output_size, hparams, bias=True,
                 input_is_parallel=False,
                 # init_method=init.xavier_normal_, stride=1,
                 init_method=None, stride=1,
                 keep_master_weight_for_test=False):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.hparams = hparams
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)

        dtype = hparams.dtype if hparams else tf.float32
        #self.weight, self.bias = init_weight_bias_parallel(dtype, scope, self.input_size_per_partition, output_size, axis=-2, use_bias=bias)
        self.weight, self.bias = init_weight_bias_parallel(dtype, scope, self.input_size_per_partition, output_size, axis=-2, use_bias='full' if bias else False)

        # # Parameters.
        # # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # # we allocate the transpose.
        # self.weight = Parameter(torch.Tensor(self.output_size,
        #                                      self.input_size_per_partition))
        # self.weight.model_parallel = True
        # if bias:
        #     self.bias = Parameter(torch.Tensor(self.output_size))
        #     # Always initialize bias to zero.
        #     with torch.no_grad():
        #         self.bias.zero_()
        # else:
        #     self.register_parameter('bias', None)

        # # Initialize weight.
        # self.master_weight = _initialize_affine_weight(
        #     self.weight, self.output_size, self.input_size,
        #     self.input_size_per_partition, 1, init_method,
        #     stride=stride, return_master_weight=keep_master_weight_for_test)

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        #import pdb; pdb.set_trace()
        # Matrix multiply.
        #output_parallel = F_linear(input_parallel, self.weight)
        weight = self.weight[self.ctx.props['index']]
        output_parallel = tf.matmul(input_parallel, weight)
        if self.bias is not None:
          #bias = self.bias[self.ctx.props['index']]
          bias = self.bias
          output_parallel = output_parallel + bias
        # # All-reduce across all the partitions.
        # output_ = reduce_from_model_parallel_region(output_parallel)
        # if self.bias is not None:
        #     output = output_ + self.bias
        # else:
        #     output = output_
        # return output
        self.ctx.reduce = True
        return output_parallel


    
# def copy_to_model_parallel_region(input_):
  # assert not isinstance(input_, list)
  # pieces = tf.split(input_, get_model_parallel_world_size(), axis=-1)
  # ops = []
  # for core in range(get_model_parallel_world_size()):
    # with tf.device(device_for_tpu_core(core=core)):
    #   x = tf.identity(pieces[core])
    #   ops.append(x)
  # return ops


def F_linear(input_parallel, weight, bias=None):
  weight_parallel = weight
  bias_parallel = bias
  ops = []
  transpose = bias is None
  if not isinstance(input_parallel, (tuple, list)):
    #input_parallel = [input_parallel]
    input_parallel = tf.unstack(input_parallel, axis=0)
  for core, input_ in enumerate(input_parallel):
    with tf.device(device_for_tpu_core(core=core)):
      W = weight_parallel[core]
      op = tf.matmul(input_, W, transpose_a=transpose)
      if bias is not None:
        B = bias_parallel[core]
        op += B
      ops.append(op)
  return ops


class GPT2ParallelMLP(_Module):
  def __init__(self, hidden_size, hparams, init_method=None, output_layer_init_method=None):
    super(GPT2ParallelMLP, self).__init__()
    self.hidden_size = hidden_size
    self.hparams = hparams
    # Project to 4h.
    self.dense_h_to_4h = ColumnParallelLinear('mlp/c_fc', hidden_size, 4*hidden_size,
                                              gather_output=False,
                                              init_method=init_method,
                                              hparams=hparams,
                                              activation=None,
                                              #activation=gelu,
                                              )
    # Project back to h.
    self.dense_4h_to_h = RowParallelLinear(
        'mlp/c_proj',
        4*hidden_size,
        hidden_size,
        input_is_parallel=True,
        init_method=output_layer_init_method,
        hparams=hparams)
    
  def forward(self, hidden_states):
    return mlp_parallel(hidden_states, 'mlp', self.hidden_size*4, hparams=self.hparams)
    # def mlp1(x, scope, n_state, *, hparams):
    #     dtype = hparams.dtype if hparams else tf.float32
    #     with variable_scope(scope, dtype=dtype):
    #         nx = x.shape[-1].value
    #         h0 = conv1d(x, 'c_fc', n_state, hparams=hparams)
    #         h1 = gelu(h0)
    #         return h1
    #         h2 = conv1d(h1, 'c_proj', nx, hparams=hparams)
    #         h2 = dropout(h2, hparams.res_dropout)
    #         if api.should_break: pdb.set_trace()
    #         return h2
    # #intermediate_parallel = [mlp1(hidden_states, 'mlp', self.hidden_size*4, hparams=self.hparams)]

    # #return mlp(hidden_states, 'mlp', self.hidden_size*4, hparams=self.hparams)

    # # [b, s, 4hp]
    # intermediate_parallel = self.dense_h_to_4h(hidden_states)
    # #intermediate_parallel = [gelu(x) for x in intermediate_parallel]
    # #import pdb; pdb.set_trace()

    # # [b, s, h]
    # output = self.dense_4h_to_h(intermediate_parallel)
    # #output = self.dropout(output)
    # output = dropout(output, self.hparams.res_dropout)
    # #import pdb; pdb.set_trace()
    # return output
    
    

