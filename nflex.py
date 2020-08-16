import numpy as np
import mel_common

def val(a):
  #return np.array(a)
  #return mel_common.cast(a)
  return mel_common.const(a)

def shape(a):
  #return list(val(a).shape)
  return mel_common.shaped(a)

def rank(a):
  return len(shape(a))

def transposed(a, axis=-2):
  a = val(a)
  indices = list(range(rank(a))) # e.g. [0, 1, 2, 3]
  indices[axis], indices[-1] = indices[-1], indices[axis] # e.g. [0, 1, 3, 2]
  return a.transpose(indices)

def grow_along_last_axis(a, n, initial_value=0):
  a = val(a)
  #value = np.zeros(shape(a)[0:-1] + [n], dtype=a.dtype)
  value = mel_common.zeros(shape(a)[0:-1] + [n], dtype=a.dtype)
  value.fill(initial_value)
  #a1 = np.append(a, value, axis=-1)
  #a1 = np.concatenate([a, value], axis=-1)
  #return a1.astype(a.dtype)
  return mel_common.append(a, value, axis=-1)

def grow_along_axis(a, n, axis=-1, initial_value=0):
  if axis == -1 or axis == shape(a)[-1]:
    return grow_along_last_axis(a, n, initial_value=initial_value)
  a = transposed(a, axis=axis)
  a = grow_along_last_axis(a, n, initial_value=initial_value)
  a = transposed(a, axis=axis)
  return a

from functools import partial

grow_along_rows = partial(grow_along_axis, axis=-1)
grow_along_cols = partial(grow_along_axis, axis=-2)

def padding(size, alignment):
  assert alignment > 0
  return -size % alignment

def aligned(size, alignment):
  return size + padding(size, alignment)

def pad_along_axis(a, n, axis=-1, initial_value=0):
  a = val(a)
  padding_ = padding(shape(a)[axis], n)
  return grow_along_axis(a, padding_, axis=axis, initial_value=initial_value)

pad_along_rows = partial(pad_along_axis, axis=-1)
pad_along_cols = partial(pad_along_axis, axis=-2)

