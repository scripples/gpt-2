import functools

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.parallel_for.pfor import PFor
from tensorflow.python.ops.parallel_for.pfor import PForConfig
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export

# import tflex_pfor; from importlib import reload; reload(tflex_pfor);
#
# zz = tflex_pfor.for_loop(lambda i, *ta: [prn(tf.ones([42, i]) + tf.cond(tf.less_equal(i, 0), lambda: tf.constant(0.0), lambda: tf.cast(ta[-1].read(tf.maximum(0, i-1))[0][1], tf.float32)), i), tf.stack([i, i*i])], [tf.float32, tf.int32], 8, shape_invariants=[[1, 42, None], [1, 2]])
#
# r(zz[0].read(6))


# zz = tflex_pfor.for_loop(lambda i, *ta: [tf.ones([42, i]) + (0 if ta[-1] is None else tf.cast(ta[-1].read(tf.maximum(0, i-3))[0][1], tf.float32)), tf.stack([i, i*i])], [tf.float32, tf.int32], 8, shape_invariants=[[1, 42, None], [1, 2]])

# r(zz[0].read(4))


def for_loop(loop_fn, loop_fn_dtypes, iters, parallel_iterations=None, shape_invariants=None):
  """Runs `loop_fn` `iters` times and stacks the outputs.


  Runs `loop_fn` `iters` times, with input values from 0 to `iters - 1`, and
  stacks corresponding outputs of the different runs.

  Args:
    loop_fn: A function that takes an int32 scalar tf.Tensor object representing
      the iteration number, and returns a possibly nested structure of tensor
      objects. The shape of these outputs should not depend on the input.
    loop_fn_dtypes: dtypes for the outputs of loop_fn.
    iters: Number of iterations for which to run loop_fn.
    parallel_iterations: The number of iterations that can be dispatched in
      parallel. This knob can be used to control the total memory usage.

  Returns:
    Returns a nested structure of stacked output tensor objects with the same
    nested structure as the output of `loop_fn`.
  """

  flat_loop_fn_dtypes = nest.flatten(loop_fn_dtypes)
  is_none_list = []

  if shape_invariants is None:
    shape_invariants = [None] * len(flat_loop_fn_dtypes)

  def while_body(i, *ta_list, ta_first=None):
    """Body of while loop."""
    fn_output = nest.flatten(loop_fn(i, *ta_list))
    if len(fn_output) != len(flat_loop_fn_dtypes):
      raise ValueError(
          "Number of expected outputs, %d, does not match the number of "
          "actual outputs, %d, from loop_fn" % (len(flat_loop_fn_dtypes),
                                                len(fn_output)))
    outputs = []
    del is_none_list[:]
    is_none_list.extend([x is None for x in fn_output])
    xs = list(zip(fn_output, ta_first if ta_first is not None else ta_list))
    #import pdb; pdb.set_trace()
    for out, ta in xs:
      # TODO(agarwal): support returning Operation objects from loop_fn.
      if out is not None:
        # out may be a ref tensor, wrap it in identity to get a non-ref tensor.
        ta = ta.write(i, array_ops.expand_dims(out, 0))
      outputs.append(ta)
    return tuple([i + 1] + outputs)

  if parallel_iterations is not None:
    extra_args = {"parallel_iterations": parallel_iterations}
  else:
    extra_args = {}
  ta = [tensor_array_ops.TensorArray(dtype.base_dtype, iters, infer_shape=False, dynamic_size=True, element_shape=shape_invariants[i], clear_after_read=False)
             for i, dtype in enumerate(flat_loop_fn_dtypes)]
  outputs = while_body(tf.constant(0), *[None for _ in range(len(flat_loop_fn_dtypes))], ta_first=ta)
  #import pdb; pdb.set_trace()
  out0 = control_flow_ops.while_loop(
      lambda i, *ta: i < iters,
      while_body,
      #[0] + ta,
      outputs,
      **extra_args)
  ta_list = out0[1:]

  if True:
    output = [None if is_none else ta
              for ta, is_none in zip(ta_list, is_none_list)]
    return output
  else:
    # TODO(rachelim): enable this for sparse tensors

    output = [None if is_none else ta.concat()
              for ta, is_none in zip(ta_list, is_none_list)]
    assert len(output) in (0, len(flat_loop_fn_dtypes))
    if not output:
      # This may happen for the case where iters == 0.
      return None
    else:
      return nest.pack_sequence_as(loop_fn_dtypes, output)

