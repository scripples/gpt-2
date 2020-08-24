import tensorflow as tf
import numpy as np
from glob import glob
import os
import sys
import re
from tensorflow.python import pywrap_tensorflow
import tqdm
import h5py
import shutil
import tempfile
import math
from pprint import pprint as pp

from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver
from tensorflow.core.protobuf import config_pb2

def get_tpu_addr(tpu_name=None):
    # Get the TPU's location
    if tpu_name is not None:
      return TPUClusterResolver(tpu_name).get_master()
    if 'COLAB_TPU_ADDR' in os.environ:
      return TPUClusterResolver().get_master()
    elif 'TPU_NAME' in os.environ:
      return TPUClusterResolver(os.environ['TPU_NAME']).get_master()

def get_session_target(target='auto'):
    if target == 'auto':
      target = get_tpu_addr()
    elif target is not None:
      target = get_tpu_addr(target)
    if target is not None:
      print("Using TPU %s" % target)
    return target

def get_tpu_config():
  session_config = config_pb2.ConfigProto(allow_soft_placement=True, isolate_session_state=False)
  master = None
  res = None
  cluster_spec = None
  cluster_def = None
  job_names = None
  master_job = 'worker'
  try:
    if 'TPU_NAME' in os.environ or 'COLAB_TPU_ADDR' in os.environ:
      res = TPUClusterResolver(os.environ.get('TPU_NAME', os.environ.get('COLAB_TPU_ADDR', None)))
      master = res.get_master()
      cluster_spec = res.cluster_spec()
      if cluster_spec:
        cluster_def = cluster_spec.as_cluster_def()
        session_config.cluster_def.CopyFrom(cluster_def)
        job_names = set([job.name for job in cluster_def.job])
        assert len(job_names) == 1
        master_job = cluster_def.job[0].name
    elif 'TPU_IP' in os.environ:
      master = os.environ['TPU_IP'].replace('grpc://', '')
      if ':' not in master:
        master = master + ':8470'
      master = 'grpc://' + master
  except:
    import traceback
    traceback.print_exc()
  return session_config

def get_session_config(config='auto'):
    if config != 'auto':
      return config
    elif 'TPU_NAME' in os.environ or 'COLAB_TPU_ADDR' in os.environ:
      #return None # TODO: create a session config with a timeout of 10min?
      #import pdb; pdb.set_trace()
      return get_tpu_config()
    elif 'NUM_CORES' in os.environ:
      n_inter = int(os.environ['NUM_CORES'])
      n_intra = int(os.environ['NUM_CORES'])
      n_devs = int(os.environ['NUM_CORES'])
      print('Using %d CPU cores' % n_devs)
      return tf.ConfigProto(
              device_count={ "CPU": n_devs },
              inter_op_parallelism_threads=n_inter,
              intra_op_parallelism_threads=n_intra,
        ) # https://github.com/tensorflow/tensorflow/issues/22619#issuecomment-426355744
    else:
      return None

def get_current_device(session):
  # is there a nicer way?
  try:
    return session.graph._device_function_stack.peek_top_obj().name
  except IndexError:
    return None

def pretty(s, n=64):
  if len(s) > n:
    return s[0:n - len('...')] + '...'
  return s

def prettify(l, n=64):
  if isinstance(l, (tuple, list)):
    return [prettify(x, n=n) for x in l]
  elif isinstance(l, dict):
    return {prettify(k, n=n): prettify(v, n=n) for k, v in l.items()}
  elif isinstance(l, str):
    return pretty(l, n=n)
  elif isinstance(l, np.ndarray):
    #return pretty('{!r}: {}'.format(l.shape, str(l)), n=n)
    return [{'shape': l.shape}, pretty(str(l), n=n)]
  else:
    return pretty(str(l), n=n)

def pretty_obj(x, n=64):
  return repr(prettify(x, n=n))

def p(x, n=64):
  pp(prettify(x, n=n))
  return x

class Session(tf.Session):
  def __init__(self, target='auto', graph=None, config='auto'):
    target = get_session_target(target)
    config = get_session_config(config)
    super(Session, self).__init__(target, graph=graph, config=config)
    self.target = target
    self.config = config
    self._tflex_cached_devices = self.list_devices()
    pp(self._tflex_cached_devices)

  def run(self, *args, **kws):
    now = time.time()
    device = get_current_device(self)
    print('Session.run device=%s %s %s' % (repr(device), pretty_obj(args), pretty_obj(kws)))
    result = super(Session, self).run(*args, **kws)
    elapsed = time.time() - now
    print('Session.run device=%s finished in %.2fsec: %s %s' % (repr(device), elapsed, pretty_obj(args), pretty_obj(kws)))
    return result

def split_by_params(vs, n=200e6, f=None):
  if f is None:
    f = lambda x: np.prod(x.shape.as_list())
  i = 0
  xs = []
  for variable in vs:
    xs.append(variable)
    count = f(variable)
    i += count
    if i >= n:
      yield xs
      xs = []
      i = 0
  yield xs

def latest_checkpoint(checkpoint_dir, latest_filename=None):
  if checkpoint_dir.startswith('gs://'):
    return tf.train.latest_checkpoint(checkpoint_dir)
  paths = [x for x in glob(os.path.join(checkpoint_dir, 'model-*.*')) if not x.endswith(".tmp")]
  ctrs = np.array([[int(y) for y in re.findall(r'model-([0-9]+)(?:-[0-9]+)?[.](?:npy|hdf5)', x)] for x in paths]).flatten()
  if len(ctrs) <= 0:
    ckpt = tf.train.latest_checkpoint(checkpoint_dir, latest_filename=latest_filename)
    return ckpt
  ctr = ctrs.max()
  return os.path.join(checkpoint_dir, 'model-{}').format(ctr)

def truncate_value(variable, value, reshape=True):
  if not reshape:
    return value
  shape = variable.shape.as_list()
  params = np.prod(shape)
  params2 = np.prod(value.shape)
  if params == params2:
    return value
  if params2 > params:
    print('Truncating {} from shape {} to shape {}'.format(variable.name, value.shape, shape))
    sys.stdout.flush()
    value = np.array(value)
    value = value.reshape([-1])
    value = value[0:params]
    value = value.reshape(shape)
  else:
    print('Expanding {} from shape {} to shape {}'.format(variable.name, value.shape, shape))
    sys.stdout.flush()
    value = np.array(value)
    value = value.reshape([-1])
    n = math.ceil(params / params2)
    value = np.tile(value, n)
    value = value.reshape(shape)
  return value

def grab_values(variables, reader, reshape=False):
  for variable in variables:
    name = variable.name.split(':')[0]
    value = reader.get_tensor(name)
    value = truncate_value(variable, value, reshape=reshape)
    yield variable, value

def assign_values(variables, values, session=None):
  session = session or tf.get_default_session()
  ops = [x.initializer for x in variables]
  vals = dict([(x.initializer.inputs[1], value) for x, value in zip(variables, values)])
  #for x, (k, v) in zip(variables, vals.items()):
  #  print(x.name, x.shape.as_list(), k, v.shape)
  session.run(ops, vals)

def load_snapshot(ckpt, session=None, var_list=None, reshape=False):
  session = session or tf.get_default_session()
  reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
  vs = var_list or tf.trainable_variables()
  for variables in tqdm.tqdm(list(split_by_params(vs))):
    values = [value for variable, value in grab_values(variables, reader, reshape=reshape)]
    assign_values(variables, values, session=session)

def get_variable(name, var_list=None):
  name, num = name.split(':') if ':' in name else (name, '0')
  num = int(num)
  name = os.path.join(tf.get_variable_scope().name, name)
  vs = var_list or tf.trainable_variables()
  for x in vs:
      if x.name.startswith(name + ':%d' % num):
          return x

def load_weights(ckpt, session=None, var_list=None, reshape=False):
  session = session or tf.get_default_session()
  vs = var_list or tf.trainable_variables()
  files = list(sorted(glob(ckpt + '-*.npy')))
  for out in tqdm.tqdm(files):
    for name, value in np.load(out, allow_pickle=True):
      variable = get_variable(name)
      if variable is None:
        print('Warning: variable %s not loaded' % name)
      else:
        value = truncate_value(variable, value, reshape=reshape)
        variable.load(value, session)

def load_variables(ckpt, session=None, var_list=None, reshape=False):
  session = session or tf.get_default_session()
  vs = var_list or tf.trainable_variables()
  with h5py.File(ckpt, "r") as f:
    for variables in tqdm.tqdm(list(split_by_params(vs))):
      values = [truncate_value(x, f[x.name], reshape=reshape)  for x in variables]
      assign_values(variables, values, session=session)

def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass

def save_variables(ckpt, session=None, var_list=None):
    session = session or tf.get_default_session()
    vs = var_list or tf.trainable_variables()
    maketree(os.path.dirname(ckpt))
    fname = ckpt+'.tmp'
    with h5py.File(fname, "w") as f:
      for variables in tqdm.tqdm(list(split_by_params(vs))):
        values = session.run(variables)
        for value, variable in zip(values, variables):
          name = variable.name
          shape = variable.shape.as_list()
          dtype = variable.dtype
          dset = f.create_dataset(name, shape, dtype=np.float32)
          dset[:] = value
    print('Writing snapshot %s' % ckpt)
    os.rename(ckpt+'.tmp', ckpt)

class Saver(object):
  def __init__(
    self,
    var_list=None,
    reshape=False,
    sharded=False,
    max_to_keep=5,
    keep_checkpoint_every_n_hours=10000.0,
    name=None,
    restore_sequentially=False,
    saver_def=None,
    builder=None,
    defer_build=False,
    allow_empty=False,
    write_version=tf.train.SaverDef.V2,
    pad_step_number=False,
    save_relative_paths=False,
    filename=None):
    self.var_list = var_list
    self.reshape = reshape
    self.sharded = sharded
    self.max_to_keep = max_to_keep
    self.keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
    self.name = name
    self.restore_sequentially = restore_sequentially
    self.saver_def = saver_def
    self.builder = builder
    self.defer_build = defer_build
    self.allow_empty = allow_empty
    self.write_version = write_version
    self.pad_step_number = pad_step_number
    self.save_relative_paths = save_relative_paths
    self.filename = filename
    self.checkpoints = []

  def restore(self, sess, save_path):
    if save_path.startswith('gs://'):
      saver = tf.train.Saver(var_list=self.var_list)
      saver.restore(sess, save_path)
    elif '.ckpt' in os.path.basename(save_path):
      load_snapshot(save_path, session=sess, var_list=self.var_list, reshape=self.reshape)
    elif save_path.endswith('.hdf5'):
      load_variables(save_path, session=sess, var_list=self.var_list, reshape=self.reshape)
    elif os.path.exists(save_path + '.npy') or os.path.exists(save_path + '-0.npy'):
      load_weights(save_path, session=sess, var_list=self.var_list, reshape=self.reshape)
    elif os.path.exists(save_path + '.hdf5'):
      load_variables(save_path + '.hdf5', session=sess, var_list=self.var_list, reshape=self.reshape)
    else:
      raise Exception("Can't load checkpoint %s" % save_path)

  def save(self,
        sess,
        save_path,
        global_step=None,
        latest_filename=None,
        meta_graph_suffix="meta",
        write_meta_graph=True,
        write_state=True,
        strip_default_attrs=False,
        save_debug_info=False):
    if global_step is not None:
      name = '%s-%d.hdf5' % (save_path, global_step)
    else:
      name = '%s.hdf5' % save_path
    save_variables(name, session=sess, var_list=self.var_list)
    self.checkpoints.append(name)
    if self.max_to_keep > 0:
      while len(self.checkpoints) > self.max_to_keep:
        fname = self.checkpoints[0]
        if fname != name:
          print('Truncating %s' % fname)
          try:
            with open(fname, "wb") as f:
              pass
          except:
            print('Failed to truncate %s' % fname)
        self.checkpoints = self.checkpoints[1:]


# https://tensorflow.github.io/lingvo/_modules/lingvo/core/bfloat16_variables.html

from tensorflow.python.training import saver as tf_saver


class Bfloat16VariableSaveable(tf_saver.BaseSaverBuilder.SaveableObject):
  """Saveable that loads Variables as bfloat16."""
  def __init__(self, var, orig_dtype, slice_spec, name):
    # TODO(rohananil): Investigate if we can avoid using a callable, instead
    # change the saveable api to make use of dtype passed in.
    def _make_callable_var():
      return var
    spec = tf_saver.BaseSaverBuilder.SaveSpec(
        _make_callable_var,
        slice_spec,
        name,
        dtype=orig_dtype,
        device=var.device)
    super().__init__(var, [spec], name)
  def restore(self, restored_tensors, restored_shapes):
      restored_tensor = restored_tensors[0]
      if restored_shapes is not None:
        restored_tensor = tf.reshape(restored_tensor, restored_shapes[0])
      return tf.assign(
          self.op,
          tf.cast(restored_tensor, tf.bfloat16),
          validate_shape=restored_shapes is None and
          self.op.get_shape().is_fully_defined())


def get_saver_spec_for_variables_with_bf16_overrides(variables_to_restore):
  """Returns a dictionary containing overrides to load variables as bf16.

  Args:
    variables_to_restore: A mapping from variable to name (on checkpoint) to the
      Variable object.

  Returns:
    A saver dictionary which can be used to load from checkpoints.
  """
  saver_dict = {}
  for var_name, v in variables_to_restore.items():
    if v.dtype == tf.bfloat16:
      # TODO(rohananil): Add support for PartitionedVariables if there is
      # demand.
      savable = Bfloat16VariableSaveable(v, tf.float32, '', var_name)
      saver_dict[var_name] = savable
    else:
      saver_dict[var_name] = v
  return saver_dict


def get_bf16_var_list(var_list):
  if isinstance(var_list, list):
    var_list = dict([(x.name.split(':')[0], x) for x in var_list])
  return get_saver_spec_for_variables_with_bf16_overrides(var_list)


class Commands(object):
  def __init__(self, path='commands'):
    self.path = path
    self.commands = []
    self.args = []
    self.keys = {}
    self.frozen = False

  def has(self, name, **keys):
    if 'action' in keys:
      action = keys.pop('action')
      for name1, action1 in self.commands:
        if name == name1 and action1 == action:
          return True
    else:
      for name1, action1 in self.commands:
        if name == name1:
          return True
    return False

  def add(self, name, action=None):
    if not self.has(name=name, action=action):
      self.commands.append((name, action))
      full = self.full_path(name)
      maketree(full)

  def full_path(self, name):
    return os.path.join(self.path, name)

  def check(self, *args, **keys):
    if not self.frozen:
      heartbeat()
    ops = []
    seen = set()
    for name, action in self.commands:
      full = self.full_path(name)
      if not os.path.isdir(full):
        if name not in seen:
          seen.add(name)
          ops.append(name)
    for op in ops:
      self.run(op, *args, **keys)
    return ops

  def run(self, op):
    ran = False
    for name, action in self.commands:
      if name == op:
        print('Running command', name, action)
        if not ran:
          full = self.full_path(op)
          maketree(full)
          ran = True
        if action:
          action()
    if not ran:
      raise Exception('Commands.execute failed: no such command: {}'.format(op))
  
  def run_with_args(self, op, *args, **keys):
    with CommandArgs(*args, **keys):
      return self.run(op)

commander = None

def commands(**keys):
  global commander
  if commander is None:
    commander = Commands()
  cmds = keys.pop('commands') if 'commands' in keys else None
  if cmds is not None:
    for cmd in cmds:
      action = None
      if isinstance(cmd, str):
        name = cmd
      elif len(cmd) >= 2:
        name, action = cmd
      elif len(cmd) >= 1:
        name = cmd[0]
      else:
        continue
      commander.add(name=name, action=action)
  return commander

class CommandArgs(object):
  def __init__(self, *args, **keys):
    self.args = list(args)
    self.keys = keys.copy()
    self.cmdr = commands()

  def __enter__(self):
    self.args_prev = self.cmdr.args
    self.keys_prev = self.cmdr.keys
    self.cmdr.args = self.args
    self.cmdr.keys = self.keys

  def __exit__(self, *excinfo):
    self.cmdr.args = self.args_prev
    self.cmdr.keys = self.keys_prev

def check_commands():
  cmdr = commands()
  return cmdr.check()

def check_commands_with_args(*args, **keys):
  cmdr = commands()
  with CommandArgs(*args, **keys):
    return cmdr.check()

def add_command(name, action=None, **keys):
  cmdr = commands()
  return cmdr.add(name=name, action=action)

def register_command(*args, **keys):
  fn = args[0]
  if isinstance(fn, str):
    add_command(fn)
  else:
    name = fn.__qualname__
    name = name.replace('.<locals>.', '_command_')
    if name.endswith('_command_save'):
      name = 'save'
    name = name.replace('___', '/')
    action = fn
    print(name, action)
    add_command(name, action)
  return fn

def has_command(name):
  cmdr = commands()
  return cmdr.has(name)

def run_command(command_name):
  cmdr = commands()
  return cmdr.run(command_name)

def run_command_with_args(command_name, *args, **keys):
  cmdr = commands()
  return cmdr.run_with_args(command_name, *args, **keys)

def command_arg(x, unset=None):
  cmdr = commands()
  if isinstance(x, int):
    try:
      return cmdr.args[x]
    except:
      return unset
  else:
    if x in cmdr.keys:
      return cmdr.keys[x]
    return unset

def command_args():
  cmdr = commands()
  return cmdr.args, cmdr.keys

@register_command
def attach_debugger():
  import pdb
  pdb.set_trace()

from pprint import pprint

@register_command
def print_status():
  args, props = command_args()
  for k, v in enumerate(args):
    pprint(v)
  for k, v in props.items():
    pprint({k: v})


#
# return current UTC timestamp.
#
def utc():
    from datetime import datetime
    d = datetime.utcnow()
    import calendar
    return calendar.timegm(d.utctimetuple())

def heartbeat():
  pongfile=os.environ['PONG'] if 'PONG' in os.environ else 'pong.txt'
  with open(pongfile, "a+") as f:
    nonce = os.urandom(8).hex()
    now=utc()
    out="pid{}_time{}_nonce{}\n".format(os.getpid(), now, nonce)
    #print("PONG! Writing {} to {}".format(out, pongfile))
    f.write(out)
    f.flush()

import time

@register_command
def freeze_forever():
  cmdr = commands()
  if cmdr.frozen:
    print("Already frozen.")
    return
  prev = cmdr.frozen
  cmdr.frozen = True
  print('Simulating a freeze; going into an infinite loop:')
  prev=time.time()
  try:
    while not should_quit():
      elapsed=time.time() - prev
      print('Frozen for {}s'.format(elapsed))
      time.sleep(1)
      check_commands()
  finally:
    cmdr.frozen = prev

_quit = False

import sys

@register_command
def quit():
  global _quit
  if _quit:
    print("Failed to quit; running sys.exit(1)")
    sys.exit(1)
  else:
    print("Quitting...")
    _quit = True

def should_quit():
  return _quit

@register_command
def save_and_quit():
  global _quit
  if has_command('save'):
    print("Saving...")
    run_command('save')
  quit()

