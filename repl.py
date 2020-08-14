batch_size = 1
#factor = int(os.environ['NUM_CORES'])
#factor = 2; model_name='117M'
#factor = 4; model_name='774M' # failed?
#factor = 2; model_name='345M'; batch_size = 4 # OOM: 11.89GB
#factor = 2; model_name='345M'; batch_size = 2 # Ran out of memory in memory space hbm. Used 9.53G of 8.00G hbm. Exceeded hbm capacity by 1.53G.

factor = 4; model_name='345M'; batch_size = 2 # Ran out of memory in memory space hbm. Used 9.62G of 8.00G hbm. Exceeded hbm capacity by 1.62G.
# https://gist.githubusercontent.com/shawwn/50ba2e6f6fcffd1e768b723af9ef7bab/raw/7aaed9673190036a6a5e50bb68fc322718a6deda/factor4_model345M_batchsize2_OOM.txt

#factor = 8; model_name='1558M'
import tflex_tpu_device_assignment; import tflex_tpu_topology; topology = tflex_tpu_topology.get_topology(res); dev = tflex_tpu_device_assignment.spatial_partition(topology, factor)

os.environ['NUM_CORES'] = str(factor)

#zz = tpu_ops.shard(alloc_op(), outputs_from_all_shards=True, num_shards=dev.num_replicas, inputs=[], device_assignment=dev)

import model, sample, encoder

import json
import time

from natsort import natsorted
from pprint import pprint as pp

from tensorflow.contrib import tpu

from tensorflow.contrib.tpu.python.tpu import tpu_function

args = model.Context()
args.only_train_transformer_layers = False
args.sample_length = 48
args.batch_size = batch_size
args.temperature = 0.8
args.top_k = 40
args.top_p = 0.0
args.learning_rate = 0.00002
args.iterations = 20


restore_from=None
seed=None
nsamples=0
batch_size=batch_size
length=None
temperature=0.8
top_k=40
top_p=0.0


enc = encoder.get_encoder(model_name)
hparams = model.default_hparams()
with open(os.path.join('models', model_name, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))


#context = tf.placeholder(tf.int32, [args.batch_size, None])
context = tf.placeholder(tf.int32, [dev.num_replicas * args.batch_size, None])
#context_in = randomize(context, hparams, args.noise)
#context_in = context
#output = model.model(hparams=hparams, X=context_in)

def output_op(context):
  output = model.model(hparams=hparams, X=context)
  print(output)
  #result = output['logits']
  result = list(output.values())
  #print(result)
  return result

# op = tpu_ops.shard(output_op, outputs_from_all_shards=True, num_shards=dev.num_replicas, inputs=[context], device_assignment=dev)


lr = tf.Variable(args.learning_rate, collections=['local_variables'], dtype=tf.float32, shape=())

sess.run(lr.initializer)

#tokens = tf.Variable(tf.zeros([10000]), dtype=tf.uint16, collections=['local_variables'], name='tokens')


# Sample 1024(+1) tokens from the stitched together text
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


# def train_op(context):
#   output = model.model(hparams=hparams, X=context)
#   loss = tf.reduce_mean(
#       tf.nn.sparse_softmax_cross_entropy_with_logits(
#         labels=context[:, 1:], logits=output['logits'][:, :-1]))
#   opt = tf.train.AdamOptimizer(learning_rate=lr)
#   opt = tf.tpu.CrossShardOptimizer(opt)
#   all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
#   train_vars = [v for v in all_vars if '/h' in v.name] if args.only_train_transformer_layers else all_vars
#   opt_grads = tf.gradients(loss, train_vars)
#   opt_grads = list(zip(opt_grads, train_vars))
#   global sorted_grads
#   sorted_grads = list(natsorted(opt_grads, key=lambda gv: gv[0].name))
#   #import pdb; pdb.set_trace()
#   pp(sorted_grads)
#   gs = tf.train.get_or_create_global_step()
#   opt_apply = opt.apply_gradients(opt_grads, global_step=gs)
#   with tf.control_dependencies([opt_apply]):
#     loss = tf.identity(loss)
#   result = output['logits']
#   return [loss]

# #train = tpu_ops.shard(train_op, outputs_from_all_shards=True, num_shards=dev.num_replicas, inputs=[context], device_assignment=dev)
# (compile_train, train) = tpu_ops.split_compile_and_shard(train_op, outputs_from_all_shards=True, num_shards=dev.num_replicas, inputs=[context], device_assignment=dev)



import tputil

toks = tputil.tf_file_tok16('gs://tpu-usc1/datasets/dota2-ftfy.txt.tok16')
#toks = tputil.tf_file_tok16('gs://tpu-usc1/datasets/openwebtext-3b.tok16')

sess.run(toks.initializer)

# toks2 = tputil.tf_file_tok16('gs://tpu-usc1/datasets/openwebtext.tok16')
# import time; now = time.time(); results = sess.run(toks2.initializer) ; elapsed = time.time() - now; results, elapsed



def device_for_host(task=0, cpu=0, job="worker"):
  #return "/job:%s/task:%d/device:CPU:%d" % (job, task, cpu)
  #return "/job:%s/replica:0/task:%d/device:CPU:%d" % (job, task, cpu)
  return dev.host_device(replica=task, logical_core=cpu, job=job)


# def sample_tokens():
#   def on_tpu_cpu():
#     with tf.device(device_for_host()):
#       context, context2 = sample_text(toks, amount=hparams.n_ctx, batch_size=args.batch_size)
#       return tf.identity(context, name="sampled_tokens")
#   return tpu_ops.outside_compilation(on_tpu_cpu) 


def sample_tokens(tokens_var=None):
  if tokens_var is None:
    tokens_var = toks
  context, context2 = sample_text(tokens_var, amount=hparams.n_ctx, batch_size=args.batch_size)
  return tf.identity(context, name="sampled_tokens")


def tpu_infeed(tokens_var=None):
  ops = []
  for host_device, device_ordinal, logical_core, replica in tflex_tpu_device_assignment.device_mapping(dev):
    if logical_core == 0:
      print(replica, host_device, device_ordinal)
      with tf.device(host_device):
        #x = tf.constant(replica, tf.float32)
        x = sample_tokens(tokens_var=tokens_var)
        ops.append(tpu_ops.tpu_ops.infeed_enqueue(x, shape=x.shape, device_ordinal=device_ordinal));
  with tf.control_dependencies(ops):
    return tf.no_op(name='infeed')


from functools import partial


def tpu_infeed_batch(tokens_var=None):
  return tpu.repeat(iterations_var, partial(tpu_infeed, tokens_var=tokens_var))


def dequeue_tokens():
  with tf.device(model.device_for_tpu_core()):
    tokens = tpu_ops.tpu_ops.infeed_dequeue(tf.int32, shape=(args.batch_size, hparams.n_ctx))
  return tokens


def op():
  #with tf.device(dev.tpu_device(replica=0, job='worker')):
  with tf.device(model.device_for_tpu_core()):
    tokens = tpu_ops.tpu_ops.infeed_dequeue(tf.int32, shape=(args.batch_size, hparams.n_ctx))
    return [tokens]


def deq():
  zz = tpu_ops.shard(op, outputs_from_all_shards=True, num_shards=dev.num_replicas, inputs=[], device_assignment=dev); qq = sess.run(zz); print(qq)

# #with tf.device(dev.host_device(replica=0, job='worker')):
#zz = tpu_ops.shard(op, outputs_from_all_shards=True, num_shards=dev.num_replicas, inputs=[], device_assignment=dev); qq = sess.run(zz); print(qq)


def tpu_step(initial_loss):
  del initial_loss
  with tf.variable_scope('', reuse=tf.AUTO_REUSE, use_resource=True):
    #context, context2 = sample_text(toks, amount=hparams.n_ctx, batch_size=args.batch_size)
    #context = sample_tokens()
    context = dequeue_tokens()
    output = model.model(hparams=hparams, X=context)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=context[:, 1:], logits=output['logits'][:, :-1]))
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    opt = tf.tpu.CrossShardOptimizer(opt)
    all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
    train_vars = [v for v in all_vars if '/h' in v.name] if args.only_train_transformer_layers else all_vars
    opt_grads = tf.gradients(loss, train_vars)
    opt_grads = list(zip(opt_grads, train_vars))
    global sorted_grads
    sorted_grads = list(natsorted(opt_grads, key=lambda gv: gv[0].name))
    #import pdb; pdb.set_trace()
    pp(sorted_grads)
    gs = tf.train.get_or_create_global_step()
    opt_apply = opt.apply_gradients(opt_grads, global_step=gs)
    with tf.device(model.device_for_tpu_core()):
      with tf.control_dependencies([opt_apply]):
        return tf.identity(loss, name="tpu_loss_op")
  
  # with tf.control_dependencies([opt_apply]):
  #   loss = tf.identity(loss)
  # result = output['logits']
  # return loss



_INITIAL_LOSS = 1e7

iterations_var = tf.Variable(args.iterations, collections=['local_variables'], dtype=tf.int32, shape=())

sess.run(iterations_var.initializer)

infeed_batch = tpu_infeed_batch(toks)
sess.run(infeed_batch)

# infeed_batch2 = tpu_infeed_batch(toks2)
# sess.run(infeed_batch2)

#infeed = tpu_infeed()
#sess.run(infeed)


@tpu_function.on_device_training_loop
def tpu_loop():
  return tpu.repeat(iterations_var, tpu_step, [_INITIAL_LOSS])


#train = tpu_ops.shard(train_op, outputs_from_all_shards=True, num_shards=dev.num_replicas, inputs=[context], device_assignment=dev)
#(compile_train, train) = tpu_ops.split_compile_and_shard(train_op, outputs_from_all_shards=True, num_shards=dev.num_replicas, inputs=[context], device_assignment=dev)
(compile_train2, train2) = tpu_ops.split_compile_and_shard(tpu_loop, outputs_from_all_shards=True, num_shards=dev.num_replicas, inputs=[], device_assignment=dev)

sess.run(tf.global_variables_initializer())
#sess.run(tf.local_variables_initializer())
sess.run([x.initializer for x in sess.graph.get_collection_ref('variables')])

#sess.run(compile_train2)
import time; now = time.time(); results = sess.run(compile_train2) ; elapsed = time.time() - now; results, elapsed

all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
train_vars = [v for v in all_vars if '/h' in v.name] if args.only_train_transformer_layers else all_vars

saver = tf.train.Saver(var_list=train_vars)

import time
import threading

os.environ['MODEL_DIR'] = 'gs://tpu-usc1/runs/gpt-2/gptrun10parallel117m'

if 'savers' not in globals():
  savers = []

def train_forever(save_every_minutes=30.0, model_dir=None):
  if model_dir is None:
    model_dir = os.environ['MODEL_DIR']
  gs = tf.train.get_global_step()
  start = time.time()
  #next_save = start + save_every_minutes * 60.0
  next_save = start
  while True:
    now = time.time()
    if now >= next_save:
      global_step = sess.run(gs)
      def thunk():
        path = os.path.join(model_dir, 'model')
        print('Saving to {}-{}'.format(path, global_step))
        saving = time.time()
        saver.save(sess, path, global_step=global_step)
        elapsed = time.time() - saving
        print('Saved to {}-{} in {:.2f}sec'.format(path, global_step, elapsed))
      thread = threading.Thread(target=thunk, daemon=True)
      thread.start()
      savers.append(thread)
      next_save = now + save_every_minutes * 60.0
    sess.run(infeed_batch2);
    now = time.time(); results = sess.run(train2) ; elapsed = time.time() - now;
    print(results, elapsed, sess.run(tf.train.get_global_step()))

iterations_var.load(1)

import time; now = time.time(); results = sess.run(train2) ; elapsed = time.time() - now; results, elapsed




tokens = [enc.encode('Hello there! My name is Shawn')]

sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

# loss = tf.reduce_mean(
#   tf.nn.sparse_softmax_cross_entropy_with_logits(
#       labels=context[:, 1:], logits=output['logits'][:, :-1]))


# tf_sample = sample.sample_sequence(
#     hparams=hparams,
#     length=args.sample_length,
#     context=context,
#     batch_size=args.batch_size,
#     temperature=args.temperature,
#     top_k=args.top_k,
#     top_p=args.top_p)



all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
train_vars = [v for v in all_vars if '/h' in v.name] if args.only_train_transformer_layers else all_vars

parameter_count = sum([np.prod(v.shape.as_list()) for v in train_vars])
print("This model is using %d parameters (%.2fM)" % (parameter_count, parameter_count/(1024.0*1024.0)))



