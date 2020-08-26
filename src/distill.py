#!/usr/bin/env python3

import os
import sys
sys.path += [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')]
sys.path += [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))]

import fire
import json
import numpy as np
import tensorflow as tf
import tflex
from tflex import p
import numpy as np

import model, sample, encoder

from sample import top_k_logits, top_p_logits, penalize_used

from pprint import pprint as pp

import nflex as nf

from functools import partial

from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.python.tpu import tpu; tpu_ops = tpu.tpu_ops;
from tensorflow.python.platform import tf_logging as logging

import tputil
import scipy
import threading

from tputil import op_scope


def get_num_shards():
  num_shards = tpu_function.get_tpu_context().number_of_shards
  if num_shards is None:
    logging.warning(
        "CrossShardOptimizer should be used within a tpu_shard_context, but "
        "got unset number_of_shards. Assuming 1.")
    num_shards = 1
  return num_shards


def read_prompt(prompt):
  if os.path.isfile(prompt):
    with open(prompt) as f:
      text = f.read()
  elif prompt is None or len(prompt) <= 0:
    text = input("Model prompt >>> ")
    if not text:
      text="\n"
  else:
    text = prompt
  if len(text) > 1 and text.endswith('\n'):
    text = text[:-1]
  return text

def read_tokens(enc, prompt):
  return enc.encode(read_prompt(prompt))

def distill_model(
    batch_size=1,
    sample_batch_size=None,
    model_name='117M',
    restore_from=None,
    distill_model_name='117M_half',
    distill_restore_from=None,
    init_tpu=False,
    prompt="Hello, my name is",
    length=32,
    #learning_rate=0.00002,
    #learning_rate=1e-09,
    learning_rate=4e-11,
    optimizer='adam',
):
    enc = encoder.get_encoder(distill_model_name)
    src_hparams = model.default_hparams(trainable=False, dtype=tf.bfloat16, scope='model')
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        src_hparams.override_from_dict(json.load(f))
    #dst_hparams = model.default_hparams(trainable=True, dtype=tf.bfloat16, scope='distill')
    dst_hparams = model.default_hparams(trainable=True, dtype=tf.float32, scope='distill')
    with open(os.path.join('models', distill_model_name, 'hparams.json')) as f:
        dst_hparams.override_from_dict(json.load(f))

    with tflex.Session(graph=tf.Graph()) as sess:

        if init_tpu:
          sess.run(tf.compat.v1.tpu.initialize_system())
        
        if sample_batch_size is None:
          sample_batch_size = batch_size

        context_IN = tf.placeholder(tf.int32, [sample_batch_size, None], name='context_IN')
        context_IN_fixed = tf.placeholder(tf.int32, [sample_batch_size, dst_hparams.n_ctx], name='context_IN_fixed')
        src = model.model(src_hparams, context_IN)
        dst = model.model(dst_hparams, context_IN_fixed, scope='distill')
        src_fixed = model.model(src_hparams, context_IN_fixed)
        dst_fixed = model.model(dst_hparams, context_IN_fixed, scope='distill')

        src_samples = tf.multinomial(src['logits'][:, -1, :], num_samples=128)

        #dst_saver = tf.train.Saver(var_list=tflex.get_bf16_var_list([x for x in tf.trainable_variables() if x.name.startswith(dst_hparams.scope+'/')]))
        dst_var_list = [x for x in tf.trainable_variables() if x.name.startswith(dst_hparams.scope+'/')]
        #dst_var_list_f32 = tflex.get_f32_var_list(dst_var_list)
        #dst_saver = tf.train.Saver(var_list=dst_var_list_f32)
        dst_saver = tf.train.Saver(var_list=dst_var_list)
        if distill_restore_from is None:
          distill_restore_from = 'gs://tpu-usc1/runs/gpt-2/distill/' + distill_model_name + 'f32'
        dst_ckpt = tflex.latest_checkpoint(distill_restore_from)
        if dst_ckpt is not None:
          dst_saver.restore(sess, dst_ckpt)

        #mid_saver = tf.train.Saver(var_list={v.name.replace('distill', 'model'): v for v in train_vars})
        
        if True:
          #saver = tflex.Saver(var_list=tf.local_variables())
          src_var_list = [x for x in tf.local_variables() if x.name.startswith(src_hparams.scope+'/')]
          src_saver = tflex.Saver(var_list=tflex.get_bf16_var_list(src_var_list))
          if restore_from is None:
            restore_from = 'gs://tpu-usc1/models/gpt-2/' + model_name
          src_ckpt = tflex.latest_checkpoint(restore_from)
          src_saver.restore(sess, src_ckpt)
        else:
          sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        tokens1 = [15496, 11, 616, 1438, 318, 10821, 2635, 13, 314,
            1101, 1719, 257, 2151, 379, 262, 7541, 13, 314, 1101, 407,
            1016, 284, 307, 994, 329, 257, 981, 13, 314, 1101, 655,
            1016, 284, 307, 994, 329, 257, 1310, 981, 13, 314, 1101,
            407, 1016, 284, 307, 994, 329, 257, 981, 13, 921, 760, 11,
            314, 1101, 407, 1016, 284, 307, 1804, 428, 13, 887, 11,
            345, 760, 11, 1081, 64, 16385, 43967, 2492, 470, 3446,
            1804, 1997, 1257, 329, 262, 717, 640, 428, 3329, 13, 679,
            373, 351, 3932, 30711, 553, 373, 477, 673, 531, 11, 607,
            3809, 39228, 13, 198, 198, 1, 10449, 20437, 553, 673, 531,
            13, 198, 198, 1, 1639, 3221, 1064, 661, 287, 360, 13, 34,
            13, 783, 508, 836, 470, 1107, 760, 703, 284, 17438, 553,
            673, 531, 13, 198, 198, 1, 39, 40133, 13, 1374, 714, 314,
            1701, 3932, 30711, 1965, 13, 198, 198, 1, 5812, 11, 645,
            11, 780, 611, 340, 373, 502, 326, 3088, 284, 29791, 465,
            3252, 11, 407, 616, 2095, 11, 3729, 407, 465, 49650, 1399,
            880, 11, 345, 760, 508, 2073, 373, 1804, 326, 30, 3271,
            3932, 30711, 526, 198, 198, 9360, 2951, 288, 19112, 510,
            284, 683, 290, 2993, 810, 465, 1986, 373, 422, 13, 366,
            5211, 407, 5490, 11, 17308, 553, 673, 531, 17707, 13,
            20899, 5921, 1678, 287, 1948, 550, 284, 1461, 503, 329,
            617, 9299, 13, 45209, 1568, 340, 373, 3489, 326, 3932,
            30711, 373, 2636, 13, 632, 373, 3016, 15896, 11, 475, 339,
            290, 20899, 5921, 1678, 2391, 3664, 7241, 290, 30442, 992,
            13, 679, 308, 5813, 13, 11735, 19036, 1660, 319, 465,
            1986, 355, 880, 355, 5548, 319, 340, 13, 1375, 6204, 510,
            6364, 13, 198, 198, 1, 5756, 502, 910, 703, 17533, 345,
            925, 502, 503, 994, 1909, 553, 673, 531, 13, 366, 40,
            1807, 356, 547, 2460, 783, 13, 887, 11, 262, 584, 4813,
            3751, 510, 351, 299, 3775, 11, 8148, 13, 18578, 9439, 30,
            399, 3775, 1165, 526, 198, 198, 11696, 30711, 714, 766,
            262, 3734, 1627, 1022, 262, 6678, 387, 2787, 33401, 5937,
            290, 11, 355, 355, 277, 17484, 278]

        def feed(s):
          if isinstance(s, str):
            s = enc.encode(s)
          if nf.rank(s) == 1:
            s = [s] * sample_batch_size
          return {context_IN: s, context_IN_fixed: [nf.pad_along_axis(x, src_hparams.n_ctx) for x in s]}

        #tokens_fixed = nf.pad_along_axis(tokens, src_hparams.n_ctx)

        #feed_dict = feed("Hello, my name")

        #tf.random.categorical(src_fixed['logits'][:, i, :], 1)

        start = len(enc.encode(prompt)) - 1
        assert start >= 0
        assert length > 0 and length <= src_hparams.n_ctx
        i = tf.Variable(start, dtype=tf.int32, collections=['local_variables'], trainable=False, name='i')
        length_var = tf.Variable(length, dtype=tf.int32, collections=['local_variables'], trainable=False, name='length')
        sess.run([x.initializer for x in [i, length_var]])
        enwik8 = tputil.tf_tok16_variable('gs://tpu-usc1/datasets/enwik8.tok16', name='enwik8')
        print('Loading dataset...')
        sess.run(enwik8.initializer)
        #features_op, labels_op = tputil.sample_text(enwik8, dst_hparams.n_ctx, batch_size=batch_size)

        openwebtext_ftfy = tputil.tf_tok16_variable('gs://tpu-usc1/datasets/openwebtext-ftfy.tok16', name='openwebtext-ftfy')
        openwebtext_ftfy_thread = threading.Thread(target=lambda: sess.run(openwebtext_ftfy.initializer), daemon=True)
        openwebtext_ftfy_thread.start() # Session.run device=None finished in 398.51sec: ['name: "tf_tok16_variable_1/openwebtext-ftfy/Assign"\nop: "Assi...'] {}

        @op_scope
        def cond(i, *args):
          return tf.less(i, length_var)

        @op_scope
        def out(X, hparams, past=None):
          output = model.model(hparams=hparams, X=X, past=past)
          if hparams.dtype != tf.float32:
            output['logits'] = tf.cast(output['logits'], tf.float32)
          return output

        # indices_of(batch_size, hparams.n_ctx)
        @op_scope
        def indices_of(*axes):
          #inds = tf.stack(tf.meshgrid(tf.range(4), tf.range(1024), indexing='ij'), axis=-1)
          return tf.stack(tf.meshgrid(*map(tf.range, axes), indexing='ij'), axis=-1)

        #upd = lambda tensor, indices, updates, name=None: tf.raw_ops.TensorScatterUpdate(tensor=tensor, indices=indices, updates=updates, name=name)

        @op_scope
        def body(i, tokens, hparams=src_hparams):
          #context = tf.transpose(tokens, [1, 0])
          context = tokens
          logits = out(context, hparams)['logits']
          #logits = logits[:, i, :] # errors on TPUs
          #logits = logits[:, 0, :] # works
          #logits1 = tf.gather(logits[0], [i])
          #logits1 = tf.transpose(logits, [1,0,2])
          #ind = tf.concat([tf.reshape(tf.range(sample_batch_size), [-1,1]), tf.tile([[i]], [sample_batch_size,1])], axis=1) # [[0, i], [1, i], ...]
          ind = tf.stack(tf.meshgrid(tf.range(sample_batch_size),[i]), axis=-1)[0] # [[0, i], [1, i], ...]
          logits1 = tf.gather_nd(logits, ind)
          samples = tf.random.categorical(logits1, num_samples=1, dtype=tf.int32)
          #samples = tf.random.categorical(logits[:, 0, :], num_samples=1, dtype=tf.int32)
          #return [i+1, tf.tensor_scatter_nd_update(tokens, [[i+1]], samples)]
          # inds = indices_of(sample_batch_size, hparams.n_ctx)
          # pp(samples)
          # pp(tokens)
          # pp(inds)
          # pp(inds[:, i+1])
          #final = tf.tensor_scatter_nd_update(tokens, [inds[:, i+1]], samples)
          #final = upd(tokens, inds[:, i+1], samples)
          #final = tf.tensor_scatter_nd_update(tokens, [[i+1]], samples)
          ind_out = tf.stack(tf.meshgrid(tf.range(sample_batch_size),[i+1]), axis=-1)[0] # [[0, i+1], [1, i+1], ...]
          final = tf.tensor_scatter_nd_update(tokens, ind_out, samples[:, 0])
          return [i+1, final]



        @op_scope
        def samp(i, hparams=src_hparams):
          #_, result = tf.while_loop(cond=cond, body=body, loop_vars=[i, tf.transpose(context_IN_fixed, [1, 0])], back_prop=False)
          _, result = tf.while_loop(cond=cond, body=partial(body, hparams=hparams), loop_vars=[i, context_IN_fixed], back_prop=False)
          return [result]
          

        @op_scope
        def body2(i, tokens, past, hparams=src_hparams):
          context = tokens
          lm_output = out(context, hparams, past=past)
          logits = lm_output['logits']
          presents = lm_output['present']
          ind = tf.stack(tf.meshgrid(tf.range(sample_batch_size),[i]), axis=-1)[0] # [[0, i], [1, i], ...]
          logits1 = tf.gather_nd(logits, ind)
          samples = tf.random.categorical(logits1, num_samples=1, dtype=tf.int32)
          ind_out = tf.stack(tf.meshgrid(tf.range(sample_batch_size),[i+1]), axis=-1)[0] # [[0, i+1], [1, i+1], ...]
          final = tf.tensor_scatter_nd_update(tokens, ind_out, samples[:, 0])
          return [i+1, final, presents]

        @op_scope
        def samp2(i, hparams=src_hparams):
          context = context_IN_fixed
          lm_output = out(context, hparams, past=None)
          logits = lm_output['logits']
          presents = lm_output['present']
          past_shape = model.past_shape(hparams=hparams, batch_size=sample_batch_size, sequence=hparams.n_ctx)
          _, result, past = tf.while_loop(cond=cond, body=partial(body2, hparams=hparams), loop_vars=[i, context, presents], shape_invariants=[tf.TensorShape([]), tf.TensorShape([sample_batch_size, hparams.n_ctx]), tf.TensorShape(past_shape)], back_prop=False)
          return [result]

        #import pdb; pdb.set_trace()
        #prompt = "Hello, my name is"
        #i.load(start)
        start = i.eval()
        N = length_var.eval()
        res = tf.tpu.shard(lambda: samp(i), num_shards=8, output_shard_axes=[0])
        # samp2 is slower than samp? why? it uses pasts...
        #res2 = tf.tpu.shard(lambda: samp2(i), num_shards=8, output_shard_axes=[0])
        #import pdb; pdb.set_trace()

        # qq=sess.run(res, feed(prompt))
        # #qq=sess.run(res2, feed(prompt))

        # # 345M takes 9.5sec for N=128
        # # 345M takes 18.42sec for N=256
        # # 345M takes 36.30sec for N=512
        # # 345M takes 71.81sec for N=1024

        # # 744M takes 6.55s for N=32
        # # 744M takes 19.90s for N=128
        # # 744M takes 37.65s for N=256

        # # 1558M (bfloat16) takes 8.93s for N=32
        # # 1558M (bfloat16) takes 29.67s for N=128
        # # 1558M (bfloat16) takes 57.95s for N=256

        # #pp([enc.decode(x) for x in np.transpose(qq).reshape([-1,1024])[:, 0:16]])
        # #pp([pp(enc.decode(x)) for x in np.transpose(qq).reshape([-1,1024])[:, 0:length_var.eval()]])
        # #import pdb; pdb.set_trace()
        # pp([pp(enc.decode(x)) for x in qq[0][:, 0:length_var.eval()]])
        # #import pdb; pdb.set_trace()

        # qq=sess.run(res, feed(prompt)); pp([pp(enc.decode(x)) for x in qq[0][:, 0:length_var.eval()]])

        
        @op_scope
        def distill(context):
          assert dst_hparams.n_ctx == src_hparams.n_ctx
          dst_output = out(context, dst_hparams)
          src_output = out(context, src_hparams)
          src_mask = tf.where(tf.range(src_hparams.n_ctx, dtype=tf.int32) < length_var, tf.ones([src_hparams.n_ctx]), tf.zeros([src_hparams.n_ctx]))
          dst_mask = tf.where(tf.range(dst_hparams.n_ctx, dtype=tf.int32) < length_var, tf.ones([dst_hparams.n_ctx]), tf.zeros([dst_hparams.n_ctx]))
          src_mask = tf.reshape(src_mask, [1,-1,1])
          dst_mask = tf.reshape(dst_mask, [1,-1,1])
          src_logits = src_output['logits'] * src_mask
          src_logits = tf.stop_gradient(src_logits)
          dst_mask = tf.stop_gradient(dst_mask)
          dst_logits = dst_output['logits'] * dst_mask
          mse = tf.squared_difference(src_logits, dst_logits)
          #loss = tf.reduce_mean(mse, axis=[1, 2])
          #loss = tf.reduce_mean(tf.reduce_sum(mse, axis=[1]), axis=[1])
          loss = tf.reduce_sum(mse, axis=[1]) # [batch, n_vocab]
          loss = tf.reduce_mean(loss, axis=[0]) # [n_vocab]
          loss = tf.reduce_mean(loss, axis=[0]) # []
          return loss


        @op_scope
        def valid_loss(context, hparams):
          output = out(context, hparams)
          logits = output['logits']
          loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=context[:, 1:], logits=logits[:, :-1])
          return loss


        @op_scope
        def kl_div_log_target(input, target):
          output = tf.exp(target) * (target - input)
          return output

        @op_scope
        def kl_div_non_log_target(input, target):
          output_pos = target * (tf.log(target) - input)
          zeros = tf.zeros_like(output_pos)
          output = tf.where(target > 0, output_pos, zeros)
          return output

        @op_scope
        def kl_div_1(input, target, log_target=False):
          if log_target:
            return kl_div_log_target(input=input, target=target)
          else:
            return kl_div_non_log_target(input=input, target=target)

        @op_scope
        def kl_div(input, target, reduction='sum', log_target=False):
          assert reduction in ['batchmean', 'sum']
          output = kl_div_1(input=input, target=target, log_target=log_target)
          reduced = tf.reduce_sum(output, axis=[0])
          if reduction == 'batchmean':
            reduced = reduced / reduced.shape[0].value
          return reduced

        @op_scope
        def distill_loss(context, temperature=None, alpha_ce=0.5, alpha_clm=0.5):
          assert dst_hparams.n_ctx == src_hparams.n_ctx
          if temperature is None:
            temperature = model.init_variable('temperature', shape=(), initializer=tf.constant_initializer(1.0, dtype=tf.float32), trainable=False)
          dst_output = out(context, dst_hparams)
          src_output = out(context, src_hparams)
          #src_n = src_hparams.n_ctx
          #dst_n = dst_hparams.n_ctx
          src_n = tf.shape(context)[-1]
          dst_n = src_n
          src_mask = tf.where(tf.range(src_n, dtype=tf.int32) < length_var, tf.ones([src_n]), tf.zeros([src_n]))
          dst_mask = tf.where(tf.range(dst_n, dtype=tf.int32) < length_var, tf.ones([dst_n]), tf.zeros([dst_n]))
          src_mask = tf.reshape(src_mask, [1,-1,1])
          dst_mask = tf.reshape(dst_mask, [1,-1,1])
          src_logits = src_output['logits'] * src_mask
          src_logits = tf.stop_gradient(src_logits)
          dst_mask = tf.stop_gradient(dst_mask)
          dst_logits = dst_output['logits'] * dst_mask
          input = tf.nn.log_softmax(dst_logits / temperature, axis=-1)
          #target = tf.nn.softmax(src_logits / temperature, axis=-1)
          #loss_ce = kl_div(input=input, target=target)
          target = tf.nn.log_softmax(src_logits / temperature, axis=-1)
          loss_ce = kl_div_1(input=input, target=target, log_target=True)
          loss_ce = loss_ce * (temperature) ** 2
          alpha_ce = alpha_ce + alpha_clm # TODO
          loss = alpha_ce * loss_ce
          if alpha_clm > 0.0:
            # TODO
            pass
          #loss = tf.reduce_mean(loss, axis=-1) # mean across batch
          # loss is [batch, ?, 50257]
          loss = tf.reduce_sum(loss, axis=-1)
          # loss is [batch, ?]
          return loss
          

        @op_scope
        def distill_train(context, loss_fn=distill_loss):
          with tf.variable_scope('', reuse=tf.AUTO_REUSE, use_resource=True):
            loss = loss_fn(context) # loss is [batch, ?]
            loss1 = tf.reduce_mean(loss, axis=-1) # loss1 is [batch]
            loss2 = tf.reduce_mean(loss1, axis=-1) # loss1 is []
            gs = tf.train.get_or_create_global_step()
            lr = model.init_variable('learning_rate', shape=(), initializer=tf.constant_initializer(learning_rate, dtype=tf.float32), trainable=False)
            beta1 = model.init_variable('beta1_power', shape=(), initializer=tf.constant_initializer(0.9, dtype=tf.float32), trainable=False)
            beta2 = model.init_variable('beta2_power', shape=(), initializer=tf.constant_initializer(0.999, dtype=tf.float32), trainable=False)
            opt = tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2)
            #opt = tf.contrib.tpu.CrossShardOptimizer(opt)
            all_vars = [v for v in tf.trainable_variables() if dst_hparams.scope in v.name]
            train_vars = all_vars
            #grads = tf.gradients(loss2, train_vars)
            scale = 1.0 / get_num_shards()
            loss3 = loss2 * scale
            #grads = opt.compute_gradients(loss3, train_vars)
            grads_and_vars = opt.compute_gradients(loss3, train_vars)
            #all_ok = tf.reduce_all([tf.reduce_all(tf.is_finite(g)) for g, v in grads_and_vars])
            #opt_grads = list(zip(grads, train_vars))
            #opt_apply = lambda: opt.apply_gradients(opt_grads, global_step=gs)
            # @op_scope
            # def opt_apply():
            #   # grads1 = [tf.compat.v1.tpu.cross_replica_sum(g) for g in grads]
            #   # opt_grads = list(zip(grads1, train_vars))
            #   # return opt.apply_gradients(opt_grads, global_step=gs)
            #   #grads_and_vars = list(zip(grads, train_vars))
            #   summed_grads_and_vars = []
            #   group_assignment=None
            #   for (grad, var) in grads_and_vars:
            #     if grad is None:
            #       summed_grads_and_vars.append((grad, var))
            #     else:
            #       with tf.colocate_with(grad):
            #         summed_grads_and_vars.append((tpu_ops.cross_replica_sum(
            #             grad, group_assignment), var))
            #   return opt.apply_gradients(summed_grads_and_vars, global_step=gs)
            # opt_train = tf.cond(all_ok, opt_apply, tf.no_op)
            # with tf.control_dependencies([opt_train]):
            #   return [tf.identity(loss, name='distill_train_op')]
            summed_grads_and_vars = []
            group_assignment=None
            for (grad, var) in grads_and_vars:
              if grad is None:
                summed_grads_and_vars.append((grad, var))
              else:
                with tf.colocate_with(grad):
                  summed_grads_and_vars.append((tpu_ops.cross_replica_sum(
                      grad, group_assignment), var))
            all_ok = tf.reduce_all([tf.reduce_all(tf.is_finite(g)) for g, v in summed_grads_and_vars])
            @op_scope
            def opt_apply():
              return opt.apply_gradients(summed_grads_and_vars, global_step=gs)
            opt_train = tf.cond(all_ok, opt_apply, tf.no_op)
            with tf.control_dependencies([opt_train]):
              return [tf.identity(loss, name='distill_train_op')]

        
          
        import pdb; pdb.set_trace()

        #lr = model.init_variable('learning_rate', shape=(), initializer=tf.constant_initializer(learning_rate, dtype=tf.float32), trainable=False)
        with tf.variable_scope('', use_resource=True):
          gs = tf.train.get_or_create_global_step()

        context_IN_train = tf.placeholder(tf.int32, [8, batch_size, dst_hparams.n_ctx], name='context_IN_train')
        train2 = tf.tpu.shard(lambda x: distill_train(x[0]), num_shards=8, inputs=[context_IN_train]); train2

        context_INs = tf.placeholder(tf.int32, [8, batch_size, None], name='context_INs')
        train3 = tf.tpu.shard(lambda x: distill_train(x[0], loss_fn=distill_loss), num_shards=8, inputs=[context_INs]); train3

        iters = model.init_variable('iterations', shape=(), initializer=tf.constant_initializer(4, dtype=tf.int32), trainable=False, dtype=tf.int32)


        @op_scope
        @tpu_function.on_device_training_loop
        def train_loop(x, shape=None):
          x1 = x() if callable(x) else x
          # if callable(x):
          #   assert shape is not None
          # else:
          #   shape = x.shape
          shape = x1.shape
          def train_step(loss):
            del loss
            x1 = x() if callable(x) else x
            print('x'); pp(x1)
            op = distill_train(x1, loss_fn=distill_loss)
            print('op'); pp(op)
            #with tf.control_dependencies(op):
            #  return tf.constant(0.0, dtype=tf.float32)
            return tf.reduce_sum(op, axis=-1)
          #print('train_step'); pp(train_step)
          #print('train_step_1'); pp(train_step(1e7))
          #print('train_step_2'); pp(train_step([1e7]))
          op = tf.contrib.tpu.repeat(tf.maximum(0,iters-1), train_step, [[[1e7] * shape[0].value]])
          print('op_final'); pp(op)
          #op = tf.constant(0.0, dtype=tf.float32)
          #return op
          with tf.control_dependencies(op):
            loss = distill_train(x1, loss_fn=distill_loss)
            #return [loss]
            return loss

            
        train4 = tf.tpu.shard(lambda x: train_loop(x[0]), num_shards=8, inputs=[context_INs]); train4

        all_vars = [v for v in tf.trainable_variables() if dst_hparams.scope in v.name]
        train_vars = all_vars
        adam_vars = [x for x in tf.get_collection('variables') if '/Adam' in x.name or 'beta1_power' in x.name or 'beta2_power' in x.name]
        gs = tf.train.get_or_create_global_step()
        lr = model.get_variable('learning_rate', tf.local_variables())
        beta1 = model.get_variable('beta1_power', tf.local_variables())
        beta2 = model.get_variable('beta2_power', tf.local_variables())
        temp = model.get_variable('temperature', tf.local_variables())
        other_vars = [gs, lr, beta1, beta2, temp, iters]

        #import pdb; pdb.set_trace()
        #train2 = tf.tpu.shard(distill_train, num_shards=8, inputs=tf.unstack(context_IN_train)); train2
        #train2 = tf.tpu.shard(lambda x: distill_train(x[0]), num_shards=8, inputs=[context_IN_train]); train2

        import pdb; pdb.set_trace()
        sess.run([x.initializer for x in other_vars])

        sess.run([x.initializer for x in tf.global_variables() if x.name.startswith('beta1') or x.name.startswith('beta2')])

        #sess.run([x.initializer for x in adam_vars + all_vars + other_vars])
        sess.run([x.initializer for x in adam_vars + other_vars])
        #sess.run([x.initializer for x in adam_vars])

        length_var.load(6)
        gs.load(264)
        lr.load(4e-11)
        i.load(3)
        iters.load(320)
        gs.load(9971)
        lr.load(8e-11)
        gs.load(14144)
        lr.load(1.6e-10)
        temp.load(2.0)

        dst_val_loss = valid_loss(context_IN, dst_hparams)
        src_val_loss = valid_loss(context_IN, src_hparams)
        chk = tf.tpu.shard(lambda: samp(i, hparams=dst_hparams), num_shards=8, output_shard_axes=[0])

        gs.load(0)
        iters.load(80)

        import pdb; pdb.set_trace()

        src_wte = model.get_variable('model/wte', tf.local_variables())
        src_wpe = model.get_variable('model/wpe', tf.local_variables())
        dst_wte = model.get_variable('distill/wte')
        dst_wpe = model.get_variable('distill/wpe')
        transfer_we = lambda: tf.group([tf.assign(dst_wte, tf.cast(src_wte, dst_hparams.dtype)), tf.assign(dst_wpe, tf.cast(src_wpe, dst_hparams.dtype))])

        #import pdb; pdb.set_trace()

        iters.load(40); iters.load(10)
        i.load(0)
        length_var.load(32)
        #lr.load(0.000055) # from old swarm training
        lr.load(5e-4) # from transformers distill example
        #features_op, labels_op = tputil.sample_text(enwik8, dst_hparams.n_ctx, batch_size=batch_size)

        #sample_fn = lambda batch_size: tf.cast(tputil.sample_tok16(enwik8, dst_hparams.n_ctx, batch_size=batch_size), tf.int32)
        #sample_fn = lambda batch_size: tputil.sample_tok16(enwik8, dst_hparams.n_ctx, batch_size=batch_size)
        #sample_fn = lambda batch_size: tf.stop_gradient(tputil.sample_text(enwik8, dst_hparams.n_ctx, batch_size=batch_size)[0])

        @op_scope
        def sample_fn(batch_size, dataset=None):
          if dataset is None:
            if openwebtext_ftfy_thread.isAlive():
              print('Using enwik8')
              dataset = enwik8
            else:
              print('Using openwebtext_ftfy')
              dataset = openwebtext_ftfy
          tokens = tputil.sample_tok16(dataset, dst_hparams.n_ctx, batch_size=batch_size)
          #tokens = tputil.sample_text(dataset, dst_hparams.n_ctx, batch_size=batch_size)[0]
          tokens = tf.stop_gradient(tokens)
          return tokens

        # @op_scope
        # def sample_batch(*args, **kws):
        #   return sample_fn(*args, batch_size=batch_size, **kws)
        features_op = sample_fn(batch_size); sharded_features = tf.tpu.shard(lambda: sample_fn(batch_size), num_shards=8)
        shards_op = sample_fn(batch_size*8)

        #train_op1 = tf.tpu.shard(lambda x: distill_train(x[0]), num_shards=8, inputs=[[features_op]*8])
        #train_op1 = tf.tpu.shard(lambda x: distill_train(x[0][0:4]), num_shards=8, inputs=[[features_op]*8])
        #features_op2, labels_op2 = tputil.sample_text(enwik8, dst_hparams.n_ctx, batch_size=batch_size*8)
        train_op1 = tf.tpu.shard(lambda x: distill_train(x), num_shards=8, inputs=[shards_op])
        train_op2 = tf.tpu.shard(lambda: distill_train(sample_fn(batch_size)), num_shards=8)
        #train_op3 = tf.tpu.shard(lambda: train_loop(partial(sample_fn, batch_size=batch_size)), num_shards=8)
        import pdb; pdb.set_trace()
        #train_op4 = tf.tpu.shard(lambda: tf.contrib.tpu.repeat(iters, lambda loss: distill_train(sample_fn(batch_size)), np.ones([batch_size, dst_hparams.n_ctx], dtype=np.float32) * 1e7), num_shards=8)
        #adam_vars = [x for x in tf.get_collection('variables') if '/Adam' in x.name or 'beta1_power' in x.name or 'beta2_power' in x.name]; sess.run([x.initializer for x in adam_vars])
        #train_op2 = tf.tpu.shard(lambda x: train_loop(x), num_shards=8, inputs=[features_op2])
        #train_op0 = tf.tpu.shard(lambda x: pp(x), num_shards=8, inputs=[features_op2])

        sn = tflex.get_tpu_serial_number()
        swarm = []
        # swarm = ['n-b89be459-w-0', 'n-ba4ed764-w-0']
        # if sn not in swarm:
        #   print('Swarm changed!')
        #   import pdb; pdb.set_trace()

        #features_op1, labels_op1 = tputil.sample_text(enwik8, dst_hparams.n_ctx, batch_size=batch_size)
        #enq_ops = [tpu_ops.infeed_enqueue_tuple([features_op1], shapes=[features_op1.shape.as_list()], device_ordinal=i) for i in range(8)]
        #enq_fn = lambda batch_size: [tpu_ops.infeed_enqueue_tuple([tputil.sample_text(enwik8, dst_hparams.n_ctx, batch_size=batch_size)[0]], shapes=[features_op1.shape.as_list()], device_ordinal=i) for i in range(8)]
        enq_tuple = lambda values, device_ordinal: tpu_ops.infeed_enqueue_tuple(values, [x.shape for x in values], device_ordinal=device_ordinal)
        #enq_fn = lambda batch_size: [tpu_ops.infeed_enqueue_tuple([sample_fn(batch_size)], device_ordinal=i) for i in range(8)]
        enq_fn = lambda batch_size: [enq_tuple([sample_fn(batch_size)], device_ordinal=i) for i in range(8)]
        enq_batch = partial(enq_fn, batch_size=batch_size)
        enq_ops = enq_batch()
        enq = tf.group(enq_ops)
        #enq_iters1 = tf.contrib.tpu.repeat(iters+1, enq_fn)
        enq_iters = tf.contrib.tpu.repeat(iters, enq_batch)

        def dequeue(): return tpu_ops.infeed_dequeue_tuple([tf.int32], shapes=[[batch_size, dst_hparams.n_ctx]])[0]


        deq = tf.tpu.shard(dequeue, num_shards=8)
        deq_iters = tf.contrib.tpu.repeat(iters, lambda: tf.group(tf.tpu.shard(dequeue, num_shards=8)) )
        #import pdb; pdb.set_trace()

        train_op3 = tf.tpu.shard(lambda: train_loop(dequeue), num_shards=8)
        #with tf.control_dependencies([enq]): train_op4 = tf.tpu.shard(lambda: train_loop(dequeue()[0]), num_shards=8)


        @op_scope
        def train_fn():
          with tf.control_dependencies([enq_iters]):
            #return tf.tpu.shard(lambda: train_loop(lambda: dequeue()[0], shape=[features_op1.shape]), num_shards=8)
            return tf.identity(train_op3[0])


        train_op4 = train_fn()
        adam_vars = [x for x in tf.get_collection('variables') if '/Adam' in x.name or 'beta1_power' in x.name or 'beta2_power' in x.name]; sess.run([x.initializer for x in adam_vars])
        import pdb; pdb.set_trace()

        distill_restore_from = distill_restore_from[0:-3] + 'tmp'
        dst_ckpt = tflex.latest_checkpoint(distill_restore_from); dst_saver.restore(sess, dst_ckpt); gs.load(int(dst_ckpt.split('-')[-1]))
        length_var.load(1024); iters.load(20);
        import pdb; pdb.set_trace()

        # while True: update_swarm(); zz1=[sess.run(train_op4)]; resave_swarm(); [pp(scipy.stats.describe(x)) for x in zz1[0][:,0:length_var.eval()]]; pp([np.mean(x) for x in zz1[0][:,0:length_var.eval()]][0:8]); pp(scipy.stats.describe([np.mean(x) for x in zz1[0][:,0:length_var.eval()]])); pp(sess.run({'gs': gs, 'length_var': length_var, 'i': i, 'iters': iters, 'temp': temp})); print('--------');

        # while True: zz1=[sess.run(train_op4)]; [pp(scipy.stats.describe(x)) for x in zz1[0][:,0:length_var.eval()]]; pp([np.mean(x) for x in zz1[0][:,0:length_var.eval()]][0:8]); pp(scipy.stats.describe([np.mean(x) for x in zz1[0][:,0:length_var.eval()]])); pp(sess.run({'gs': gs, 'length_var': length_var, 'i': i, 'iters': iters, 'temp': temp})); # print('saving...'); sess.run(h_save); sess.run(h_save_gs); print('saved.'); print('--------');

      # def train_fn_after():
      #   with tf.control_dependencies([enq_iters]):
      #     #return tf.tpu.shard(lambda: train_loop(lambda: dequeue()[0], shape=[features_op1.shape]), num_shards=8)
      #     return tf.identity(train_op3[0])


      # train_op5 = train_fn()

      # def train_fn():
      #   enqueue_op = tf.contrib.tpu.repeat(iters+1, enq_fn)
      #   with tf.control_dependencies([enqueue_op]):
      #     #return tf.tpu.shard(lambda: train_loop(lambda: dequeue()[0], shape=[features_op1.shape]), num_shards=8)
      #     return tf.identity(train_op3[0])


      # train_op5 = train_fn()

        
        
        
        
        adam_vars = [x for x in tf.get_collection('variables') if '/Adam' in x.name or 'beta1_power' in x.name or 'beta2_power' in x.name]; sess.run([x.initializer for x in adam_vars])
        import pdb; pdb.set_trace()

        # while True: zz1=sess.run(train_op4); [pp(scipy.stats.describe(x)) for x in zz1[0][:,0:length_var.eval()]]; pp([np.mean(x) for x in zz1[0][:,0:length_var.eval()]][0:8]); pp(scipy.stats.describe([np.mean(x) for x in zz1[0][:,0:length_var.eval()]])); pp(sess.run({'gs': gs, 'length_var': length_var, 'i': i, 'iters': iters}))

        # while True: sess.run(enq_iters); zz1=sess.run(train_op3); [pp(scipy.stats.describe(x)) for x in zz1[0][:,0:length_var.eval()]]; pp([np.mean(x) for x in zz1[0][:,0:length_var.eval()]][0:8]); pp(scipy.stats.describe([np.mean(x) for x in zz1[0][:,0:length_var.eval()]])); pp(gs.eval())

        # lv = length_var.eval(); length_var.load(32); "qq2=sess.run(chk, feed(qq[0][0:8,0:length_var.eval()-1]))"; qq2=sess.run(chk, feed([enc.encoder['<|endoftext|>']])); [pp(enc.decode(x)) for x in qq2[0][:, 0:length_var.eval()]]; length_var.load(lv)

        # while True: zz1=sess.run(train_op2); [pp(scipy.stats.describe(x)) for x in zz1[0][:,0:length_var.eval()]]; pp([np.mean(x) for x in zz1[0][:,0:length_var.eval()]][0:8]); pp(scipy.stats.describe([np.mean(x) for x in zz1[0][:,0:length_var.eval()]])); pp(gs.eval())

        # """lv = length_var.eval(); length_var.load(32); qq=sess.run(res, feed(prompt)); length_var.load(lv)"""; qq_in=np.transpose(qq, [1,0,2])[:,:,0:length_var.eval()]; qq_in=np.transpose(qq, [1,0,2])[:,:,0:length_var.eval()]; qq_in = np.reshape(qq_in, [-1, 8, length_var.eval()]); zz1=sess.run(train4, {context_IN_train: np.reshape(np.transpose(qq, [1,0,2]), [8, batch_size, -1]), context_INs: qq_in})[0]; print('------');pp(zz1); pp(np.mean(zz1, axis=-1)); pp(np.mean(np.mean(zz1, axis=-1), axis=-1)); pp(np.mean(zz1)); print('######'); vv="""qq2=sess.run(res, feed(qq_in[1,:,0:length_var.eval()-1])); pp([pp(enc.decode(x)) for x in qq2[0][:, 0:length_var.eval()]]);"""; zz=sess.run(dst_val_loss, feed(prompt)); [pp(scipy.stats.describe(x)) for x in zz1]; print('%%%%'); pp(zz); pp(np.sum(zz, axis=-1)); print('%%%%'); pp(gs.eval())

        # qq_in=np.transpose(qq, [1,0,2])[:,:,0:length_var.eval()]; zz1=sess.run(train4, {context_IN_train: np.transpose(qq, [1,0,2]), context_INs: qq_in})[0]; print('------');pp(zz1); pp(np.sum(zz1, axis=-1)); print('######'); qq2=sess.run(chk, feed(qq_in[1,:,0:length_var.eval()-1])); pp([pp(enc.decode(x)) for x in qq2[0][:, 0:length_var.eval()]]); zz=sess.run(dst_val_loss, feed(prompt)); pp(zz); pp(np.sum(zz)); pp(gs.eval())

        #pp(sess.run(train3, {context_IN_train: np.transpose(qq, [1,0,2]), context_INs: np.transpose(qq, [1,0,2])[:,:,0:length_var.eval()]})[0]); qq2=sess.run(chk, feed(prompt)); pp([pp(enc.decode(x)) for x in qq2[0][:, 0:length_var.eval()]]); zz=sess.run(dst_val_loss, feed(prompt)); pp(zz); pp(np.sum(zz))

        #pp(sess.run(train3, {context_IN_train: np.transpose(qq, [1,0,2]), context_INs: np.transpose(qq, [1,0,2])[:,:,0:length_var.eval()]})[0]); qq2=sess.run(chk, feed(prompt)); pp([pp(enc.decode(x)) for x in qq2[0][:, 0:length_var.eval()]])


        loss0 = distill_loss(context_IN)
        loss1 = tf.reduce_mean(loss0, axis=-1)
        loss2 = tf.reduce_mean(loss1, axis=-1)
        grads0 = tf.gradients(loss0, train_vars)
        grads1 = tf.gradients(loss1, train_vars)
        grads2 = tf.gradients(loss2, train_vars)

        i.load(0)
        length_var.load(31); temp.load(1.0);

        import pdb; pdb.set_trace()

        sess.run(train3, {context_IN_train: np.transpose(qq, [1,0,2]), context_INs: np.transpose(qq, [1,0,2])[:,:,0:6]})
        #sess.run(train2, {context_IN_train: np.transpose(qq, [1,0,2])})

        import pdb; pdb.set_trace()

        saving = tf.train.Saver(var_list=tf.trainable_variables())
        #saving.save(sess, 'gs://tpu-usc1/runs/gpt-2/distill/' + distill_model_name + 'f32/model', global_step=gs.eval())

        import pdb; pdb.set_trace()

        qq2=sess.run(chk, feed(prompt)); pp([pp(enc.decode(x)) for x in qq2[0][:, 0:length_var.eval()]])

        # sess.run(train2, {context_IN_train: np.transpose(qq, [1,0,2])}); qq2=sess.run(chk, feed(prompt)); pp([pp(enc.decode(x)) for x in qq2[0][:, 0:length_var.eval()+1]]); qq3=sess.run(res, feed(prompt)); pp([pp(enc.decode(x)) for x in qq3[0][:, 0:length_var.eval()+1]]);

        import pdb; pdb.set_trace()

        raw_text = read_prompt(prompt)
        encoded_tokens = enc.encode(raw_text)
        context_tokens = np.array([encoded_tokens] * batch_size, dtype=np.int32)

        generated = 0
        while nsamples == 0 or generated < nsamples:

            # step_In = {context_IN: context_tokens}
            # step_Out = sess.run(step_OUT, step_In)

            # #body_In = {past_IN: step_Out['present'], prev_IN: context_tokens[:, -1], output_IN: context_tokens}
            # body_In = {past_IN: step_Out['present'], output_IN: context_tokens}
            # body_Out = sess.run(body_OUT, body_In)


            fused_In = {context_IN: context_tokens}
            fused_Out = sess.run(fused_OUT, fused_In)

#             print('=== step() ===')
#             print('')
#             print('--- step() input: context_IN=context_tokens ---')
#             print('')
#             for k, v in step_In.items(): p([k, v]) # p({k: v})
#             print('')
#             #print('--- step() output: step_OUT["logits"], step_OUT["present"] = step(tokens=context_IN) ---')
#             print('--- step() output: step(tokens=context_IN) ---')
#             print('')
#             for k, v in step_Out.items(): p([k, v]) # p({k: v})

#             print('=== body() ===')
#             print('')
#             #print('--- body() input: past_IN=step_OUT["present"], prev_IN=context_tokens[:, -1], output_IN=context_tokens ---')
#             print('--- body() input: past_IN=step_OUT["present"], output_IN=context_tokens ---')
#             print('')
#             for k, v in body_In.items(): p([k, v]) # p({k: v})
#             print('')
#             #print('--- body() output: body_OUT["past"], body_OUT["prev"], body_OUT["output"] = body(past_IN, prev_IN, output_IN) ---')
#             #print('--- body() output: body(past_IN, prev_IN, output_IN) ---')
#             print('--- body() output: body(past_IN, output_IN) ---')
#             print('')
#             for k, v in body_Out.items(): p([k, v]) # p({k: v})

            print('')
            print('=== final results ===')
            print('')
            p(['prompt', raw_text])
            #p(['completion', [enc.decode(x) for x in body_Out['output']]])
            p(['completion', [enc.decode(x) for x in fused_Out[-1]]])
            print('')
            import pdb; pdb.set_trace()
            context_tokens = fused_Out[-1]
            

if __name__ == '__main__':
    fire.Fire(distill_model)
