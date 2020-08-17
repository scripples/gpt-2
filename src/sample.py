import tensorflow as tf

import model

import functools


def op_scope(fn, name=None):
    if name is None:
        name = fn.__name__
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        with tf.name_scope(fn.__name__):
            return fn(*args, **kwargs)
    return _fn

@op_scope
def penalize_used(logits, output, penalize=0.85):

    # I want to change the indices of logits wherever the index is found in output
    change_tensor = tf.zeros_like(logits, dtype=logits.dtype)
    unique = tf.unique(output[0])[0]
    ones = tf.ones_like(unique, dtype=unique.dtype)
    indices = tf.expand_dims(unique, 1)

    updates = tf.scatter_nd(indices, ones, [logits.shape[1]])

    bool_tensor = tf.expand_dims(tf.cast(updates, tf.bool), 0)

    return tf.compat.v1.where(bool_tensor, logits * penalize, logits)

@op_scope
def top_k_logits(logits, k, epsilon=-1e10):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * epsilon,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


@op_scope
def top_p_logits(logits, p, epsilon=-1e10):
    with tf.variable_scope('top_p_logits'):
        logits_sort = tf.sort(logits, direction='DESCENDING')
        probs_sort = tf.nn.softmax(logits_sort)
        probs_sums = tf.cumsum(probs_sort, axis=1, exclusive=True)
        logits_masked = tf.where(probs_sums < p, logits_sort, tf.ones_like(logits_sort)*1000) # [batchsize, vocab]
        min_logits = tf.reduce_min(logits_masked, axis=1, keepdims=True) # [batchsize, 1]
        return tf.where(
            logits < min_logits,
            tf.ones_like(logits, dtype=logits.dtype) * epsilon,
            logits,
        )


@op_scope
def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=0.0, epsilon=-1e10, penalize=0.0):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    @op_scope
    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)
        if hparams.dtype != tf.float32:
            lm_output["logits"] = tf.cast(lm_output["logits"], tf.float32)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    @op_scope
    def body(past, prev, output):
        next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
        logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
        if penalize > 0.0:
            logits = penalize_used(logits, output, penalize=penalize)
        if top_p > 0.0:
            logits = top_p_logits(logits, p=top_p, epsilon=epsilon)
        else:
            logits = top_k_logits(logits, k=top_k, epsilon=epsilon)
        samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
        return [
            tf.concat([past, next_outputs['presents']], axis=-2),
            tf.squeeze(samples, axis=[1]),
            tf.concat([output, samples], axis=1),
        ]

    @op_scope
    def fused(i, tokens):
        next_outputs = step(hparams, tokens)
        past, prev, output = body(past=next_outputs['presents'], prev=tokens[..., -1], output=tokens)
        #past, output = body(past=next_outputs['present'], output=tokens)
        return [i+1, output]


    with tf.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        #context_output = step(hparams, context[:, :-1])

        def cond(i, *args):
            return i < length

        tokens = tf.while_loop(
            cond=cond, body=fused,
            loop_vars=[0, context],
            shape_invariants=[tf.TensorShape([]), tf.TensorShape([batch_size, None])],
            back_prop=False,
        )

        return tokens[-1]


@op_scope
def sample_sequence(*, hparams, length, start_length=None, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=0.0, epsilon=-1e10, penalize=0.0):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        #assert start_length is not None, 'Specify start_length!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        #context = tf.fill([batch_size, 1], start_token)
        #start_length = 1
        context = tf.fill([batch_size, length], start_token)
        #start_length = length
        start_length = 0

    @op_scope
    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)
        if hparams.dtype != tf.float32:
            lm_output["logits"] = tf.cast(lm_output["logits"], tf.float32)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    @op_scope
    def body(past, prev, output):
        next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
        logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
        if penalize > 0.0:
            logits = penalize_used(logits, output, penalize=penalize)
        if top_p > 0.0:
            logits = top_p_logits(logits, p=top_p, epsilon=epsilon)
        else:
            logits = top_k_logits(logits, k=top_k, epsilon=epsilon)
        samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
        return [
            tf.concat([past, next_outputs['presents']], axis=-2),
            tf.squeeze(samples, axis=[1]),
            tf.concat([output, samples], axis=1),
        ]

    size = None
    if size is None:
      #size = hparams.n_ctx
      size = length
      assert size > 0

    @op_scope
    def fused(i, tokens):
        #@op_scope
        #def op0(): return tokens[0:i, ...]
        #op0 = op0()
        #op0 = tf.gather(tokens, tf.range(0, i))
        #op0 = tf.gather(tokens, [i])
        #op0 = tf.concat([tokens[-1:, :], tokens[0:-1, :]], axis=0)
        #op0 = tf.concat([tokens[0:1, :], tokens[0:-1, :]], axis=0)
        op1 = tf.roll(tokens, shift=-1, axis=0)
        #op0 = op1[0:size-1, :]
        op0 = op1[0:-1, :]
        import pdb; pdb.set_trace()
        #op0 = tokens
        #import pdb; pdb.set_trace()
        output = tf.transpose(op0, [1, 0])
        next_outputs = step(hparams, output)
        past, prev, output = body(past=next_outputs['presents'], prev=output[..., -1], output=output)
        #return [i+1, tf.tensor_scatter_nd_update(tokens, [[i]], [prev])]
        N = tokens.shape.as_list()[0]
        out0 = tf.tensor_scatter_nd_update(tokens, [[N - 1]], [prev])
        out1 = out0
        #out1 = tf.roll(out0, shift=-1, axis=0)
        import pdb; pdb.set_trace()
        return [i+1, out1]


    with tf.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        #context_output = step(hparams, context[:, :-1])

        @op_scope
        def cond(i, *args):
            return i < length

        #import pdb; pdb.set_trace()
        def zero_padded_context():
          #context.set_shape([batch_size, start_length])
          #tokens = tf.zeros([batch_size, size - start_length], dtype=tf.int32)
          #import pdb; pdb.set_trace()
          tokens = tf.zeros([batch_size, size], dtype=tf.int32)
          tokens = tf.concat([context, tokens], axis=-1)
          tokens = tokens[:, 0:size]
          return tokens

        tokens = zero_padded_context()
        #tokens.set_shape([batch_size, hparams.n_ctx])
        tokens = tf.transpose(tokens, [1, 0])
        if start_length is None:
          start_length = tf.shape(context)[-1]
        start_length = 0

        tokens = tf.while_loop(
            cond=cond, body=fused,
            loop_vars=[start_length, tokens],
            shape_invariants=[tf.TensorShape([]), tf.TensorShape([tokens.shape.as_list()[0], batch_size])],
            back_prop=False,
        )

        @op_scope
        def afterwards(tokens):
          tokens = tokens[-1]
          tokens = tf.transpose(tokens[0:length, :], [1, 0])
          return tokens
        return afterwards(tokens)
