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
import numpy as np

import model, sample, encoder

from sample import top_k_logits, top_p_logits, penalize_used

from pprint import pprint as pp

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

def sample_model(
    model_name='117M',
    prompt='\n',
    restore_from=None,
    seed=None,
    nsamples=0,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=0.0,
    penalize=0,
    epsilon=-1e10,
):
    """
    Run the sample_model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to
     reproduce results
    :nsamples=0 : Number of samples to return, if 0, continues to
     generate samples indefinately.
    :batch_size=1 : Number of batches (only affects speed/memory).
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    :penalize=0.0 : Float value controlling "used" penalty. Implements repetition
     reduction (similar to CTRL) if set to a value > 0. A decent setting might be 0.85
     with temperature 0.3 and top_k 40.
    """
    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tflex.Session(graph=tf.Graph()) as sess:
        np.random.seed(seed)
        tf.set_random_seed(seed)


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
        

        context = tf.placeholder(tf.int32, [batch_size, None])
      

        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        #context_output = step(hparams, context[:, :-1])
        context_output = step(hparams, context)

        past_ctx = tf.placeholder(tf.float32, model.past_shape(hparams=hparams, batch_size=batch_size))
        prev_ctx = tf.placeholder(tf.int32, [batch_size])
        output_ctx = tf.placeholder(tf.int32, [batch_size, None])

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

        past, prev, output = body(past_ctx, prev_ctx, output_ctx)
        sample = {'past': past, 'prev': prev, 'output': output}
    
        

        # output = sample.sample_sequence(
        #     hparams=hparams, length=length,
        #     start_token=enc.encoder['<|endoftext|>'],
        #     batch_size=batch_size,
        #     temperature=temperature, top_k=top_k, top_p=top_p, penalize=penalize
        # )[:, 1:]

        saver = tflex.Saver()
        if restore_from is None:
          restore_from = os.path.join('models', model_name)
        ckpt = tflex.latest_checkpoint(restore_from)
        saver.restore(sess, ckpt)

        generated = 0
        while nsamples == 0 or generated < nsamples:
            raw_text = read_prompt(prompt)
            encoded_tokens = enc.encode(raw_text)
            context_tokens = np.array([encoded_tokens] * batch_size, dtype=np.int32)
            out = sess.run(context_output, {context: context_tokens})
            #out1 = {k+'_'+str(i): x[i] for k, v in out.items() for i, x in enumerate(v)}

            past0 = out['presents']
            prev0 = context_tokens[:, -1]
            output0 = context_tokens

            result = sess.run(sample, {past_ctx: past0, prev_ctx: prev0, output_ctx: output0})
            o = {}
            p = lambda x: pp(tflex.prettify(x))
            for k, v in {'past_ctx': past0, 'prev_ctx': prev0, 'output_ctx': output0}.items():
              p({k: v})
            print('--- step() ---')
            for k, v in out.items():
              p({k: v})
            print('--- body(step()) ---')
            for k, v in result.items():
              p({k: v})
            print('--- output: ---')
            p({'prompt': raw_text})
            p({'completion': [enc.decode(x) for x in result['output']]})
            #result1 = {k+'_'+str(i): x[i] for k, v in result.items() for i, x in enumerate(v)}

            #pp(tflex.pretty_obj({'past': past, 'prev': prev, 'output': output}))
            #pp(tflex.prettify(out, n=40))
            #pp(tflex.prettify(result, n=40))
            #pp(tflex.prettify(o))

            # loop_vars=[
            #     context_output['presents'],
            #     context[:, -1],
            #     context,
            # ],
            
            import pdb; pdb.set_trace()

if __name__ == '__main__':
    fire.Fire(sample_model)
