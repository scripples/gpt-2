# Actual Code

#Inits

#!/usr/bin/env python3

%cd '/content/drive/My Drive/ML/scripp_gpt-2/'

import os
import sys
sys.path.append("/content/drive/My Drive/ML/scripp_gpt-2/src")
print(sys.path)
import fire
import json
import numpy as np
import tensorflow as tf
import tflex
import time
import re
import time

import model, sample, encoder

def truncate_sentence(text):
  allowed_ends = [".", "?", "!", "\""]
  newlist = []
  endtextchar = str([char for char in text[-1]][-1])
  if endtextchar not in allowed_ends:
    text = text.rsplit('\n', 1)[0]
    print("Truncated!")
  newlist.append(text)
  return newlist

model_name='1558M'
restore_from='checkpoint/Jerry_1K.hdf5'
seed=None
nsamples=2
batch_size=1
length=2048
temperature=1
top_p=0.7
top_k=60
penalize=.3
split_context = .5

enc = encoder.get_encoder(model_name)
hparams = model.default_hparams()
with open(os.path.join('models', model_name, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))

#init session with default graph
sess = tflex.Session(graph=tf.get_default_graph(), config=tf.ConfigProto(log_device_placement=True))

#seeds
np.random.seed(seed)
tf.set_random_seed(seed)

#Encoder
enc = encoder.get_encoder(model_name)
hparams = model.default_hparams()
with open(os.path.join('models', model_name, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))

output = sample.sample_sequence(
    hparams=hparams, length=length,
    start_token=enc.encoder['<|endoftext|>'],
    batch_size=batch_size,
    temperature=temperature, top_k=top_k, top_p=top_p, penalize=penalize
)[:, 1:]
print(output)
saver = tflex.Saver(reshape=True)
if restore_from is None:
  restore_from = os.path.join('models', model_name)
ckpt = tflex.latest_checkpoint(restore_from)
ckpt = restore_from
saver.restore(sess, ckpt)


if not os.path.exists("pklsave_gpt2.pkl"):
  templist = []
  with open("pklsave_gpt2.pkl", "wb") as pklinit:
    pickle.dump(templist, pklinit)


#Do loop

while True:

  with open("pklsave_gpt2.pkl", "rb") as savepkl:
    content = pickle.load(savepkl)

  addlist = []
  if len(content) < 10:
    while(len(addlist)) < 2:

      prefix = None
      start = time.time()
      generated = 0
      gen_texts = []
      truncate = '<|endoftext|>'
      destination_path = None
      return_as_list = True
      context = tf.placeholder(tf.int32, [batch_size, None])

      if batch_size is None:
          batch_size = 1
      assert nsamples % batch_size == 0
      if nsamples == 1:
          sample_delim = ''
      else:
          sample_delim = '\n'
      if not length:
          assert truncate is not None, "If generating a non-fixed length \
                  sample, must have a truncation term."
      assert 0 < split_context < 1

      sess = sess

      while generated < nsamples:
          gen_text = [''] * batch_size
          truncated = [False] * batch_size
          if prefix:
              context_tokens = [prefix_enc] * batch_size
          else: 
              context_tokens = [[enc.encoder['<|endoftext|>']]] * batch_size
          total_tokens = len(context_tokens[0])
          #loop

          while False in truncated:
              num_tokens = 1023 - (len(context_tokens[0]))
              output = sample.sample_sequence(
                  hparams=hparams,
                  length=min(length if length else 1023, num_tokens),
                  context=context,
                  batch_size=batch_size,
                  temperature=temperature, top_k=top_k, top_p=top_p
              )[:, 1:]
              print("Generating sequence...")
              out = sess.run(output, feed_dict={
                      context: context_tokens
                  })
              total_tokens += num_tokens
              print("Total tokens: " + str(total_tokens))
              for i in range(batch_size):
                  text = enc.decode(out[i])
                  if prefix:
                      text = enc.decode(context_tokens[i][:1]) + text
                  if truncate or all(gen_text):
                      context_tokens[i] = out[i][int(len(out[i])*(1-split_context)):]
                      if gen_text[i] != '':
                          split = re.split('[.!?]', gen_text[i])
                          text = text.partition(list(filter(None, split))[-1])[-1]

                      if truncate:
                          #Fix truncate escape characters
                          truncate_esc = re.escape(truncate)
                          if prefix and not include_prefix:
                              prefix_esc = re.escape(prefix)
                              pattern = '(?:{})(.*?)(?:{})'.format(prefix_esc,
                                                                  truncate_esc)
                          else:
                              pattern = '(.*?)(?:{})'.format(truncate_esc)

                          trunc_text = re.search(pattern, text, re.S)
                          if trunc_text:
                              text = trunc_text.group(1)

                  if not truncated[i]:
                      gen_text[i] += text.lstrip('\n')
                  #Break the truncation while loop
                  if trunc_text or (length is not None and total_tokens >= length-1):
                      truncated[i] = True

          for gen in gen_text:
              gen = truncate_sentence(gen)
              if destination_path:
                  f.write("{}\n{}".format(gen, sample_delim))
              if not return_as_list and not destination_path:
                  print("{}\n{}".format(gen, sample_delim), end='')
              gen_texts.extend(gen)
              print(type(gen_texts))
              print(len(gen_texts))
          generated += batch_size  

          end = time.time()
          print(end - start)
      addlist.extend(gen_texts)

  with open("pklsave_gpt2.pkl", "rb") as savepkl:
    content = pickle.load(savepkl)
    content.extend(addlist)

  with open("pklsave_gpt2.pkl", "wb") as savepkl:
    pickle.dump(content, savepkl)

  savepkl.close()
  time.sleep(5)
