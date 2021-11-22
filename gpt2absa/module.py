import numpy as np
import tensorflow as tf
from collections import OrderedDict
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')

restaurant2014_aspect_categories = ['ambience', 'anecdotes/miscellaneous', 'food', 'price', 'service']
sentiment_candidates = [' positive', ' negative', ' neutral', ' conflict']
sentiment_map = OrderedDict()
for candidates in sentiment_candidates:
    sentiment_map[candidates[1:]] = tokenizer.encode((candidates if candidates != ' conflict' else ' conflicting'))[0]

def get_aspects(review):
  return list(review['sentiment'].keys())

def get_text(review):
  texts = []
  for aspect, sentiment in review['sentiment'].items():
    text = review['text']
    
    aspect_tokens = aspect.split('#')
    
    if len(aspect_tokens) == 1:
      text += ' My opinion of the ' + (aspect_tokens[0] if aspect_tokens[0] != 'anecdotes/miscellaneous' else 'restaurant') + ' is very ' + (sentiment if sentiment != 'conflict' else 'conflicting')
    else:
      text += ' My opinion of the ' + aspect_tokens[0] + ' ' + (aspect_tokens[1] if aspect_tokens[1] != 'general' else 'in general') + ' is very ' + (sentiment if sentiment != 'conflict' else 'conflicting')
    texts.append(text)
  return texts

def predict_sentiments_from_aspects(model, text, aspects):
  sentiment_each_aspect = OrderedDict()
  for aspect in aspects:
    aspect_tokens = aspect.split('#')

    if len(aspect_tokens) == 1:
      aspect_text = text + ' My opinion of the ' + (aspect_tokens[0] if aspect_tokens[0] != 'anecdotes/miscellaneous' else 'restaurant') + ' is very'
    else:
      aspect_text = text + ' My opinion of the ' + aspect_tokens[0] + ' ' + (aspect_tokens[1] if aspect_tokens[1] != 'general' else 'in general') + ' is very'
    encoded_aspect_text = tokenizer.encode(aspect_text, return_tensors='tf')
    model_output = model(encoded_aspect_text)
    next_logits = model_output.logits[:, -1, :]
    next_logits = tf.keras.activations.softmax(next_logits)[0]
    sentiment_values = OrderedDict()
    for sentiment, ids in sentiment_map.items():
      sentiment_values[sentiment] = float(next_logits[ids])
    softmaxed_output = tf.keras.activations.softmax(tf.constant([list(sentiment_values.values())]), axis=-1)[0]
    for idx, sentiment in enumerate(sentiment_values.keys()):
      sentiment_values[sentiment] = float(softmaxed_output[idx])
    sentiment_each_aspect[aspect] = sentiment_values
  return sentiment_each_aspect

def generatecandidaatetree(categories):
  aspect_tokens = [cat.split('#') for cat in categories]

  treelistkey = set()
  for pair in aspect_tokens:
    treelistkey.add(pair[0])

  treelist = dict()
  for key in treelistkey:
    treelist[key] = []

  if len(aspect_tokens[0]) > 1:
    for pair in aspect_tokens:
      treelist[pair[0]].append(pair[1])

  return treelist

def get_aspect_categories(model, text, categories, treshold = 0.15):
  text = text + ' My opinion of the'
  treelist = generatecandidaatetree(categories)

  rootids = OrderedDict()
  rootscore = OrderedDict()
  for root, branch in treelist.items():
    rootids[root] = tokenizer.encode(' ' + (root if root != 'anecdotes/miscellaneous' else 'restaurant'))[0] 

  encoded = tokenizer.encode(text, return_tensors='tf')
  logits = model(encoded).logits[:, -1, :]
  logits = tf.keras.activations.softmax(logits)[0]

  for aspect, ids in rootids.items():
    rootscore[aspect] = float(logits[ids])

  softmaxed_output = tf.keras.activations.softmax(tf.constant([list(rootscore.values())]), axis=-1)[0]

  for idx, sentiment in enumerate(rootids.keys()):
      rootids[sentiment] = float(softmaxed_output[idx])

  categorycandidates = OrderedDict()

  if (len(categories[0].split('#')) > 1):
    for root, brances in treelist.items():
      text = text + ' My opinion of the ' + root 
      encoded = tokenizer.encode(text, return_tensors='tf')
      blogits = model(encoded).logits[:, -1, :]
      blogits = tf.keras.activations.softmax(blogits)[0]

      for branch in brances:
        branchid = tokenizer.encode(' ' + (branch if branch != 'general' else 'in'))[0]
        categorycandidates[root + '#' + branch] = (rootscore[root] * float(blogits[branchid])) ** (1.0 / 2.0)
  else:
    for root, score in rootscore.items():
      categorycandidates[root] = score

  categorycandidates = OrderedDict(sorted(categorycandidates.items(), key=lambda item: item[1], reverse=True))
  # print(categorycandidates)

  finalcategory = OrderedDict()

  for key, value in categorycandidates.items():
    if (value < treshold):
      break
    finalcategory[key] = [value]

  return list(finalcategory.keys())

## API ##
def sentiment_polarity_classification(model, text, aspects):
  output = predict_sentiments_from_aspects(model, text, aspects)
  sentiment_polarities = dict()
  for aspect, probability_vector_map in output.items():
    sentiment_polarities[aspect] = list(probability_vector_map.keys())[np.argmax(list(probability_vector_map.values()))]
  return sentiment_polarities

def aspect_polarity_pair(model, text, categories, treshold = 0.15):
  aspects = get_aspect_categories(model, text, categories, treshold)
  return sentiment_polarity_classification(model, text, aspects)
