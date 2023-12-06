# !/usr/bin/env python
import random
import pickle
import numpy as np
from proc import *
from tqdm import *
import torchfile
import time
import utils
import os
# from ..args import get_parser
import time
import lmdb
import shutil
import sys
sys.path.append("..")
from args import get_parser
from transformers import DistilBertTokenizer, TFDistilBertModel

# Maxim number of images we want to use per recipe
maxNumImgs = 5

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

DATASET = opts.dataset

# don't use this file once dataset is clean
with open('remove1M.txt','r') as f:
    remove_ids = {w.rstrip(): i for i, w in enumerate(f)}

print('Loading dataset.')
# print DATASET
dataset = utils.Layer.merge([utils.Layer.L1, utils.Layer.L2, utils.Layer.INGRS],DATASET)

with open('classes1M.pkl','rb') as f:
    class_dict = pickle.load(f)
    id2class = pickle.load(f)

st_ptr = 0
numfailed = 0

if os.path.isdir('../data/train_lmdb_trans_txt_embs'):
    shutil.rmtree('../data/train_lmdb_trans_txt_embs')
if os.path.isdir('../data/val_lmdb_txt_embs'):
    shutil.rmtree('../data/val_lmdb_txt_embs')
if os.path.isdir('../data/test_lmdb_txt_embs'):
    shutil.rmtree('../data/test_lmdb_txt_embs')

env = {'train' : [], 'val':[], 'test':[]}
env['train'] = lmdb.open('../data/train_lmdb_txt_embs',map_size=int(1e11))
env['val']   = lmdb.open('../data/val_lmdb_txt_embs',map_size=int(1e11))
env['test']  = lmdb.open('../data/test_lmdb_txt_embs',map_size=int(1e11))

print('Assembling dataset.')
img_ids = dict()
keys = {'train' : [], 'val':[], 'test':[]}

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
MAX_LEN = 512

for i,entry in tqdm(enumerate(dataset)):

    print(i)

    ninstrs = len(entry['instructions'])
    instructions_concat = ' '.join(list(map(lambda x: x['text'], entry['instructions'])))
    ningrs = len(entry['ingredients'])
    ingredients_concat = ' '.join(list(map(lambda x: x['text'], entry['ingredients'])))
    imgs = entry.get('images')

    if ninstrs >= opts.maxlen or ningrs >= opts.maxlen or ningrs == 0 or not imgs or remove_ids.get(entry['id']):
        continue

    instructions_enc = tokenizer.encode_plus(
            text=instructions_concat,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
            )
        
    ingredients_enc = tokenizer.encode_plus(
        text=ingredients_concat,  # Preprocess sentence
        add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
        max_length=MAX_LEN,                  # Max length to truncate/pad
        pad_to_max_length=True,         # Pad sentence to max length
        return_tensors='pt',           # Return PyTorch tensor
        return_attention_mask=True      # Return attention mask
        )

    instructions_enc_ids = instructions_enc.get('input_ids')
    instructions_enc_mask = instructions_enc.get('attention_mask')

    ingredients_enc_ids = ingredients_enc.get('input_ids')
    ingredients_enc_mask = ingredients_enc.get('attention_mask')

    instructions_enc_ids = np.array(instructions_enc_ids)
    instructions_enc_mask = np.array(instructions_enc_mask)
    ingredients_enc_ids = np.array(ingredients_enc_ids)
    ingredients_enc_mask = np.array(ingredients_enc_mask)

    outputs_instructions = model(input_ids=instructions_enc_ids, attention_mask=instructions_enc_mask)
    outputs_ingredients = model(input_ids=ingredients_enc_ids, attention_mask=ingredients_enc_mask)

    partition = entry['partition']

    serialized_sample = pickle.dumps( {'ingrs':outputs_ingredients[0][:, 0, :], 'intrs':outputs_instructions[0][:, 0, :],
        'classes':class_dict[entry['id']]+1, 'imgs':imgs[:maxNumImgs]} ) 

    with env[partition].begin(write=True) as txn:
        txn.put('{}'.format(entry['id']).encode('latin1'), serialized_sample)
    # keys to be saved in a pickle file    
    keys[partition].append(entry['id'])

for k in keys.keys():
    with open('../data/{}_keys.pkl'.format(k),'wb') as f:
        pickle.dump(keys[k],f)

print('Training samples: %d - Validation samples: %d - Testing samples: %d' % (len(keys['train']),len(keys['val']),len(keys['test'])))


