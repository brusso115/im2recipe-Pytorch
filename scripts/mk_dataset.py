# !/usr/bin/env python
import random
import pickle
import numpy as np
from tqdm import *
import time
import utils
import os
# from ..args import get_parser
import time
import lmdb
import shutil
import sys
import torch
sys.path.append("..")
#from args import get_parser
#from transformers import DistilBertTokenizer, TFDistilBertModel

# Maxim number of images we want to use per recipe
maxNumImgs = 5

# =============================================================================
#parser = get_parser()
#opts = parser.parse_args()
# =============================================================================

DATASET = '../data/recipe1M'

# don't use this file once dataset is clean
with open('remove1M.txt','r') as f:
    remove_ids = {w.rstrip(): i for i, w in enumerate(f)}

print('Loading dataset.')
# print DATASET
dataset = utils.Layer.merge([utils.Layer.L1, utils.Layer.L2, utils.Layer.INGRS],DATASET)
#torch.save(dataset, '../data/dataset.pt')
#dataset = torch.load('../data/dataset.pt')

with open('../data/classes1M.pkl','rb') as f:
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

with open('../data/instruction_stranf_embeddings.pkl', 'rb') as handle:
    instr_emb_dict = pickle.load(handle)
    
with open('../data/ingredients_stranf_embeddings.pkl', 'rb') as handle:
    ingr_emb_dict = pickle.load(handle)

for i,entry in tqdm(enumerate(dataset)):

    print(i)

    ninstrs = len(entry['instructions'])
    instructions_concat = ' '.join(list(map(lambda x: x['text'], entry['instructions'])))
    ningrs = len(entry['ingredients'])
    ingredients_concat = ' '.join(list(map(lambda x: x['text'], entry['ingredients'])))
    imgs = entry.get('images')

    if ninstrs >= 20 or ningrs >= 20 or ningrs == 0 or not imgs or remove_ids.get(entry['id']):
        continue

    partition = entry['partition']
    
    serialized_sample = pickle.dumps( {'ingrs':ingr_emb_dict[entry['id']], 'intrs':instr_emb_dict[entry['id']],
        'classes':class_dict[entry['id']]+1, 'imgs':imgs[:maxNumImgs]} )

    with env[partition].begin(write=True) as txn:
        txn.put('{}'.format(entry['id']).encode('latin1'), serialized_sample)
    # keys to be saved in a pickle file
    keys[partition].append(entry['id'])

for k in keys.keys():
    with open('../data/{}_keys_txt_embs.pkl'.format(k),'wb') as f:
        pickle.dump(keys[k],f)

print('Training samples: %d - Validation samples: %d - Testing samples: %d' % (len(keys['train']),len(keys['val']),len(keys['test'])))


