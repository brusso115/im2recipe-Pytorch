
import time
import torch
import torch.nn as nn
import torch.nn.parallel
# import torch.optim
# import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
from data_loader import ImagerLoader # our data_loader
import numpy as np
from trijoint import im2recipe
import pickle
from args import get_parser
from PIL import Image
import sys
import os
import random
# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================


if not(torch.cuda.device_count()):
    device = torch.device(*('cpu',0))
else:
    torch.cuda.manual_seed(opts.seed)
    device = torch.device(*('cuda',0))

torch.manual_seed(opts.seed)
random.seed(opts.seed)
np.random.seed(opts.seed)

def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)

def image2embedding(im_path=None):
    ext = os.path.basename(im_path).split('.')[-1]
    if ext not in ['jpeg','jpg','png']:
        raise Exception("Wrong image format.")

    # create model
    model = im2recipe()
    model.visionMLP = torch.nn.DataParallel(model.visionMLP)
    model.to(device)

    # load checkpoint
    print("=> loading checkpoint '{}'".format(opts.model_path))
    if device.type=='cpu':
        checkpoint = torch.load(opts.model_path, encoding='latin1', map_location='cpu')
    else:
        checkpoint = torch.load(opts.model_path, encoding='latin1')
    opts.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(opts.model_path, checkpoint['epoch']))

    # data preparation, loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
                transforms.Resize(256), # rescale the image keeping the original aspect ratio
                transforms.CenterCrop(224), # we get only the center of that rescaled
                transforms.ToTensor(),
                normalize])

    # load image
    im = Image.open(im_path).convert('RGB')
    im = transform(im)
    im = im.view((1,)+im.shape)

    # get model output
    visual_emb = model.visionMLP(im)
    visual_emb = visual_emb.view(visual_emb.size(0), -1)
    visual_emb = model.visual_embedding(visual_emb)
    visual_emb = norm(visual_emb)
    visual_emb = visual_emb.data.cpu().numpy().flatten()
    return visual_emb

def find_closest_idx(visual_emb, instr_vecs):
    sims = np.dot(visual_emb, instr_vecs.T)
    closest_idx = np.argmax(sims)
    print(f'Found closest recipe at index {closest_idx} with similarity {sims[closest_idx]}')
    return closest_idx

def convert_ingredients_to_string(ingredients):
    vocab_file = 'data/text/vocab.txt'
    with open(vocab_file, 'r') as f:
        vocab = f.read().splitlines()
        vocab = ['', '</s>'] + vocab

    ingredients_str = []
    for token in ingredients:
        ingredients_str.append(vocab[token])
    return ingredients_str

def tensor_to_image(tensor):
    undo_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                        std=[1/0.229, 1/0.224, 1/0.225])
    tensor = undo_normalize(tensor) * 255
    image = transforms.ToPILImage()(tensor.byte())
    return image

def main(im_path, save_path):
    visual_emb = image2embedding(im_path)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_dataset = ImagerLoader(opts.img_path,
 	    transforms.Compose([
            transforms.Resize(256), # rescale the image keeping the original aspect ratio
            transforms.CenterCrop(224), # we get only the center of that rescaled
            transforms.ToTensor(),
            normalize,
        ]),data_path=opts.data_path,sem_reg=opts.semantic_reg,partition='test')


    print("Loading results from {}".format(opts.path_results))
    with open(os.path.join(opts.path_results,'rec_embeds.pkl'),'rb') as f:
        instr_vecs = pickle.load(f)
        
    closest_idx = find_closest_idx(visual_emb, instr_vecs)
    x, y = test_dataset[closest_idx]
    closest_img_tens = x[0]
    closest_img = tensor_to_image(closest_img_tens)
    closest_ingredients_num = x[3]
    closest_ingredients_str = convert_ingredients_to_string(closest_ingredients_num)

    # Save images
    os.makedirs(save_path, exist_ok=True)
    closest_img.save(os.path.join(save_path, 'closest_img.jpg'))

    img = Image.open(im_path)
    img.save(os.path.join(save_path, 'original_img.jpg'))


    # Save ingredients
    with open(os.path.join(save_path, 'closest_ingredients.txt'), 'w') as f:
        f.write('\n'.join(closest_ingredients_str))


if __name__ == '__main__':
    img_path1 = 'data/images/val/0/0/0/2/0002694171.jpg' # Salmon with tomatoes?
    img_path2 = 'data/images/val/0/0/0/3/0003f7346e.jpg' # Cinnamon roll
    img_path3 = 'data/images/val/0/0/1/8/0018e60166.jpg' # Ham cheese avocado sandwich with soup

    selected_img = img_path2

    model_name = 'my_model'
    img_name = os.path.basename(selected_img).split('.')[0]
    save_path = os.path.join('retrieval_results', model_name, img_name)

    main(selected_img, save_path)