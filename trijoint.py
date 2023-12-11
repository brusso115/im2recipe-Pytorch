import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchwordemb
from args import get_parser

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# # =============================================================================

class TableModule(nn.Module):
    def __init__(self):
        super(TableModule, self).__init__()
        
    def forward(self, x, dim):
        y = torch.cat(x, dim)
        return y

def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)


class CustomViT(nn.Module):
    def __init__(self, weights='DEFAULT'):
        super(CustomViT, self).__init__()
        self.vit = models.vit_b_16(weights=weights)
    
    def forward(self, img):
        feats = self.vit._process_input(img)

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(img.shape[0], -1, -1)
        feats = torch.cat([batch_class_token, feats], dim=1)

        feats = self.vit.encoder(feats)

        # We're only interested in the representation of the classifier token that we appended at position 0
        feats = feats[:, 0]

        return feats

    

# Im2recipe model
class im2recipe(nn.Module):
    def __init__(self):
        super(im2recipe, self).__init__()
        if opts.preModel=='resNet50':
            print('using resNet50')
            resnet = models.resnet50(pretrained=True)
            modules = list(resnet.children())[:-1]  # we do not use the last fc layer.
            self.visionMLP = nn.Sequential(*modules)

            self.visual_embedding = nn.Sequential(
                nn.Linear(opts.imfeatDim, opts.embDim),
                nn.Tanh(),
            )
            
            self.recipe1 = nn.Linear(opts.instrDim + opts.ingrDim, opts.embDim)
            #torch.nn.init.xavier_uniform(self.recipe1.weight)
            #self.bn1 = nn.BatchNorm1d(512)
            #self.recipe2 = nn.Linear(512, opts.embDim)
            #torch.nn.init.xavier_uniform(self.recipe2.weight)
            #self.bn2 = nn.BatchNorm1d(opts.embDim)
            self.tanh = nn.Tanh()
            

        elif opts.preModel=='ViT':
            print('using ViT')
            self.visionMLP = CustomViT()

            imfeatDim = 768
            self.visual_embedding = nn.Sequential(
                nn.Linear(imfeatDim, opts.embDim),
                nn.Tanh(),
            )

            self.recipe1 = nn.Linear(opts.instrDim + opts.ingrDim, opts.embDim)
            #torch.nn.init.xavier_uniform(self.recipe1.weight)
            #self.bn1 = nn.BatchNorm1d(512)
            #self.recipe2 = nn.Linear(512, opts.embDim)
            #torch.nn.init.xavier_uniform(self.recipe2.weight)
            #self.bn2 = nn.BatchNorm1d(opts.embDim)
            self.tanh = nn.Tanh()

        else:
            raise Exception('Only resNet50 and ViT model is implemented.')

        self.table      = TableModule()
 
        if opts.semantic_reg:
            self.semantic_branch = nn.Linear(opts.embDim, opts.numClasses)

    def forward(self, x, y1, z1): # we need to check how the input is going to be provided to the model
        # recipe embedding

        if y1.isnan().any():
            print(y1)

        if z1.isnan().any():
            print(z1)
        
        recipe_emb = torch.cat((y1, z1), dim=1)
        recipe_emb = self.recipe1(recipe_emb)
        #recipe_emb = self.bn1(recipe_emb)
        #recipe_emb = self.recipe2(recipe_emb)
        #recipe_emb = self.bn2(recipe_emb)
        recipe_emb = self.tanh(recipe_emb)
        recipe_emb = norm(recipe_emb)

        # visual embedding
        visual_emb = self.visionMLP(x)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        visual_emb = self.visual_embedding(visual_emb)
        visual_emb = norm(visual_emb)
        
        if opts.semantic_reg:            
            visual_sem = self.semantic_branch(visual_emb)
            recipe_sem = self.semantic_branch(recipe_emb)
            # final output
            if recipe_sem.isnan().any():
                print('Recipe semantics has nans')

            if recipe_emb.isnan().any():
                print('Recipe embedding has nans')
            output = [visual_emb, recipe_emb, visual_sem, recipe_sem]
        else:
            # final output 
            output = [visual_emb, recipe_emb] 
        return output 



