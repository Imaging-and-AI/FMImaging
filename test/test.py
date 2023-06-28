# Testing TODO s
## Norms 2D vs 3D
## striding conv for temp att

import sys
from pathlib import Path

import torchmetrics
import torch

import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils import *
from model_base.losses import *
from utils.save_model import save_final_model
from utils.running_inference import running_inference

preds = torch.rand([3, 3, 256, 256], generator=torch.manual_seed(42))
target = preds * 0.75

# msssim_loss = torchmetrics.functional.multiscale_structural_similarity_index_measure(preds, target, reduction=None)
# print(msssim_loss)

device = 'cuda'

for k in range(1, 10):
    msssim_loss = torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure(kernel_size=7, reduction=None, data_range=None)
    msssim_loss.to(device=device)
    v = msssim_loss(preds.to(device) * k, target.to(device))
    print(v)
    
print("-----------------------")

for k in range(1, 10):
    msssim_loss = torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure(kernel_size=7, reduction=None, data_range=None)
    msssim_loss.to(device=device)
    v = msssim_loss(preds.to(device) * k, target.to(device) * k)
    print(v)

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from model_base.attention_modules import *

print("all done")
