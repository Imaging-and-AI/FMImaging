# Testing TODO s
## Norms 2D vs 3D
## weight decay vs no_w_decay

import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from model_base.attention_modules import *

print("all done")
