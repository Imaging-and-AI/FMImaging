# Testing TODO s
## Norms 2D vs 3D
## striding conv for temp att

import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from model_base.attention_modules import *

print("all done")
