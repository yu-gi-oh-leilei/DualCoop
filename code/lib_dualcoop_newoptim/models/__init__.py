from .resnet import *
from .builder_baseline import build_baseline
# from .builder_dualclip import build_DualCLIP
from .builder_dualclip import build_DualCLIP
from .builder_baselinewitclip import build_ResNet101_CLIP



from .clip import clip
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer