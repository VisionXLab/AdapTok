from .models import register, make, models
from . import transformer
from . import bottleneck
from . import loss
from . import adaptok_ar
from . import gptc
from . import adaptok


def get_model_cls(name):
    return models[name]