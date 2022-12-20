from functools import partial
from .dino import DINO, DINOCollateFunction
from .swav import SwaV, SwaVCollateFunction
from .simsiam import SimSiam, SimCLRCollateFunction
from .mae import MAE, MAECollateFunction

class FrameworkFactory:
    def __init__(self):
        self._models = {}
        self._collate_fns = {}

    def register_model(self, key, model):
        self._models[key] = model
    
    def register_collate_fn(self, key, collate_fn):
        self._collate_fns[key] = collate_fn

    def create(self, key, **kwargs):
        model_cls = self._models.get(key)
        collate_fn = self._collate_fns.get(key)
        if not model_cls or not collate_fn:
            raise ValueError(key)
        return model_cls, collate_fn

framework_factory = FrameworkFactory()

framework_factory.register_model('dino', DINO)
framework_factory.register_collate_fn('dino', DINOCollateFunction(
                                                random_gray_scale=1.0, 
                                                global_crop_scale=(0.7, 1.0), 
                                                local_crop_scale=(0.35, 0.7),
                                                vf_prob=0.5, 
                                                cj_prob=0, 
                                                cj_hue=0,
                                                # normalize={'mean': [0, 0, 0], 'std': [1, 1, 1]}
                                            ))

framework_factory.register_model('swav', SwaV)
framework_factory.register_collate_fn('swav', SwaVCollateFunction(
                                                random_gray_scale=0.2, 
                                                crop_max_scales=[1.0, 0.7], 
                                                crop_min_scales=[0.7, 0.35],
                                                vf_prob=0.5, 
                                                cj_prob=0, 
                                                # normalize={'mean': [0, 0, 0], 'std': [1, 1, 1]}
                                            ))

framework_factory.register_model('simsiam', SimSiam)
framework_factory.register_collate_fn('simsiam', SimCLRCollateFunction(
                                                cj_prob=0, 
                                                min_scale=0.35, 
                                                random_gray_scale=0.2, 
                                                vf_prob=0.5,
                                                # normalize={'mean': [0, 0, 0], 'std': [1, 1, 1]}
                                            ))
