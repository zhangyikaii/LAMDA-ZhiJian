import torch.nn as nn


class vit_base_patch16_224_hook(nn.Module):
    def __init__(self):
        super(vit_base_patch16_224_hook, self).__init__()
        self.feature_cache = None
        self.layer_feature_cache = []

        self.main_info = {
            'get_feature': {
                'name': 'vit_base_patch16_224_hook',
                'location': [['fc_norm']],
                'hook': [['get_pre', 'pre']]
                },
            'get_layer_feature': {
                'name': 'vit_base_patch16_224_hook',
                'location': [
                    ['blocks',0],
                    ['blocks',0], ['blocks',1], ['blocks',2], ['blocks',3],
                    ['blocks',4], ['blocks',5], ['blocks',6], ['blocks',7],
                    ['blocks',8], ['blocks',9], ['blocks',10], ['blocks',11]
                    ],
                'hook': [
                    ['clear_layer_feature', 'pre'],
                    ['get_layer_post', 'post'], ['get_layer_post', 'post'], ['get_layer_post', 'post'], ['get_layer_post', 'post'],
                    ['get_layer_post', 'post'], ['get_layer_post', 'post'], ['get_layer_post', 'post'], ['get_layer_post', 'post'],
                    ['get_layer_post', 'post'], ['get_layer_post', 'post'], ['get_layer_post', 'post'], ['get_layer_post', 'post']
                    ]
                }
            }

    def get_pre(self, module, inputs):
        self.feature_cache = inputs[0]
        return inputs[0]

    def get_feature(self):
        return self.feature_cache

    def get_feature_hook_info(self):
        return [self.main_info['get_feature']]

    def get_layer_post(self, module, inputs, outputs):
        self.layer_feature_cache.append(outputs.detach())

    def get_layer_feature(self):
        return self.layer_feature_cache

    def get_layer_feature_hook_info(self):
        return [self.main_info['get_layer_feature']]

    def clear_layer_feature(self, module, inputs):
        self.layer_feature_cache.clear()

    def get_log_str(self):
        return ''

HOOKS = {
    'vit_base_patch16_224_in21k': vit_base_patch16_224_hook
}
