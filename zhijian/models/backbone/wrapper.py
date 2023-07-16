class ModelWrapper():
    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        if name == 'model':
            return super().__getattribute__('model')
        else:
            return getattr(self.model, name)

    def __setattr__(self, name, value):
        if name == 'model':
            super().__setattr__(name, value)
        else:
            setattr(self.model, name, value)

    def __call__(self, *args, **kwargs):
        if hasattr(self, 'forward'):
            return self.forward(*args, **kwargs)
        return self.model(*args, **kwargs)

    def __str__(self):
        model_info = self.model.__str__()
        return model_info

    def reuse_callback(self, outputs):
        if hasattr(self.model, 'reuse_callback'):
            return self.model.reuse_callback()
        return outputs

    def input_callback(self, input):
        if hasattr(self.model, 'input_callback'):
            return self.model.input_callback()
        return input

    #     # 在linear层之后添加"OK"字符串
    #     linear_layer_index = -1
    #     for idx, module in enumerate(self.modules()):
    #         if isinstance(module, torch.nn.Linear):
    #             linear_layer_index = idx

    #     if linear_layer_index != -1:
    #         model_info = model_info.replace('\n', f' --> OK\n', linear_layer_index)

    #     return model_info

