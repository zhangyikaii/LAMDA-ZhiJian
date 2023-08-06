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
