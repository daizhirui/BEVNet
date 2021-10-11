class AutoConvertDescriptor:
    def __init__(self, target_cls):
        self.target_cls = target_cls

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        return vars(instance)[self.name]

    def __set__(self, instance, value):
        if isinstance(value, dict):
            vars(instance)[self.name] = self.target_cls(value)
        else:
            vars(instance)[self.name] = value
