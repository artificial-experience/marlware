from .registration import ConstructRegistry


class BaseConstruct:
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ConstructRegistry.register_construct(cls.__name__, cls)
