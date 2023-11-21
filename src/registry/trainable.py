from src.abstract import ProtoTrainable


class TrainableRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = {}
        return cls._instance

    def register(self, cls):
        self._registry[cls.__name__] = cls
        return cls

    def get(self, class_name):
        """Get the class from the registry."""
        if class_name in self._instance._registry:
            return self._instance._registry[class_name]
        raise KeyError(f"No class named '{class_name}' is registered.")

    def get_registered(self):
        """return registered classes."""
        if self._instance._registry:
            return self._instance._registry
        else:
            raise RuntimeError("No classes are registered.")


# Singleton instance of the registry that will be used by the decorator
trainable_global_registry = TrainableRegistry()


def register_trainable(cls):
    if issubclass(cls, ProtoTrainable):
        trainable_global_registry.register(cls)
    return cls
