def singleton(cls):
    class SingletonWrapper(cls):
        _instance = None
        _init_flag = False

        def __new__(cls, *args, **kwargs):
            if not cls._instance:
                cls._instance = super(SingletonWrapper, cls).__new__(cls)
                cls._instance.__class__ = cls
            return cls._instance

        def __init__(self, *args, **kwargs):
            if not self._init_flag:
                super(SingletonWrapper, self).__init__(*args, **kwargs)
                self._init_flag = True

    SingletonWrapper.__name__ = cls.__name__
    SingletonWrapper.__doc__ = cls.__doc__
    return SingletonWrapper
