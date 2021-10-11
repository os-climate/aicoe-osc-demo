from abc import ABC, abstractmethod


class BaseComponent(ABC):
    def __init__(self, name="Base"):
        self.name = name

    @abstractmethod
    def run(self, *args, **kwargs):
        pass
