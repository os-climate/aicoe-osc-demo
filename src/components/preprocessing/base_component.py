"""BaseComponent."""

from abc import ABC, abstractmethod


class BaseComponent(ABC):
    """BaseComponent class."""

    def __init__(self, name="Base"):
        """Initialize BaseComponent class."""
        self.name = name

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run method."""
        pass
