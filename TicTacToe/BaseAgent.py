from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(
        self,
        size: int,
        lr: float = 1e-3,
        epsilon: float = 0.1,
        gamma: float = 0.99,
        name: str = "Random"
    ):
        self.size = size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.name = name

    # abstract methods
    @abstractmethod
    def load_model(self, path_load):
        pass

    # abstract methods
    @abstractmethod
    def save_model(self, path_save):
        pass

    # abstract methods
    @abstractmethod
    def make_move(self):
        pass

    # abstract methods
    @abstractmethod
    def train(self):
        pass
