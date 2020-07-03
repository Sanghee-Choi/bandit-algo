from abc import ABC, abstractmethod

class BanditAlgorithm(ABC):
    @abstractmethod
    def __init__(self):
        pass 

    @abstractmethod
    def select_arm(self):
        pass 

    @abstractmethod
    def play(self):
        pass
    
    @abstractmethod
    def update(self):
        pass

