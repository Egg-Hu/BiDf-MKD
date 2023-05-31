import torch
from abc import ABC, abstractclassmethod
from typing import Dict

class BaseSynthesis(ABC):
    def __init__(self, teacher, student):
        super(BaseSynthesis, self).__init__()
        self.teacher = teacher
        self.student = student
    
    @abstractclassmethod
    def synthesize(self) -> Dict[str, torch.Tensor]:
        """ take several steps to synthesize new images and return an image dict for visualization. 
            Returned images should be normalized to [0, 1].
        """
        pass