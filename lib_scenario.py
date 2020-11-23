from abc import ABC, abstractmethod
from pathlib import Path


class Scenario(ABC):
    @abstractmethod
    def generate_training_data(self):
        pass

    @abstractmethod
    def generate_id_testing_data(self):
        pass

    @abstractmethod
    def generate_ood_testing_data(self):
        pass

    @abstractmethod
    def get_num_classes(self):
        pass

    @abstractmethod
    def get_image_size(self):
        pass
