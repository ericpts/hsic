from abc import ABC, abstractmethod
import tensorflow as tf


class Scenario(ABC):
    @abstractmethod
    def generate_training_data(self) -> tf.data.Dataset:
        pass

    @abstractmethod
    def generate_id_testing_data(self) -> tf.data.Dataset:
        pass

    @abstractmethod
    def generate_ood_testing_data(self) -> tf.data.Dataset:
        pass

    @abstractmethod
    def get_num_classes(self) -> int:
        pass

    @abstractmethod
    def get_input_size(self):
        pass
