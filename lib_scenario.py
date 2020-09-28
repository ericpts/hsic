class Scenario(object):
    def generate_training_data(self):
        raise NotImplemented("Must implement method in derived class!")

    def generate_id_testing_data(self):
        raise NotImplemented("Must implement method in derived class!")

    def generate_ood_testing_data(self):
        raise NotImplemented("Must implement method in derived class!")
