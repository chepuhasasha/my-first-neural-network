import numpy
import scipy.special


class NN:
    def __init__(self, lr, config=[3, 3, 3]):
        self.actvation = lambda x: scipy.special.expit(x)  # сигмойда
        self.lr = lr  # коэф. обучения
        self.network = []
        
        for i, item in enumerate(config):  # создание слоев
            lvl = { 'id': i, 'in': [], 'out': [], 'err':[] }
            if (i + 1) < len(config):
                lvl['w'] = numpy.random.normal(
                    0.0, pow(config[i + 1], -0.5), (config[i + 1], item))
            self.network.append(lvl)
        pass

n = NN(0.3)
