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

    def query(self, input): # опрос сети
        for i, lvl in enumerate(self.network):
            if i == 0: #входной слой
                in_lvl = numpy.array(input, ndmin=2).T
                lvl['in'] = in_lvl
                lvl['out'] = in_lvl
            else:
                in_lvl = numpy.dot(self.network[i-1]['w'], self.network[i-1]['out'])
                lvl['in'] = in_lvl
                lvl['out'] = self.actvation(in_lvl)
        return self.network

    def update(self, error, out, out_prev): # дефференцированая функция градиентного спуска
        return self.lr * numpy.dot((e * out * (1.0 - out)), numpy.transpose(out_prev))

    def train(self, input, target):
        pass

n = NN(0.3)
print(n.query([10,10,10]))
