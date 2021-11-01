import numpy
import scipy.special


class NN:
    def __init__(self, lr, config=[3, 3, 3]):
        self.actvation = lambda x: scipy.special.expit(x)  # сигмойда
        self.lr = lr  # коэф. обучения
        self.W = []  # веса
        for i, lvl in enumerate(config):  # матрицы весовых коэф.
            if (i + 1) < len(config):
                w = numpy.random.normal(
                    0.0, pow(config[i + 1], -0.5), (config[i + 1], lvl))
                self.W.append(w)
        pass

    def query(self, inputs):  # опрос сети
        in_nn = numpy.array(inputs, ndmin=2).T
        network = [
            # входной слой
            {
                'in': in_nn,
                'out': in_nn
            }
        ]
        for i, w in enumerate(self.W):
            in_lvl = numpy.dot(w, network[i]['out'])
            network.append({
                'in': in_lvl,
                'out': self.actvation(in_lvl)
            })
        print(network)
        return network


n = NN(0.3, [3, 3, 3])
n.query([10, 10, 10])
