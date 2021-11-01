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
                'out': self.actvation(in_lvl),
                'w': w
            })
        return network

    # def update(self, e, out, out_prev):
    #     return self.lr * numpy.dot((e * out * (1.0 - out)), numpy.transpose(out_prev))

    def train(self, inputs, target):
        network = self.query(inputs) # состояние сети
        t = numpy.array(target, ndmin=2).T, # приведение к двумерному массиву
        out = network[-1]['out'] # выход с последнего слоя
        error = pow((t - out), 2) # функция ошибки
        # обратное распростронение ошибки
        for lvl in reversed(network):
            e = numpy.dot(lvl['w'].T, error)
            lvl['w'] += self.update(error, lvl['out'], )

            error = e


n = NN(0.3, [3, 3, 3])
n.query([10, 10, 10])