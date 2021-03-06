import numpy
import scipy.special
import matplotlib.pyplot as plt

class NN:
    def __init__(self, lr, config=[3, 3, 3]):
        self.actvation = lambda x: scipy.special.expit(x)  # сигмойда
        self.lr = lr  # коэф. обучения
        self.network = []

        for i, item in enumerate(config):  # создание слоев
            lvl = {'id': i, 'in': [], 'out': [], 'err': []}
            if (i + 1) < len(config):
                lvl['w'] = numpy.random.normal(
                    0.0, pow(config[i + 1], -0.5), (config[i + 1], item))
            self.network.append(lvl)
        pass

    def query(self, input):  # опрос сети
        for i, lvl in enumerate(self.network):
            if i == 0:  # входной слой
                in_lvl = numpy.array(input, ndmin=2).T
                lvl['in'] = in_lvl
                lvl['out'] = in_lvl
            else:
                in_lvl = numpy.dot(
                    self.network[i-1]['w'], self.network[i-1]['out'])
                lvl['in'] = in_lvl
                lvl['out'] = self.actvation(in_lvl)
        return self.network

    def update(self, error, out, out_prev):  # дефференцированая функция градиентного спуска
        return self.lr * numpy.dot((e * out * (1.0 - out)), numpy.transpose(out_prev))

    def train(self, input, target):
        self.query(input)  # состояние сети
        t = numpy.array(target, ndmin=2).T,  # приведение к двумерному массиву
        out = self.network[-1]['out']  # выход с последнего слоя
        # обратное распростронение ошибки
        for i, lvl in reversed(list(enumerate(self.network))):
            if i == (len(self.network) - 1):
                lvl['err'] = pow(((t - out)[0]), 2)  # функция выходной ошибки
            else:
                lvl['err'] = numpy.dot(lvl['w'].T, self.network[i+1]['err'])
        pass

    def show(self):
        print('NETWORK')
        for lvl in self.network:
            print('lavel:', lvl['id'], '―――――――――――――')
            for i in range(len(lvl['in'])):
                line = '⊕  '+str(i)+'   i▼:' + \
                    str(lvl['in'][i]) + ' o▲:' + str(lvl['out'][i]) + \
                    ' e☒: ' + str(lvl['err'][i])
                print(line)
                if 'w' in lvl:
                    for k, w in enumerate(lvl['w']):
                        print('🠗 '+ str(i) + str(k) +':  '+ str(w[i]))


# n = NN(0.3)
# n.train([1, 0.8, 0.4], [1, 0.5, 0.5])
# n.show()


data_file = open("./mnist_train_100.csv", 'r') 
data_list = data_file.readlines() 
data_file.close()


all_values = data_list[1].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28,28)) 
plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.show()
# scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 
# print(scaled_input)