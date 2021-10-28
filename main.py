import numpy
import scipy.special


class neuralNetwark:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # задание количества узлов на разных слоях
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

        # коэффициент обучения
        self.lr = learning_rate

        # матрицы весовых коэффициентов
        self.wih = numpy.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = numpy.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

        # использование сигмойды в качестве функции активации
        self.activation_func = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, target_list):
        # преобразовать список входных значений в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        # расчет входяших сигналов для скрытого слоя
        h_inputs = numpy.dot(self.who, inputs)
        # расчет исходяших сигналов для скрытого слоя
        h_outputs = self.activation_func(h_inputs)

        # расчет входяших сигналов для выходного слоя
        final_inputs = numpy.dot(self.who, h_outputs)
        # расчет исходяших сигналов для выходного слоя
        final_outputs = self.activation_func(final_inputs)

        # ошибки выходного слоя = (целевое - фактическое)
        output_errors = targets - final_outputs
        # ошибки скрытого слоя - это outputs_errors,
        # распределенные прпорционально весовым коэфф. связей
        # и рекомбенированные на скрытых узлах
        h_errors = numpy.dot(self.who.T, output_errors)

        # обновление весовых коэфф. для связей между входными и скрытыми слоями
        self.who += self.lr * numpy.dot((output_errors * final_outputs *
                                        (1.0 - final_outputs)), numpy.transpose(h_outputs))

        # обновление весовых коэфф. для связей между входными и скрытыми слоями
        self.wih += self.lr * numpy.dot((h_errors * h_outputs * (1.0 - h_outputs)), numpy.transpose(inputs))

        pass

    def query(self, inputs_list):
        # преобразование списка входных значений в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T

        # расчет входяших сигналов для скрытого слоя
        h_inputs = numpy.dot(self.who, inputs)
        # расчет исходяших сигналов для скрытого слоя
        h_outputs = self.activation_func(h_inputs)

        # расчет входяших сигналов для выходного слоя
        final_inputs = numpy.dot(self.who, h_outputs)
        # расчет исходяших сигналов для выходного слоя
        final_outputs = self.activation_func(final_inputs)

        return final_outputs


n = neuralNetwark(3, 3, 3, 0.3)
print(n.query([1.0, 0.5, -1.5]))
