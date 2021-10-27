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
  def train():
    pass
  def query():
    pass

n = neuralNetwark(3, 3, 3, 0.3)