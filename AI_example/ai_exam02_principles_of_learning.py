# -*- coding: utf-8 -*-
"""AI_exam02_principles_of_learning.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VgzQtalyegj3Jxuy6HhzzLSl-RtHWr02

'그래프 연산'
"""

class add_graph:
  def __init__(self):
    pass
  def forward(self, x, y):
    out = x + y
    return out
  def backward(self, dout):
    dx = dout * 1 # 덧셈 미분은 1...?
    dy = dout * 1
    return dx, dy

class mul_graph:
  def __init__(self):
    self.x = None
    self.y = None
  def forward(self, x, y):
    self.x = x
    self.y = y
    out = x * y
    return out
  def backward(self, dout):
     dx = dout * self.y
     dy = dout * self.x
     return dx, dy

class mse_graph:
  def __init__(self):
    self.x = None
    self.loss = None
    self.t = None # target: 정답
    self.y = None
  def forward(self, y, t):
    self.t = t
    self.y = y
    self.loss = np.mean((self.y - self.t) ** 2) # 오차의 제곱들로 평균 구하기
    return self.loss
  def backward(self, x, dout = 1): # weight와 bias의 미분값 구하기
    data_size = self.t.shape[0]
    dweight_mse = (((self.y - self.t) * x).sum() * 2) / data_size
    dbias_mse = (self.y - self.t).sum() * 2 / data_size
    return dweight_mse, dbias_mse

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

mul_apple_graph = mul_graph()
mul_orange_graph = mul_graph()
add_apple_orange_graph = add_graph()
mul_tax_graph = mul_graph()

apple_price = mul_apple_graph.forward(apple, apple_num)
orange_price = mul_orange_graph.forward(orange, orange_num)
all_price = add_apple_orange_graph.forward(apple_price, orange_price)
total_price = mul_tax_graph.forward(all_price, tax)
print(total_price)

dprice = 1 # total_price에 대한 미분값
dall_price, dtax = mul_tax_graph.backward(dprice)
dapple_price, dorange_price =add_apple_orange_graph.backward(dall_price)
dorange, dorange_num = mul_orange_graph.backward(dorange_price)
dapple, dapple_num = mul_apple_graph.backward(dapple_price)
print(dapple, dapple_num, dorange, dorange_num, dtax)

"""실습해 보기~"""

import numpy as np
def celsius_to_fahrenheit(x):
  return 9 / 5 * x + 32

weight = np.random.uniform(0, 5, 1)
print(weight)
bias = 0
data_C = np.arange(0, 100)
data_F = celsius_to_fahrenheit(data_C)
scaled_data_C = data_C / 100
scaled_data_F = data_F / 100
print(scaled_data_C)
print(scaled_data_F)

#bias 그래프와 weight 그래프를 생성
weight_graph = mul_graph()
bias_graph = add_graph()

weighted_data_C = weight_graph.forward(weight, scaled_data_C)
predict_data = bias_graph.forward(weighted_data_C, bias)
print(predict_data)

dout = 1 # backward 함수에 넣을 인자값을 최초로 1로 설정
dbias, dweighted_data = bias_graph.backward(dout)
print(dbias, dweighted_data)
dweight, dscaled_data = weight_graph.backward(dweighted_data)
print(dweight)

# MSE 구하기
mseGraph = mse_graph()
mse = mseGraph.forward(predict_data, scaled_data_F)
print(mse)

weight_mse_gradient, bias_mse_gradient = mseGraph.backward(scaled_data_C)
print(weight_mse_gradient)
print(bias_mse_gradient)

learning_rate = 0.01

learned_weight = weight - learning_rate * weight_mse_gradient * np.average(dweight)
print(weight)
print(learned_weight)

learned_bias = bias - learning_rate * bias_mse_gradient * np.average(dbias)
print(bias)
print(learned_bias)

learning_rate = 0.1
error_list = []
weight_list = []
bias_list = []
for i in range(1000):
  #forward
  weighted_data_C = weight_graph.forward(weight, scaled_data_C)
  predict_data = bias_graph.forward(weighted_data_C, bias)
  #backward
  dout = 1
  dbias, dweighted_data = bias_graph.backward(dout)
  dweight, dscaled_data = weight_graph.backward(dweighted_data)
  #mse
  mse = mseGraph.forward(predict_data, scaled_data_F)
  error_list.append(mse)
  #mse gradient
  weight_mse_gradient, bias_mse_gradient = mseGraph.backward(scaled_data_C)
  #learning
  weight = weight - learning_rate * weight_mse_gradient * np.average(dweight)
  weight_list.append(weight)
  bias = bias - learning_rate * bias_mse_gradient * np.average(dbias)
  bias_list.append(bias)

print(weight)
print(bias)

import matplotlib.pyplot as plt

plt.plot(error_list)
plt.show()

plt.plot(weight_list)
plt.show()

plt.plot(bias_list)
plt.show()