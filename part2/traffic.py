import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy as sp

def sq_error(sq_x, sq_y, f_x=None):
    """
    Вычисление среднеквадратичной ошибки
    """
    squared_error = []
    for i in range(len(sq_x)):
        squared_error.append((f_x(sq_x[i]) - sq_y[i])**2)
    return sum(squared_error)

data = pd.read_csv('web_traffic.tsv',sep='\t', header = None) 
X, Y = data[0], data[1]
x = list(X)
y = list(Y)

for i in range(len(y)): #убираем nan
    if math.isnan(y[i]): #isnan(x)-проверяет, x -цифра или нет
        y[i] = 0
    
np_x = np.array(x) 
np_y = np.array(y)

#вычисляем коэффициенты для разных степеней полинома
th0, th1 = np.polyfit(np_x, np_y, 1)
th2, th3, th4 = np.polyfit(np_x, np_y, 2)
th5, th6, th7, th8 = np.polyfit(np_x, np_y, 3)
th9, th10, th11, th12, th13 = np.polyfit(np_x, np_y, 4)
th14, th15, th16, th17, th18, th19 = np.polyfit(np_x, np_y, 5)

f1 = lambda x: th0*x + th1
f2 = lambda x: th2*x**2 + th3*x + th4
f3 = lambda x: th5*x**3 + th6*x**2 + th7*x + th8
f4 = lambda x: th9*x**4 + th10*x**3 + th11*x**2 + th12*x + th13
f5 = lambda x: th14*x**5 + th15*x**4 + th16*x**3 + th17*x**2 + th18*x + th19

result_1 = sq_error(x, y, f1)
result_2 = sq_error(x, y, f2)
result_3 = sq_error(x, y, f3)
result_4 = sq_error(x, y, f4)
result_5 = sq_error(x, y, f5)

print("Среднее квадратичное отклонение: ", result_1)
print("Среднее квадратичное отклонение: ", result_2)
print("Среднее квадратичное отклонение: ", result_3)
print("Среднее квадратичное отклонение: ", result_4)
print("Среднее квадратичное отклонение: ", result_5)

x1 = list(range(743))

plt.plot(x, y, 'b*')
f1 = sp.poly1d(np.polyfit(np_x, np_y, 1)) #poly1d - построение полинома
plt.plot(x1, f1(x1), label = u'полином 1-ой степени')
f2 = sp.poly1d(np.polyfit(np_x, np_y, 2))
plt.plot(x1, f2(x1), label = u' полином 2-ой степени')
f3 = sp.poly1d(np.polyfit(np_x, np_y, 3))
plt.plot(x1, f3(x1), label = u'полином 3-ей степени')
f4 = sp.poly1d(np.polyfit(np_x, np_y, 4))
plt.plot(x1, f4(x1), label = u'полином 4-ой степени')
f5 = sp.poly1d(np.polyfit(np_x, np_y, 5))
plt.plot(x1, f5(x1), linewidth = 3, label = u'полином 5-ой степени')
plt.legend()
plt.ylabel('y')
plt.xlabel('x')
plt.show()
plt.savefig('plot1.png')
