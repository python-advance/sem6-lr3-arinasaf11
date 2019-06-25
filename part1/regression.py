import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

data = pd.read_csv('ex1data1.csv', header = None) 
x, y = data[0], data[1] #присваиваем x -первую колонку данных, y-вторую

x1, y1 = [0, 25], [-10, 40 ] #строим прямую y=2x-10

def gradient_descent(x, y, k, n):
    """
    Функция с алгоритмом ручного градиентного спуска
    """   
    m = len(x)
    theta0, theta1 = 0, 0
    for i in range(n):
        sum1 = 0
        for i in range(m):
            sum1 += theta0 + theta1 * x[i] - y[i]
        res1 = theta0 - k * (1 / m) * sum1
        sum2 = 0
        for i in range(m):
            sum2 += (theta0 + theta1 * x[i] - y[i]) * x[i]
        res2 = theta1 - k * (1 / m) * sum2

        theta0, theta1 = res1, res2

    return theta0, theta1

x2 = [1, 25]
y2 = [0, 0]
t0, t1 = gradient_descent(x, y, 0.001, len(x)) #приравниваем найденные коэффициенты
y2[0] = t0 + x2[0] * t1
y2[1] = t0 + x2[1] * t1


np_x = np.array(x) #для дальнейшей передачи в полифит
np_y = np.array(y)
np_t1, np_t0 = np.polyfit(np_x, np_y, 1) #p = polyfit(x, y, n) находит коэффициенты полинома p(x) степени n, который аппроксимирует функцию y(x) в смысле метода наименьших квадратов. 

num_y1 = [0, 0]
num_y1[0] = np_t0 + x1[0] * np_t1
num_y1[1] = np_t0 + x1[1] * np_t1


print('Коэффициенты равны', np_t0, np_t1)

fig = plt.plot(x, y, 'g*') #координаты из файла
fig1 = plt.plot(x1, y1, 'y', label = u'Прямая y=2x-10') #прямая y=2x-10
plt.plot(x2, y2, 'r', label = u'Коэффициенты, найденные вручную')#строим прямую с новыми коэффициентами
plt.plot(x1, num_y1, 'b', label = u'Коэффициенты, найденные автоматически')

plt.title('Линейная регрессия с одной переменной.')
plt.legend()
plt.ylabel('y')
plt.xlabel('x')
plt.show()
plt.savefig('plot.png')
