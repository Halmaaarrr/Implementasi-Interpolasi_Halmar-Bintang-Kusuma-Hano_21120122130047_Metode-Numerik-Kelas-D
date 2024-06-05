import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([5, 10, 15, 20, 25, 30, 35, 40])
y_data = np.array([40, 30, 25, 40, 18, 20, 22, 15])

# Implementasi Interpolasi Lagrange
def lagrange_interpolation(x, x_data, y_data):
    n = len(x_data)
    result = 0
    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if i != j:
                term = term * (x - x_data[j]) / (x_data[i] - x_data[j])
        result += term
    return result

# Implementasi Interpolasi Newton
def divided_differences(x_data, y_data):
    n = len(x_data)
    coef = np.zeros([n, n])
    coef[:,0] = y_data
    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (x_data[i + j] - x_data[i])
    return coef[0, :]

def newton_interpolation(x, x_data, coef):
    n = len(x_data)
    result = coef[0]
    product_term = 1.0
    for i in range(1, n):
        product_term *= (x - x_data[i - 1])
        result += coef[i] * product_term
    return result

coef_newton = divided_differences(x_data, y_data)

x_range = np.linspace(5, 40, 100)
y_lagrange = [lagrange_interpolation(x, x_data, y_data) for x in x_range]
y_newton = [newton_interpolation(x, x_data, coef_newton) for x in x_range]

plt.plot(x_data, y_data, 'o', label='Data Points')
plt.plot(x_range, y_lagrange, '-', label='Lagrange Interpolation', color='orange', linewidth=2)
plt.plot(x_range, y_newton, '--', label='Newton Interpolation', color='green', linewidth=2)
plt.xlabel('Tegangan, x (kg/mm^2)')
plt.ylabel('Waktu patah, y (jam)')
plt.legend()
plt.title('Interpolasi Polinomial')
plt.grid(True)
plt.show()
