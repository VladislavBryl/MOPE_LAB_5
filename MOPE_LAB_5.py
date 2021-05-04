from sklearn import linear_model
from functools import partial
from scipy.stats import f, t
import math, os, sys
from pyDOE2 import *
import numpy as np
import random
import math

class Experiment:

    def __init__(self, n, m):

        self.n = n
        self.m = m

        self.f1 = self.m - 1
        self.f2 = self.n
        self.f3 = self.f1*self.f2

        self.p = 0.95
        self.q = 1 - self.p

        self.N = [i+1 for i in range(self.n+1)]

        self.ranges = ((-9, 1), (-7, 10), (-1, 2))

        self.av_x_max = sum([x[1] for x in self.ranges]) / 3
        self.av_x_min = sum([x[0] for x in self.ranges]) / 3

        self.min_y = 200+int(self.av_x_min)
        self.max_y = 200+int(self.av_x_max)

        self.x, self.y, self.x_norm = self.matrix(self.min_y, self.max_y, self.ranges, self.n, self.m)

        self.av_y = [round(sum(i) / len(i), 3) for i in self.y]
        self.b = self.koef(self.x, self.av_y)

        self.Result(self.x_norm, self.y, self.b, self.n, self.m, self.f1, self.f2, self.f3, self.q)


    def koef(self, X, Y, norm=False):
        skm = linear_model.LinearRegression(fit_intercept=False)
        skm.fit(X, Y)
        B = skm.coef_

        if norm == 1:
            print('\nКоефіцієнти рівняння регресії з нормованими X:')
        else:
            print('\nКоефіцієнти рівняння регресії:')
        B = [round(i, 3) for i in B]
        print(B)
        print('\nРезультат рівняння зі знайденими коефіцієнтами:\n', np.dot(X, B))
        return B

    def s_kv(self, y, y_aver, n, m):
        res = []
        for i in range(n):
            s = sum([(y_aver[i] - y[i][j]) ** 2 for j in range(m)]) / m
            res.append(round(s, 3))
        return res

    def regression(self, x, b):
            y = sum([x[i] * b[i] for i in range(len(x))])
            return y

    def add_sq_nums(self, x):
            for i in range(len(x)):
                x[i][4] = x[i][1] * x[i][2]
                x[i][5] = x[i][1] * x[i][3]
                x[i][6] = x[i][2] * x[i][3]
                x[i][7] = x[i][1] * x[i][3] * x[i][2]
                x[i][8] = x[i][1] ** 2
                x[i][9] = x[i][2] ** 2
                x[i][10] = x[i][3] ** 2
            return x

    def matrix(self, y_min, y_max, x_range, n, m):

        print("\nРівняння регресії з урахуванням квадратичних членів:")
        print("ŷ = b0 + b1*x1 + b2*x2 + b3*x3 + b12*x1*x2 + b13*x1*x3 + b23*x2*x3 + b123*x1*x2*x3 + b11x1^2 + b22x2^2 + b33x3^2\n")

        print(f'\nМатриця планування для n = {n}, m = {m}')

        y = np.zeros(shape=(n, m))
        for i in range(n):
            for j in range(m):
                y[i][j] = random.randint(y_min, y_max)

        if n > 14:
            no = n - 14
        else:
            no = 1
        x_norm = ccdesign(3, center=(0, no))
        x_norm = np.insert(x_norm, 0, 1, axis=1)

        for i in range(4, 11):
            x_norm = np.insert(x_norm, i, 0, axis=1)

        l = 1.215

        for i in range(len(x_norm)):
            for j in range(len(x_norm[i])):
                if x_norm[i][j] < -1 or x_norm[i][j] > 1:
                    if x_norm[i][j] < 0:
                        x_norm[i][j] = -l
                    else:
                        x_norm[i][j] = l

        x_norm = self.add_sq_nums(x_norm)

        x = np.ones(shape=(len(x_norm), len(x_norm[0])), dtype=np.int64)
        for i in range(8):
            for j in range(1, 4):
                if x_norm[i][j] == -1:
                    x[i][j] = x_range[j - 1][0]
                else:
                    x[i][j] = x_range[j - 1][1]

        for i in range(8, len(x)):
            for j in range(1, 3):
                x[i][j] = (x_range[j - 1][0] + x_range[j - 1][1]) / 2

        dx = [x_range[i][1] - (x_range[i][0] + x_range[i][1]) / 2 for i in range(3)]

        x[8][1] = l * dx[0] + x[9][1]
        x[9][1] = -l * dx[0] + x[9][1]
        x[10][2] = l * dx[1] + x[9][2]
        x[11][2] = -l * dx[1] + x[9][2]
        x[12][3] = l * dx[2] + x[9][3]
        x[13][3] = -l * dx[2] + x[9][3]

        x = self.add_sq_nums(x)

        print('\nX:\n', x)
        print('\nX нормоване:\n')
        for i in x_norm:
            print([round(x, 2) for x in i])
        print('\nY:\n', y)

        return x, y, x_norm

    # -------------------------------------------------------
    # Перевірка однорідності дисперсії за критерієм Кохрена:
    # -------------------------------------------------------
    def kohrenCriteriy(self, y, y_aver, n, m, f1, f2, q):
        S_kv = self.s_kv(y, y_aver, n, m)
        Gp = max(S_kv) / sum(S_kv)
        print('\nПеревірка за критерієм Кохрена')
        return Gp


    def kohren(self, f1, f2, q=0.05):
        q1 = q / f1
        fisher_value = f.ppf(q=1 - q1, dfn=f2, dfd=(f1 - 1) * f2)
        return fisher_value / (fisher_value + f1 - 1)


    def Betas(self, x, y_aver, n):
        res = [sum(1 * y for y in y_aver) / n]

        for i in range(len(x[0])):
            b = sum(j[0] * j[1] for j in zip(x[:, i], y_aver)) / n
            res.append(b)
        return res

    # -------------------------------------------------------
    # Перевірка однорідності дисперсії за критерієм Стьюдента:
    # -------------------------------------------------------
    def studentCriteriy(self, x, y, y_aver, n, m):
        S_kv = self.s_kv(y, y_aver, n, m)
        s_kv_aver = sum(S_kv) / n

        s_Bs = (s_kv_aver / n / m) ** 0.5
        Bs = self.Betas(x, y_aver, n)
        ts = [round(abs(B) / s_Bs, 3) for B in Bs]

        return ts

    # -------------------------------------------------------
    # Перевірка однорідності дисперсії за критерієм Фішера:
    # -------------------------------------------------------
    def fisherCriteriy(self, y, y_aver, y_new, n, m, d):
        S_ad = m / (n - d) * sum([(y_new[i] - y_aver[i]) ** 2 for i in range(len(y))])
        S_kv = self.s_kv(y, y_aver, n, m)
        S_kv_aver = sum(S_kv) / n

        return S_ad / S_kv_aver

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Вивід даних:
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def Result(self, X, Y, B, n, m, f1, f2, f3, q):

        student = partial(t.ppf, q=1 - q)
        t_student = student(df=f3)

        G_kr = self.kohren(f1, f2)

        y_aver = [round(sum(i) / len(i), 3) for i in Y]
        print('\nСереднє значення y:', y_aver)

        disp = self.s_kv(Y, y_aver, n, m)
        print('Дисперсія y:', disp)

        Gp = self.kohrenCriteriy(Y, y_aver, n, m, f1, f2, q)
        print(f'Gp = {Gp}')
        if Gp < G_kr:
            print(f'З ймовірністю {1 - q} дисперсії однорідні.')
        else:
            print("Необхідно збільшити кількість дослідів")
            m += 1
            Experiment(n, m)

        ts = self.studentCriteriy(X[:, 1:], Y, y_aver, n, m)
        print('\nКритерій Стьюдента:\n', ts)
        res = [t for t in ts if t > t_student]
        res1 = [0]*11
        final_k = [B[i] for i in range(len(ts)) if ts[i] in res]
        insignificant_k = [B[i] for i in range(len(ts)) if ts[i] not in res]
        print('\nКоефіцієнти {} статистично незначущі, тому ми виключаємо їх з рівняння.'.format(
            [round(i, 3) for i in B if i not in final_k]))

        for i in range (len(ts)):
            if B[i] in insignificant_k:
                res1[i]=B[i]
            else:
                res1[i]=0
        y = '{} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3  + {}*x2x3 + {}*x1x2x3 + {}*x1^2 + {}*x2^2 + {}*x3^2'
        print("Рівняння регресії з незначущими коефіцієнтами: \n", y.format(*res1))


        y_new = []
        for j in range(n):
            y_new.append(self.regression([X[j][i] for i in range(len(ts)) if ts[i] in res], final_k))

        print(f'\nЗначення "y" з коефіцієнтами {final_k}')
        print(y_new)

        d = len(res)
        if d >= n:
            print('\nF4 <= 0')
            print('')
            return
        f4 = n - d

        F_p = self.fisherCriteriy(Y, y_aver, y_new, n, m, d)

        fisher = partial(f.ppf, q=0.95)
        f_t = fisher(dfn=f4, dfd=f3)
        print('\nПеревірка адекватності за критерієм Фішера')
        print('Fp =', F_p)
        print('F_t =', f_t)
        if F_p < f_t:
            print('Математична модель адекватна експериментальним даним')
        else:
            print('Математична модель не адекватна експериментальним даним \nНеобхідно збільшити кількість дослідів')

if __name__ == '__main__':
    Experiment(15, 3)