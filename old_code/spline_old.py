import math

import numpy as np


def ini(n, b, max):
    x = np.linspace(0., 1., n, endpoint=True)
    return (max - b) * np.sin(np.pi * x) + b


class Spline:
    def __init__(self, x=[0.0, 0.5, 1.0], f=[1.0], bound=[0.0, 0.0]):
        self.n = len(x)
        self.x = list(x)
        self.bound = list(bound)
        self.a = [None] * self.n
        self.b = [None] * self.n
        self.c = [None] * self.n
        self.d = [None] * self.n
        self.reset_spline(f)

    def get_new_point(self):
        coeff = 0.0
        x_n = 1
        for i in range(1, self.n):
            cur_coeff = (math.fabs(self.b[i - 1]) + math.fabs(self.b[i])) * (self.x[i] - self.x[i - 1])
            # print 'cur_coeff = ',cur_coeff,' at x = ', self.x[i]
            if cur_coeff > coeff:
                coeff = cur_coeff
                x_n = i
        # print 'new point number = ', x_n
        return (x_n, (self.x[x_n] + self.x[x_n - 1]) * 0.5)

    def get_value(self, x, n=False):
        if n == False:
            n = 0
            while x > self.x[n]: n = n + 1
        h = x - self.x[n]
        # print self.a[n], self.b[n]*h, 0.5*self.c[n]*h*h,self.d[n]*h*h*h/6
        return self.a[n] + self.b[n] * h + 0.5 * self.c[n] * h * h + self.d[n] * h * h * h / 6

    def get_dev(self, x, n=False):
        if n == False:
            n = 0
            while x > self.x[n]: n = n + 1
        h = x - self.x[n];
        # print self.a[n], self.b[n]*h, 0.5*self.c[n]*h*h,self.d[n]*h*h*h/6
        return self.b[n] + self.c[n] * h + self.d[n] * h * h * 0.5

    def get_ddev(self, x, n=False):
        if n == False:
            n = 0
            while x > self.x[n]: n = n + 1
        h = x - self.x[n];
        # print self.a[n], self.b[n]*h, 0.5*self.c[n]*h*h,self.d[n]*h*h*h/6
        return self.c[n] + self.d[n] * h

    def set_new_point(self, point_n, point_x):
        i = 0
        # print 'x\ta\tb\tc\td'
        while i < self.n:
            # print	"%0.2f" % self.x[i],'\t',"%0.2f" %self.a[i],'\t',"%0.2f" % self.b[i],'\t',"%0.2f" % self.c[i],'\t',"%0.2f" % self.d[i]
            i = i + 1
        h = point_x - self.x[point_n];
        y = self.a[point_n] + self.b[point_n] * h + 0.5 * self.c[point_n] * h * h + self.d[point_n] * h * h * h / 6.0;
        # print 'new point number is ', point_n, ' at ',  point_x, ' with ', y
        self.x.insert(point_n, point_x)
        self.a.insert(point_n, y)
        h_i1 = self.x[point_n + 1] - self.x[point_n];
        h_i = self.x[point_n] - self.x[point_n - 1];
        self.c.insert(point_n, (6.0 * ((self.a[point_n + 1] - self.a[point_n]) / h_i1 - (
            self.a[point_n] - self.a[point_n - 1]) / h_i) - h_i1 * self.c[point_n] - h_i * self.c[
                                    point_n - 1]) / (2.0 * (h_i + h_i1)));
        self.d.insert(point_n, (self.c[point_n] - self.c[point_n - 1]) / h_i)
        self.b.insert(point_n, (self.a[point_n] - self.a[point_n - 1]) / h_i + h_i * (
            2.0 * self.c[point_n] + self.c[point_n - 1]) / 6.0)
        self.n = self.n + 1
        # print self.x,self.a,self.b,self.c,self.d

    def reset_spline(self, f):
        self.a = list(f)
        self.a[0] = self.bound[0]
        self.a[len(self.a) - 1] = self.bound[1]
        M = np.zeros((self.n, self.n))
        r = np.zeros(self.n)
        h = [0.0]
        for i in range(1, self.n):
            h.append(self.x[i] - self.x[i - 1])
        # print h
        M[(0, 0)] = 1
        r[0] = 0
        M[(self.n - 1, self.n - 1)] = 1
        r[self.n - 1] = 0
        for i in range(1, self.n - 1):
            r[i] = 6.0 * ((f[i + 1] - f[i]) / h[i + 1] - (f[i] - f[i - 1]) / h[i])
            M[(i, i - 1)] = h[i]
            M[(i, i)] = 2.0 * (h[i] + h[i + 1])
            M[(i, i + 1)] = h[i + 1]
        # print M
        # print r
        self.c = np.linalg.solve(M, r).tolist()
        for i in range(1, self.n):
            self.d[i] = (self.c[i] - self.c[i - 1]) / h[i]
            self.b[i] = (f[i] - f[i - 1]) / h[i] + h[i] * (2 * self.c[i] + self.c[i - 1]) / 6
        self.b[0] = self.b[1] - self.c[1] * h[1] + 0.5 * self.d[1] * h[1] * h[1]
        self.d[0] = self.d[1]

    # print 'x = ',self.x
    # print 'a = ',self.a
    # print 'b = ',self.b
    # print 'c = ',self.c
    # print 'd = ',self.d
    # print 'x = ', self.x[self.n-1], '\ta =', self.a[self.n-1], '\tb = ', self.b[self.n-1], '\tc = ', self.c[self.n-1], '\td = ', self.d[self.n-1]
    # self.graph(self.F,self.n)
    # raw_input()

    def graph(self, F_name=False):
        if F_name == False:
            f = open('graph/graph.dat', 'w')
        else:
            f = open(F_name, 'w')
        f.write('z;theta\n')
        i = 0
        while i <= 100:
            f.write(str(0.01 * i) + ';' + str(self.get_value(0.01 * i)) + '\n')
            i = i + 1
        f.close()


class functional(Spline):

    def __init__(self, x, f, bound=[0.0, 0.0], F=1.0, K=1.0, K_as=0.0, min_points=100, g_type='perp'):
        Spline.__init__(self, x, f, bound)
        self.F = F
        self.K = K
        self.K_1 = K * (1.0 + K_as)
        self.K_3 = K * (1.0 - K_as)
        self.min_points = min_points
        self.g_type = g_type

    @property
    # def min_value(self):
    #     if self.type == 'perp':
    #         summ = 0.0
    #         for k in range(1, self.min_points):
    #             i = float(k) / float(self.min_points)
    #             s = math.sin(self.get_value(i))
    #             c = math.cos(self.get_value(i))
    #             dd = self.get_ddev(i)
    #             d = self.get_dev(i)
    #             su=(self.K_1 * s * s + self.K_3 * c * c) * dd + (
    #                     self.K_3 - self.K_1) * d * d * s * c + (self.F-1.0) * s * c/self.K
    #             summ = summ + su*su
    #         return summ/float(self.min_points)
    #     elif self.type == 'par':
    #         summ = 0.0
    #         for k in range(1, self.min_points):
    #             i = float(k) / float(self.min_points)
    #             s = math.sin(self.get_value(i))
    #             c = math.cos(self.get_value(i))
    #             dd = self.get_ddev(i)
    #             d = self.get_dev(i)
    #             su = (self.K_1 * s * s + self.K_3 * c * c) * dd + (
    #                     self.K_3 - self.K_1) * d * d * s * c + (self.F + 1.0) * s * c/self.K
    #             summ = summ + su * su
    #             return summ / float(self.min_points)
    #     else:
    #         return -1
    #
    def min_value(self):
        # print('K1 ', self.K_1)
        # print('K3 ', self.K_3)
        # print('F ', self.F)
        if self.g_type == 'perp':
            summ = 0.0
            for k in range(1, self.min_points):
                i = float(k) / float(self.min_points)
                s = math.sin(self.get_value(i))
                c = math.cos(self.get_value(i))
                d = self.get_dev(i)
                summ = summ + 0.5 * ((self.K_1 * s * s + self.K_3 * c * c) * d * d - (self.F * s * s + c * c))
            # print('E = ', summ)
            return summ
        elif self.g_type == 'par':
            summ = 0.0
            for k in range(1, self.min_points):
                i = float(k) / float(self.min_points)
                s = math.sin(self.get_value(i))
                c = math.cos(self.get_value(i))
                d = self.get_dev(i)
                summ = summ + 0.5 * (self.K_1 * s * s + self.K_3 * c * c) * d * d - 0.5 * (self.F + 1.0) * c * c
            # print(self.get_value(0.5),summ)
            return summ
        else:
            return -1

    def wp(self, x):
        f = []
        f.append(self.bound[0])
        if type(x) == np.float64 or type(x) == float:
            f.append(x)
        else:
            f.extend(x.tolist())
        f.append(self.bound[1])
        self.reset_spline(f)
        # if self.F>100.0:
        # print f, self.min_value()
        # raw_input()
        return self.min_value  # self.min_value #self.free_energy
