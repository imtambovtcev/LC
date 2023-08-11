import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def size_pair(x):
    ny = 0
    while (x[ny] == x[0]):
        ny += 1
    return (int(len(x) / ny), ny)


def plot(s):
    file = open(s)
    file = file.read()
    file = file.split('\n')
    file = [i for i in file if 'K1' not in i]
    file = [i.strip().split(';') for i in file if i != '']
    file = [[float(number) for number in line] for line in file]
    x = np.array([i[0] for i in file]) * 10 ** 6
    y = np.array([i[1] for i in file]) * 10 ** 6
    z = np.array([i[2] for i in file]) * 100
    (nx, ny) = size_pair(x)
    print(nx, ny)
    x = x.reshape([nx, ny])
    y = y.reshape([nx, ny])
    z = z.reshape([nx, ny])
    levels = MaxNLocator(nbins=20).tick_values(z.min(), z.max())
    print(levels)
    plt.contourf(x, y, z, levels=levels, cmap=cm.gist_stern)
    plt.colorbar()
    plt.rc('text', usetex=True)
    plt.xlabel('$K_1$, $10^{-6}$dyn')
    plt.ylabel('$K_3$, $10^{-6}$dyn')
    #    plt.ticklabel_format(style='sci',useOffset=True, useLocale=True, useMathText=True)
    #    plt.show()
    plt.savefig('plot.eps')
    plt.savefig('plot.pdf')
    plt.savefig('plot.png')


plot('map.csv')
