import os

data_file = 'hard_E=0.1_theta=0-80_10/eps_8.dat'
f = open(data_file, 'r')
lines = f.readlines()
H = []
eps = []
for x in lines:
    try:
        H_n = float(x.split(';')[0])
        e_n = float(x.split(';')[1])  # par
        H.append(H_n)
        eps.append(e_n)
    except:
        ()
#        print('blank line' )
f.close()

print(H, eps)

i = 0
eps_dev = []
while i < len(H) - 1:
    eps_dev.append(abs((eps[i + 1] - eps[i]) / (H[i + 1] - H[i])))
    i = i + 1
print("max_dev is ", max(eps_dev), " at H;eps= ", H[eps_dev.index(max(eps_dev))], ";", eps[eps_dev.index(max(eps_dev))])
f.close()

data = []
path = 'hard_E=0.1_theta=0-80_10/8'
files = os.listdir(path)
for name in files:
    if not name == 'theta_max.csv':
        f = open(path + '/' + name, 'r')
        lines = f.readlines()
        z = []
        theta = []
        for x in lines:
            try:
                z_n = float(x.split(';')[0])
                theta_n = float(x.split(';')[1])  # par
                z.append(z_n)
                theta.append(theta_n)
            except:
                ()
        #        print('blank line' )
        f.close()
        data.append([float(name[5:].split('.dat')[0]), max(theta)])
data.sort(key=lambda x: x[0])
print(data)

f = open(path + '/theta_max.csv', 'w')
f.write('H;theta_max\n')
for x in data:
    f.write(str(x[0]) + ';' + str(x[1]) + '\n')
f.close()
