import random as rd
import numpy as np

f = open("sub.csv", "w+")
f.write('%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (
    'MSISDN', 'ARPU', 'RVE_DATA', 'VOL_DATA', 'RVE_VOICE', 'VOL_VOICE', 'CNT_VOICE', 'RVE_SMS', 'CNT_SMS'))
for i in range(400000):
    MSISDN = '84' + str(rd.randint(0, 100000000000))
    if rd.random() < 0.93:
        RVE_DATA = 0
    else:
        RVE_DATA = rd.randint(0, 4000000000)
    if rd.random() < 0.67:
        VOL_DATA = 0
    else:
        VOL_DATA = rd.randint(0, 20000000000)
    if rd.random() < 0.47:
        RVE_VOICE = 0
    else:
        RVE_VOICE = rd.randint(0, 3000000000)
    if rd.random() < 0.2:
        VOL_VOICE = 0
    else:
        VOL_VOICE = rd.randint(0, 40000)
    if rd.random() < 0.2:
        CNT_VOICE = 0
    else:
        CNT_VOICE = rd.randint(0, 300)
    if rd.random() < 0.82:
        RVE_SMS = 0
    else:
        RVE_SMS = rd.randint(0, 5000000000)
    if rd.random() < 0.74:
        CNT_SMS = 0
    else:
        CNT_SMS = rd.randint(0, 600)
    ARPU = RVE_DATA + RVE_VOICE + RVE_SMS
    f.write('%s,%d,%d,%d,%d,%d,%d,%d,%d\n' % (
        MSISDN, ARPU, RVE_DATA, VOL_DATA, RVE_VOICE, VOL_VOICE, CNT_VOICE, RVE_SMS, CNT_SMS))

# check probability element 0
data = np.genfromtxt("sub.csv", dtype='int64', delimiter=',', skip_header=1)
for i in range(1, 9):
    print(len(np.where(data[:, i] == 0)[0]) / data.shape[0])
