import unittest

from openquake import hme

import matplotlib.pyplot as plt

pac_mfd = {
    6.0: 7.594815959543139,
    6.2: 4.908342634097348,
    6.4: 3.195431033099892,
    6.6: 2.0956026089211166,
    6.8: 1.3843462018353891,
    7.0: 0.9210287041112523,
    7.2: 0.617019271413135,
    7.4: 0.41610051634947565,
    7.6: 0.2814524225587865,
    7.8: 0.19089531037149302,
    8.0: 0.12894965672429431,
    8.2: 0.07132001553282985,
    8.4: 0.030827238820692084,
    8.6: 0.011717643023081543,
    8.8: 0.00358292910653494,
    9.0: 0.001155663538693543,
    9.2: 0.0001001167484667318
}

stoch_mfds = hme.utils.plots._make_stoch_mfds(pac_mfd, 500, t_yrs=40.)

f = plt.figure()
ax = f.add_subplot(111, yscale='log')

for smfd in stoch_mfds:
    ax.plot(list(pac_mfd.keys()), smfd, lw=0.02)

ax.plot(list(pac_mfd.keys()), list(pac_mfd.values()))

plt.show()