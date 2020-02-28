import unittest

from scipy.stats import nbinom

import openquake.hme.utils.stats as ss

p = [0.01, 0.2, 0.5]
q = [0.33, 0.22, 0.44]

kld = ss.kullback_leibler_divergence(p, q)

jsdiv = ss.jensen_shannon_divergence(p, q)

jsdist = ss.jensen_shannon_distance(p, q)

nb = nbinom(5, 0.5)

rvs = nb.rvs(1000)

nb_ps = ss.estimate_negative_binom_parameters(rvs)

print(nb_ps)