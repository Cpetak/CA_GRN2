import numpy as np
import matplotlib.pyplot as plt

m=np.loadtxt("./results_testing_CA_GRN2/stats_env_seeded_5_102-94_1024-1024_0_targets.txt")
print(m.shape)
m=m.reshape(2,23,22)
print(m[0].shape)
plt.imshow(m[0])
plt.show()