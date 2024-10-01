# %% [markdown]
# # Loading and analysing a single GRN or a population of GRNs

# %%
import numpy as np
import matplotlib.pyplot as plt
import helper

# %%
# Load GRN

# Parameters
rule = 22
id = 1
grn_size = 22
num_cells = 22
dev_steps = 22
geneid = 1 #which gene was used to get fitness

filename = f"results_new_rules/stats_300_{rule}-{rule}_69904-149796_{id}" + "_best_grn.txt"
grns = np.loadtxt(filename)
num_grns = int(grns.shape[0]/(grn_size+2)/grn_size)
grns = grns.reshape(num_grns,grn_size+2,grn_size)

if len(grns.shape) == 2:
    grns = np.expand_dims(grns, axis=0)
print(grns.shape)



# %%
# Get target, phenotype, and fitnesses
seed_int=1024 
#binary of initial condition of fitness gene. 1024 is 1 seed in the middle. other frequently used initial conditions: 69904, 149796
targets, phenos, fitnesses = helper.get_pop_TPF(grns, len(grns), num_cells, grn_size, dev_steps, geneid, rule, seed_int)

print(targets.shape)
print(phenos.shape)
print(fitnesses.shape)

# %%
plt.imshow(targets)
plt.show()
plt.imshow(phenos[-1])
plt.show()
plt.scatter(list(range(len(fitnesses))), fitnesses)
plt.ylabel("Freq")
plt.xlabel("Fitnesses")
plt.show()

# %% [markdown]
# 


