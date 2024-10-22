# %% [markdown]
# # Loading and analysing a single GRN or a population of GRNs

# %%
import numpy as np
import matplotlib.pyplot as plt
import helper
import os

# %%
# Load GRN

# Parameters
rule = 102
grn_size = 22
num_cells = 22
dev_steps = 22
geneid = 1 #which gene was used to get fitness
num_reps = 10

root="/Users/csengepetak/best_grn_results/"

vari_grns=[np.loadtxt(os.path.expanduser(root+f"stats_300_{rule}-{rule}_{69904}-{149796}_{i+1}_best_grn.txt")) for i in range(num_reps)]
temp_grns = []
for grn in vari_grns:
    num_grns = int(grn.shape[0]/(grn_size+2)/grn_size)
    new_grn = grn.reshape(num_grns,grn_size+2,grn_size)
    new_grn = new_grn[:9899,:,:] #standardize num generations
    temp_grns.append(new_grn)
vari_grns=np.array(temp_grns)
print(vari_grns.shape) # repeats, generations, gene matrix dim1 and dim2

#%%

static1_grns=[np.loadtxt(os.path.expanduser(root+f"stats_100000_{rule}_69904_{i+1}_best_grn.txt")) for i in range(num_reps)]
temp_grns = []
for grn in static1_grns:
    num_grns = int(grn.shape[0]/(grn_size+2)/grn_size)
    new_grn = grn.reshape(num_grns,grn_size+2,grn_size)
    new_grn = new_grn[:9899,:,:] #standardize num generations
    temp_grns.append(new_grn)
static1_grns=np.array(temp_grns)
print(static1_grns.shape)

static2_grns=[np.loadtxt(os.path.expanduser(root+f"stats_100000_{rule}_149796_{i+1}_best_grn.txt")) for i in range(num_reps)]
temp_grns = []
for grn in static2_grns:
    num_grns = int(grn.shape[0]/(grn_size+2)/grn_size)
    new_grn = grn.reshape(num_grns,grn_size+2,grn_size)
    new_grn = new_grn[:9899,:,:] #standardize num generations
    temp_grns.append(new_grn)
static2_grns=np.array(temp_grns)
print(static2_grns.shape)


#if len(grns.shape) == 2:
    #grns = np.expand_dims(grns, axis=0)
#print(grns.shape)



# %%
# Get target, phenotype, and fitnesses
seed_int=69904 
#binary of initial condition of fitness gene. 1024 is 1 seed in the middle. other frequently used initial conditions: 69904, 149796
targets, phenos, fitnesses = helper.get_pop_TPF(vari_grns, len(vari_grns), num_cells, grn_size, dev_steps, geneid, rule, seed_int_target = seed_int, seed_int_dev = seed_int)

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


