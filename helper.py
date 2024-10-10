import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from matplotlib.lines import Line2D
from numba import njit, prange
from pathlib import Path
import seaborn as sns
#import torch
import hashlib
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import pandas as pd
import math
import os

ALPHA = 10

def prepare_run(folder_name):
    
    folder = Path(folder_name)
    folder.mkdir(parents=True, exist_ok=True)

    return folder

def map_to_range(value):
  return int(hashlib.sha256(str(value).encode()).hexdigest(), 16) % (2**32)

def calculate_distance(x1, y1, x2, y2):
    # Calculate the distance using the distance formula
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def calc_div_BH(fitnesses1, fitnesses2, tdf):
    '''
    fitnesses1 is a list of fitnesses in env A for all individuals
    fitnesses2 is a list of fitnesses in env B for all individuals
    tdf is a dataframe of values defining points along the rectange of possibilities
    '''
    dists = [] #low value, close by
    fits = list(zip(fitnesses1, fitnesses2))
    for f in fits: #for each individual
        d_A=calculate_distance(tdf.iloc[0]["x"], tdf.iloc[0]["y"], f[0], f[1]) # distance to target A
        d_B=calculate_distance(tdf.iloc[1]["x"], tdf.iloc[1]["y"], f[0], f[1]) # distance to target B
        dists.append(min(d_A, d_B))

    max_distance=calculate_distance(tdf.iloc[1]["x"],tdf.iloc[1]["y"],tdf.iloc[6]["x"],tdf.iloc[6]["y"])
    print(max_distance)
    dists = np.array(dists)
    print(dists)
    dists = 1-dists/max_distance # if distance is 0, it is 1 (max), if distance is 1, which is the biggest, it is 0 (min)
    print(dists)
    dists = dists ** 2 #make it nonlinear to punish for really bad fitness, away from any target
    print(dists)
    dists = np.mean(dists) #so range: 0-1 and the bigger the better - the closer it is to one of the targets
    print(dists)
    print("end of dists --------")

    stds = (np.std(fitnesses1) + np.std(fitnesses2)) /2
    print(stds)
    max_stds = (np.std([tdf.iloc[0]["x"], tdf.iloc[1]["x"]]) + np.std([tdf.iloc[0]["y"], tdf.iloc[1]["y"]])) /2
    tresholded_stds = min(stds, max_stds) # greater std than max_std is not needed to be a perfect diversifier
    f_stds = tresholded_stds/max_stds #what percentage of max this is, so range 0-1. 1 = as diverse as it can be, the bigger the better
    print(f_stds)

    div_BH = (f_stds + dists) / 2 #averaged so that is it between 0 and 1
    
    return div_BH

def calc_pheno_variation(p, children_locs, num_child, parent_locs, dev_steps, num_cells, where_overlap, where_no_overlap):
    child_phenotypes = p[children_locs] 
    # inner most list: first: first born of each parent, second: second borns of each parent, etc.
    # so it is NOT all kids of 1 parent, then the other parent, etc.
    reshaped=np.reshape(child_phenotypes, (num_child, len(parent_locs), (dev_steps+1)*num_cells))
    #reshaped is num child per parent, num parents, (dev_steps+1)*num_cells shaped. 
    # so [:,0,:] is all kids of one parent
    pheno_std=np.std(reshaped,axis=0) #one std for each of the parents, so pop_size*trunc_prop now 10
    pheno_std = pheno_std.mean(1).mean() #first averaged across cells, then averaged across individuals in the population
    # generic phenotypic variation among offspring of the same parent

    #looking for more sophisticated phenotypic variation:
    reshaped2D=np.reshape(reshaped, (num_child, len(parent_locs), dev_steps+1, num_cells))

    values_they_should_match = reshaped2D[:,:,where_overlap[0],where_overlap[1]]
    #values_they_should_match.shape #4 kids, 2 parents, N values, where N is the number of cells where they overlap
    matching_std = np.std(values_they_should_match, axis=0) #among the 4 kids of 1 parent, output is an N long list for each of the 2 parents
    matching_std = matching_std.mean(axis=1) #average across the N overlaps, to get 1 value for each parent

    #repeat for non-overlap
    values_they_shouldnt_match = reshaped2D[:,:,where_no_overlap[0],where_no_overlap[1]]
    #values_they_should_match.shape #4 kids, 2 parents, N values, where N is the number of cells where they don't overlap
    nonmatching_std = np.std(values_they_shouldnt_match, axis=0) #among the 4 kids of 1 parent, output is an N long list for each of the 2 parents
    nonmatching_std = nonmatching_std.mean(axis=1) #average across the N non overlaps, to get 1 value for each parent

    #minimum std is 0, max is 0.5 in the case of values that range between 0 and 1
    combined_std = nonmatching_std - matching_std
    averaged_combined_std = np.mean(combined_std)
    best_std_id = np.argmax(combined_std)

    return pheno_std, np.max(combined_std), best_std_id, averaged_combined_std

@njit("f8[:,:](f8[:,:],i8, i8)")
def sigmoid(x,a,c):
  return 1/(1 + np.exp(-a*x+c))

def fitness_function_ca(phenos, targ):
  """
  Takes phenos which is pop_size * iters+1 * num_cells and targ which is iters+1 * num_cells
  Returns 1 fitness value for each individual, np array of size pop_size
  """
  return -np.abs(phenos - targ).sum(axis=1).sum(axis=1)

def seedID2string(seed_int, num_cells):
  #takes an integer, turns it into a starting pattern
  binary_string = bin(int(seed_int))[2:]
  binary_list = [int(digit) for digit in binary_string]
  start_pattern = np.array(binary_list)
  start_pattern=np.pad(start_pattern, (num_cells-len(start_pattern),0), 'constant', constant_values=(0))
  return start_pattern

def seed2expression(start_pattern, pop_size, num_cells, grn_size, geneid):
  #takes a starting pattern and makes a population of starting gene expressions
  start_gene_values = np.zeros((pop_size, int(num_cells * grn_size)))
  start_gene_values[:,geneid::grn_size] = start_pattern
  start_padded_gene_values = np.pad(start_gene_values, [(0,0),(1,1)], "wrap")
  start_padded_gene_values = np.float64(start_padded_gene_values)
  return start_padded_gene_values

#DO THE MULTICELLULAR DEVELOPMENT
@njit("f8[:](f8[:], f8[:,:], i8, i8)")
#Make sure that numpy imput in foat64! Otherwise this code breaks
def update_with_grn(padded_gene_values, grn, num_cells, grn_size):
  """
  Gene expression pattern + grn of a single individual -> Next gene expression pattern
  Takes
  - padded_gene_values: np array with num_genes * num_cells + 2 values
  Gene 1 in cell 1, gene 2 in cell 1, etc then gene 1 in cell 2, gene 2 in cell 2... plus left-right padding
  - grn: np array with num_genes * num_genes +2 values, shape of the GRN
  """
  #This makes it so that each cell is updated simultaneously
  #Accessing gene values in current cell and neighbors
  windows = np.lib.stride_tricks.as_strided(
      padded_gene_values, shape=(num_cells, grn_size + 2), strides=(8 * grn_size, 8)
  )
  #Updating with the grn
  next_step = windows.dot(grn)
  c = ALPHA/2
  next_step = sigmoid(next_step,ALPHA,c)

  #Returns same shape as padded_gene_values
  return next_step.flatten()

@njit("f8[:](f8[:], f8[:,:], i8, i8)")
#Make sure that numpy imput in foat64! Otherwise this code breaks
def update_internal(padded_gene_values, grn, num_cells, grn_size):
  """
  Gene expression pattern + grn of a single individual -> Next gene expression pattern
  Takes
  - padded_gene_values: np array with num_genes * num_cells + 2 values
  Gene 1 in cell 1, gene 2 in cell 1, etc then gene 1 in cell 2, gene 2 in cell 2... plus left-right padding
  - grn: np array with num_genes * num_genes +2 values, shape of the GRN
  """
  #Updating with the internal grn
  internal_grn = grn[1:-1,:]
  gene_vals = padded_gene_values[1:-1].copy()
  gene_vals = gene_vals.reshape(num_cells,grn_size)
  next_step = gene_vals.dot(internal_grn)
  c = ALPHA/2
  next_step = sigmoid(next_step,ALPHA,c)

  #Returns same shape as padded_gene_values
  return next_step.flatten()

#Might be faster non-parallel depending on how long computing each individual takes!
@njit(f"f8[:,:,:](f8[:,:], f8[:,:,:], i8, i8, i8, i8)", parallel=True)
def develop(
    padded_gene_values,
    grns,
    iters,
    pop_size,
    grn_size,
    num_cells
):
  """
  Starting gene expression pattern + all grns in the population ->
  expression pattern throughout development for each cell for each individual
  DOES NOT assume that the starting gene expression pattern is the same for everyone
  returns tensor of shape: [POP_SIZE, N_ITERS+1, num_cellsxgrn_size]
  N_ITERS in num developmental steps not including the initial step
  """
  NCxNGplus2 = padded_gene_values.shape[1]
  history = np.zeros((pop_size, iters+1, NCxNGplus2 - 2), dtype=np.float64)

  #For each individual in parallel
  for i in prange(pop_size):
    #IMPORTANT: ARRAYS IN PROGRMING when "copied" just by assigning to a new variable (eg a=[1,2,3], b = a)
    #Copies location and so b[0]=5 overwrites a[0] too! Need .copy() to copy variable into new memory location
    grn = grns[i]
    state = padded_gene_values[i].copy()
    history[i, 0, :] = state[1:-1].copy() #saving the initial condition
    #For each developmental step
    for t in range(iters):
      #INTERNAL
      state[1:-1] = update_internal(state, grn, num_cells, grn_size)
      #To wrap around, change what the pads are
      state[0] = state[-2] #the last element of the output of update_with_grn
      state[-1] = state[1]
      #EXTERNAL
      state[1:-1] = update_with_grn(state, grn, num_cells, grn_size)
      #To wrap around, change what the pads are
      state[0] = state[-2] #the last element of the output of update_with_grn
      state[-1] = state[1]
      history[i, t+1, :] = state[1:-1].copy()
  return history

#Torch implementation

def update_pop_torch(state, grns, NC, NG):
    """
    Receives:
        - state of shape (POP, NCxNG)
        - grns of shape (POP, NG+2, NG)

    Updates the state applying each individual's grn
    to windows that include one communication gene from
    the immediate neighbors (see below for explanation)

    Returns:
        - new state od shape (POP, NCxNG)

    e.g.

    POP = 2 # ind1, ind2
    NC = 3  # cell1 cell2
    NG = 4  # g1, g2, g3, g4

    state:
           g1 g2 g3 g4   g1 g2 g3 g4
           [1, 2, 3, 4]  [5, 6, 7, 8]   ...

               cell1       cell2      cell3
            ----------  ----------  ----------
    ind1 [[ 1  2  3  4  5  6  7  8  9 10 11 12]
    ind2  [13 14 15 16 17 18 19 20 21 22 23 24]]

    padded w/ zeros:

        [[ 0  1  2  3  4  5  6  7  8  9 10 11 12  0]
         [ 0 13 14 15 16 17 18 19 20 21 22 23 24  0]]

    windows:

        [[[ 0  1  2  3  4  5]
          [ 4  5  6  7  8  9]
          [ 8  9 10 11 12  0]]

         [[12  0  0 13 14 15]
          [14 15 16 17 18 19]
          [18 19 20 21 22 23]]]

    assuming dtype is the size of a single entry in state

    state.shape   = (POP, NC * NG)
    state.strides = (NC * NG * dtype, dtype)

    windows.shape   = (POP, NC, NG+2)
    windows.strides = (NC * NG * dtype, NG * dtype, dtype)
    """
    device="cpu"
    POP, _ = state.shape
    #padded = np.pad(state, pad_width=[(0, 0), (1, 1)])
    padded = state.copy()
    view_shape = (POP, NC, NG + 2)
    strides = [padded.strides[0], state.strides[0] // NC, state.strides[1]]
    windows = np.lib.stride_tricks.as_strided(padded, shape=view_shape, strides=strides)
    tgrns = torch.from_numpy(grns).to(device)
    tgrns.requires_grad_(True)
    twins = torch.from_numpy(windows).to(device)
    #with torch.no_grad():
    res = torch.matmul(twins, tgrns).cpu()
    res = torch.clip(res, 0, 1).reshape(POP, NC * NG)
    return res.cpu().numpy()
    # new_state = np.clip(np.matmul(windows, grns), 0, 1)
    # return new_state.reshape(POP, NC * NG)


def develop_torch(
    state,
    grns,
    iters,
    pop_size,
    grn_size,
    num_cells,
):
    _, NCxNG = state.shape
    history = np.zeros((iters+1, pop_size, NCxNG-2), dtype=np.float64)

    # for i in prange(pop_size):
    #     state = gene_values[i].copy()
    #     grn = grns[i]
    #     for t in range(iters):
    #         state[1:-1] = update_with_grn(state, grn, num_cells, grn_size)
    #         history[i, t, :] = state[1:-1].copy()

    history[0] = state[:,1:-1].copy()
    for t in range(iters):
        print(state.shape)
        state = update_pop_torch(state, grns, num_cells, grn_size)
        print(state.shape)
        history[t+1] = state.copy()

    return history.transpose(1, 0, 2)

#MAKE TARGET DEVELOPMENTAL PATTERN OUT OF CA RULE
#ONE HOT START
def rule2targets_wrapped_onehot(r, L, N):
  """
  We need 2 flips:

  1) from value order to wolfram order

    | value | wolfram
  N | order | order
  --|-------|---------
  0 |  000  |  111
  1 |  001  |  110
  2 |  010  |  101
  3 |  011  |  100
  4 |  100  |  011
  5 |  101  |  010
  6 |  110  |  001
  7 |  111  |  000

  so we do:
      rule = rule[::-1]

  2) from array order to base order

  array order: np.arange(3) = [0, 1, 2]

  but base2 orders digits left-to-right

  e.g.
  110 = (1        1        0)    [base2]
         *        *        *
   (2^2) 4  (2^1) 2  (2^0) 1
        ---------------------
      =  4 +      2 +      0  = 6 [base10]

  so we do:
    2 ** np.arange(2)[::-1] = [4 2 1]

  """
  base = 2 ** np.arange(3)[::-1]
  rule = np.array([int(v) for v in f"{r:08b}"])[::-1]

  targets = np.zeros((L, N), dtype=np.int32)
  targets[0][int(N/2)] = 1

  for i in range(1, L):
    s = np.pad(targets[i - 1], (1, 1), "wrap")
    s = sliding_window_view(s, 3)
    s = (s * base).sum(axis=1)
    s = rule[s]
    targets[i] = s

  return targets.astype(np.float64)

#ONE HOT WITH MOVEBY
def rule2targets_wrapped_wmoveby(r, moveby, L, N):
  
  base = 2 ** np.arange(3)[::-1]
  rule = np.array([int(v) for v in f"{r:08b}"])[::-1]

  targets = np.zeros((L, N), dtype=np.int32)
  targets[0][int(N/2)+moveby] = 1

  for i in range(1, L):
    s = np.pad(targets[i - 1], (1, 1), "wrap")
    s = sliding_window_view(s, 3)
    s = (s * base).sum(axis=1)
    s = rule[s]
    targets[i] = s

  return targets.astype(np.float64)

#WITH CUSTOM START
def rule2targets_wrapped_wstart(r, L, N, start_pattern):
  base = 2 ** np.arange(3)[::-1]
  rule = np.array([int(v) for v in f"{r:08b}"])[::-1]
  targets = np.zeros((L, N), dtype=np.int32)
  
  targets[0] = start_pattern

  for i in range(1, L):
    s = np.pad(targets[i - 1], (1, 1), "wrap")
    s = sliding_window_view(s, 3)
    s = (s * base).sum(axis=1)
    s = rule[s]
    targets[i] = s

  return targets.astype(np.float64)

#FOR PLOTTING
#------------------------

def show_effectors(states, targets, M, ax):
    preds = np.where(states[:, M:] > 0.5, 1, 0)

    correct = np.where(np.abs(targets - preds) > 0, 1, 0)
    correct_mask = np.ma.array(targets, mask=correct)

    reds = np.dstack(
        [np.ones_like(targets) * 255, np.zeros_like(targets), np.zeros_like(targets)]
    )

    # Create the figure
    ax.imshow(reds, label="errors")  # red background
    ax.imshow(correct_mask)

    # a bit of a mindfuck, correct is used as a mask so it's like the inverse... xD
    error_perc = correct.sum() / targets.size * 100

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=f"Errors:\n{error_perc:.1f}%",
            markerfacecolor="r",
            markersize=10,
        )
    ]

    # print(targets.size())

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.27, 1.0))

    ax.set_title("Effector genes")

def imshow_ca(grid, ax):
    rocket_cmap = sns.color_palette("rocket", as_cmap=True)
    # im = ax.imshow(grid, cmap="magma")
    im = ax.imshow(grid, cmap=rocket_cmap,interpolation="nearest")

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)

    # And a corresponding grid
    ax.grid(which="minor", alpha=0.3)
    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    for tick in ax.yaxis.get_minor_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

    return im

def plot_three_line(ax, rule, data1, data2, data3, season_len=300, legend=False):
    #Plots the fitness over generations for 3 datasets (data1=static 1, data2=static 2, data3=variable)
    
    # Calculate mean and standard error for each list
    mean1 = np.mean(data1, axis=0)
    mean2 = np.mean(data2, axis=0)
    mean3 = np.mean(data3, axis=0)
    stderr1 = np.std(data1, axis=0) / np.sqrt(len(data1))
    stderr2 = np.std(data2, axis=0) / np.sqrt(len(data2))
    stderr3 = np.std(data3, axis=0) / np.sqrt(len(data3))

    for j in range(0, len(mean1), season_len):
        if j % (season_len * 2) == 0:
            ax.axvline(j, linestyle="--", color="gray", alpha=0.3)
        else:
            ax.axvline(j, linestyle=":", color="gray", alpha=0.3)
    
    # Plot data
    ax.plot(mean1, label='Static T1', color='blue')
    ax.tick_params(right=True, labelright=False)
    ax.plot(mean2, label='Static T2', color='orange')
    ax.plot(mean3, label='Variable env', color='red')
    
    # Fill the area between the lines and the error bars
    ax.fill_between(range(len(mean1)), mean1 - stderr1, mean1 + stderr1, color='blue', alpha=0.3)
    ax.fill_between(range(len(mean2)), mean2 - stderr2, mean2 + stderr2, color='orange', alpha=0.3)
    ax.fill_between(range(len(mean3)), mean3 - stderr3, mean3 + stderr3, color='red', alpha=0.3)
    
    ax.set_title("Rule "+str(rule))
    #ax.grid(axis="y")
    #ax.set_ylabel("Fitness")
    #plt.savefig("rule_"+str(rule)+"_lines.png")
    if legend:
        ax.legend(fontsize=14)
        height = 0.62
        base = season_len/2
        kwargs = {"ha":"center", "va":"center", "fontsize":12, "color":"gray"}
        ax.text(base, height, "T1", **kwargs)
        ax.text(base + season_len, height, "T2", **kwargs)
        ax.text(base + season_len*2, height, "T1", **kwargs)
        ax.text(base + season_len*3, height, "T2", **kwargs)

def get_pop_TPF(pop, pop_size, num_cells, grn_size, dev_steps, geneid, rule, seed_int):
  start_pattern = seedID2string(seed_int, num_cells)
  start_expression = seed2expression(start_pattern, pop_size, num_cells, grn_size, geneid)
   
  target = rule2targets_wrapped_wstart(int(rule), L=dev_steps+1, N=num_cells, start_pattern=start_pattern)
   
  all_phenos = develop(start_expression, pop, dev_steps, pop_size, grn_size, num_cells)
  phenos = all_phenos[:,:,geneid::grn_size]
   
  worst= -num_cells*dev_steps
  prefitnesses = fitness_function_ca(phenos, target)
  fitnesses=1-(prefitnesses/worst) #0-1 scaling

  return target, phenos, fitnesses

def get_pop_TPF_torch(pop, pop_size, num_cells, grn_size, dev_steps, geneid, rule, seed_int):
  start_pattern = seedID2string(seed_int, num_cells)
  start_expression = seed2expression(start_pattern, pop_size, num_cells, grn_size, geneid)
   
  target = rule2targets_wrapped_wstart(int(rule), L=dev_steps+1, N=num_cells, start_pattern=start_pattern)
   
  #all_phenos = develop(start_expression, pop, dev_steps, pop_size, grn_size, num_cells)
  all_phenos = develop_torch(start_expression, pop, dev_steps, pop_size, grn_size, num_cells)
  phenos = all_phenos[:,:,geneid::grn_size]
   
  worst= -num_cells*dev_steps
  prefitnesses = fitness_function_ca(phenos, target)
  fitnesses=1-(prefitnesses/worst) #0-1 scaling

  return target, phenos, fitnesses

def get_fits(rules, seed_ints, metric, root, season_len, num_reps, exprapolate=True):
    vari_maxs=[np.loadtxt(os.path.expanduser(root+f"stats_{season_len}_{rules[0]}-{rules[1]}_{seed_ints[0]}-{seed_ints[1]}_{i+1}_{metric}.txt")) for i in range(num_reps)]
    if rules[0] == rules[1]:
            if rules[0] in [154,82,86,18]:
                env1_maxs=[np.loadtxt(os.path.expanduser(root+f"stats_0_{rules[0]}_{seed_ints[0]}_{i+1}_{metric}.txt")) for i in range(num_reps)]
                env2_maxs=[np.loadtxt(os.path.expanduser(root+f"stats_0_{rules[0]}_{seed_ints[1]}_{i+1}_{metric}.txt")) for i in range(num_reps)]
            else:
                env1_maxs=[np.loadtxt(os.path.expanduser(root+f"stats_20000_{rules[0]}_{seed_ints[0]}_{i+1}_{metric}.txt")) for i in range(num_reps)]
                env2_maxs=[np.loadtxt(os.path.expanduser(root+f"stats_20000_{rules[0]}_{seed_ints[1]}_{i+1}_{metric}.txt")) for i in range(num_reps)]
    else:
        print("scenario not yet implemented")

    if exprapolate:
        diff_len = len(vari_maxs[0]) - len(env1_maxs[0])
        if diff_len > 1:
            env1_maxs=np.array(env1_maxs)
            env2_maxs=np.array(env2_maxs)
            last_elements = env1_maxs[:,-1]
            last_elements=np.tile(last_elements, (diff_len, 1)).T
            env1_maxs = np.hstack((env1_maxs, last_elements))
            last_elements = env2_maxs[:,-1]
            last_elements=np.tile(last_elements, (diff_len, 1)).T
            env2_maxs = np.hstack((env2_maxs, last_elements))

    return vari_maxs, env1_maxs, env2_maxs

def chunker(runs, season_len = 300):
    florp = np.array(runs).mean(axis=0) # average runs
    n_seasons = int(np.floor(florp.shape[0]/season_len))
    chunked_seasons = np.array([florp[i*300:(i+1)*300] for i in range(n_seasons)])
    assert chunked_seasons.size == season_len * n_seasons #safety check
    chunked_season1, chunked_season2 = chunked_seasons[0::2], chunked_seasons[1::2]
    max_chunked_season1, max_chunked_season2 = chunked_season1.max(axis=1),chunked_season2.max(axis=1)
    return max_chunked_season1.max(), max_chunked_season2.max()

def scatter_value(variable, season1, season2, season_len):
    vari_env1, vari_env2 = chunker(variable, season_len=season_len)
    M_env1 = np.array(season1).mean(axis=0).max()
    M_env2 = np.array(season2).mean(axis=0).max()
    diffs = (vari_env1 - M_env1, vari_env2 - M_env2)
    return diffs

def main_plt(xs, ys, rules, ax):
  ax.scatter(xs, ys, s=40, zorder=3, color="red", edgecolors="black")
  fontsize = 18

  for i, label in enumerate(rules):
      if label == 254:
          ax.annotate(
              label,
              fontsize=fontsize,
              xy=(xs[i], ys[i]),
              xytext=(xs[i] - 0.03, ys[i] + 0.02),
              arrowprops=dict(
                  facecolor="black", shrink=0.05, width=0.2, headwidth=3, headlength=5
              ),
          )
      elif label == 50:
          ax.annotate(
              label,
              fontsize=fontsize,
              xy=(xs[i], ys[i]),
              xytext=(xs[i] + 0.01, ys[i] + 0.02),
              arrowprops=dict(
                  facecolor="black", shrink=0.05, width=0.2, headwidth=3, headlength=5
              ),
          )
      else:
          ax.text(
              xs[i],
              ys[i],
              label,
              fontsize=fontsize,
              ha="center",
              va="bottom",
              color="black",
          )

  ax.set_xlim(-0.06, 0.12)
  ax.set_ylim(-0.06, 0.12)
  ax.axvline(0, lw=1, color="black")
  ax.axhline(0, lw=1, color="black")
  ax.set_xlabel("Max fit of variable - Max fit of static T1",fontsize=22)
  ax.set_ylabel("Max fit of variable - Max fit of static T2",fontsize=22)
  ax.grid(zorder=0)

def chunker_plotting(run, season_len = 300):
    gens=list(range(len(run)))
    n_seasons = int(np.floor(run.shape[0]/season_len))
    chunked_seasons = np.array([run[i*300:(i+1)*300] for i in range(n_seasons)])
    chunked_gens = np.array([gens[i*300:(i+1)*300] for i in range(n_seasons)])

    assert chunked_seasons.size == season_len * n_seasons #safety check

    chunked_season1, chunked_season2 = chunked_seasons[0::2], chunked_seasons[1::2]
    chunked_gens1, chunked_gens2 = chunked_gens[0::2], chunked_gens[1::2]
    
    return chunked_season1, chunked_season2, chunked_gens1, chunked_gens2

def try_grn(variable, rule, run_seedints, try_seedints, grn_size, geneid, root, num_cells, dev_steps):
    last_grns=[]
    for i in range(5):
        if variable:
            filename = f"{root}/stats_300_{rule}-{rule}_{run_seedints[0]}-{run_seedints[1]}_{i+1}" + "_best_grn.txt"
        else:
            if rule in [154,82,86,18]:
                filename = f"results_new_rules/stats_0_{rule}_{run_seedints}_{i+1}" + "_best_grn.txt"
            else:
                filename = f"results_new_rules/stats_600_{rule}_{run_seedints}_{i+1}" + "_best_grn.txt"
        grns = np.loadtxt(filename)
        num_grns = int(grns.shape[0]/(grn_size+2)/grn_size)
        grns = grns.reshape(num_grns,grn_size+2,grn_size)
        grn = grns[-1,:,:]
        last_grns.append(grn)
    last_grns = np.array(last_grns)

    last_phenos=[]
    fits = []
    for s in try_seedints:
        targets, phenos, fitnesses = get_pop_TPF(last_grns, len(last_grns), num_cells, grn_size, dev_steps, geneid, rule, s)
        last_phenos.append(phenos)
        fits.append(fitnesses)
    last_phenos = np.array(last_phenos)
    fits = np.array(fits)
    return last_phenos, fits, last_grns

def make_restricted_plot(all_targs, num_cells, dev_steps, dot_xs, dot_ys, labelled=True):
    
    worst= -num_cells*(dev_steps+1)
    oritargs = np.array([all_targs[0],all_targs[1]])

    where_overlap = np.where(all_targs[0]==all_targs[1])
    where_no_overlap = np.where(all_targs[0]!=all_targs[1])

    bestgen=all_targs[0].copy()
    bestgen[where_no_overlap] = 0.5
    bestgen = np.expand_dims(bestgen, axis=0)

    half= int(len(where_no_overlap[0])/2)

    a = all_targs[0].copy()
    a[tuple(idx[:half] for idx in where_no_overlap)] = 0.5
    a = np.expand_dims(a, axis=0)

    b = all_targs[1].copy()
    b[tuple(idx[:half] for idx in where_no_overlap)] = 0.5
    b = np.expand_dims(b, axis=0)

    inperfa = 1 - all_targs[0].copy()
    inperfa = np.expand_dims(inperfa, axis=0)
    inperfb = 1 - all_targs[1].copy()
    inperfb = np.expand_dims(inperfb, axis=0)

    worstgen=inperfa[0].copy()
    worstgen[where_no_overlap] = 0.5
    worstgen = np.expand_dims(worstgen, axis=0)

    c= all_targs[0].copy()
    c[where_overlap] = 0.5
    c = np.expand_dims(c, axis=0)

    d= all_targs[1].copy()
    d[where_overlap] = 0.5
    d = np.expand_dims(d, axis=0)

    labels = ["A", "B", "Overlap good, rest 0.5", "Overlap good, rest/2 0.5, A", "Overlap good, rest/2 0.5, B", "A inverse","B inverse", "Overlap inverse, rest 0.5"]
    labels.append("A but overlap 0.5")
    labels.append("B but overlap 0.5")

    pop = np.concatenate((oritargs, bestgen,a,b,inperfa,inperfb,worstgen,c,d), axis=0) #0,1, 4,5

    fitnesses1 = -np.abs(pop - all_targs[0]).sum(axis=1).sum(axis=1)
    fitnesses1=1-(fitnesses1/worst) #0-1 scaling
    fitnesses2 = -np.abs(pop - all_targs[1]).sum(axis=1).sum(axis=1)
    fitnesses2=1-(fitnesses2/worst) #0-1 scaling

    pop_df = pd.DataFrame()
    pop_df["x"]=fitnesses1
    pop_df["y"]=fitnesses2
    xs=fitnesses1
    ys=fitnesses2

    if labelled:
        labels = list(zip(fitnesses1,fitnesses2))
        plt.scatter(pop_df["x"], pop_df["y"])
        for i, label in enumerate(labels): 
            plt.text(
                xs[i],
                ys[i],
                label,
                ha="center",
                va="bottom",
                color="black",
            )

    plt.scatter(dot_xs, dot_ys)
    sns.set_style("whitegrid")

    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)

    myhatch='..'
    mycolor="C0"
    triangle = Polygon([[1,1], pop_df.iloc[0], pop_df.iloc[1]], closed=True, alpha=0.5,edgecolor=mycolor, facecolor='none',hatch=myhatch)
    plt.gca().add_patch(triangle)
    triangle = Polygon([[0,0], pop_df.iloc[5], pop_df.iloc[6]], closed=True, alpha=0.5,edgecolor=mycolor, facecolor='none',hatch=myhatch)
    plt.gca().add_patch(triangle)
    triangle = Polygon([[0,1], pop_df.iloc[1], pop_df.iloc[5]], closed=True, alpha=0.5,edgecolor=mycolor, facecolor='none',hatch=myhatch)
    plt.gca().add_patch(triangle)
    triangle = Polygon([[1,0], pop_df.iloc[0], pop_df.iloc[6]], closed=True, alpha=0.5,edgecolor=mycolor, facecolor='none',hatch=myhatch)
    plt.gca().add_patch(triangle)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    return pop_df
