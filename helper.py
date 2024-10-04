import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from matplotlib.lines import Line2D
from numba import njit, prange
from pathlib import Path
import seaborn as sns
#import torch
import hashlib

ALPHA = 10

def prepare_run(folder_name):
    
    folder = Path(folder_name)
    folder.mkdir(parents=True, exist_ok=True)

    return folder

def map_to_range(value):
  return int(hashlib.sha256(str(value).encode()).hexdigest(), 16) % (2**32)

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



