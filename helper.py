import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from matplotlib.lines import Line2D
from numba import njit, prange
from pathlib import Path

ALPHA = 10

def prepare_run(folder_name):
    
    folder = Path(folder_name)
    folder.mkdir(parents=True, exist_ok=True)

    return folder

@njit("f8[:,:](f8[:,:],i8, i8)")
def sigmoid(x,a,c):
  return 1/(1 + np.exp(-a*x+c))

def fitness_function_ca(phenos, targ):
  """
  Takes phenos which is pop_size * iters+1 * num_cells and targ which is iters+1 * num_cells
  Returns 1 fitness value for each individual, np array of size pop_size
  """
  return -np.abs(phenos - targ).sum(axis=1).sum(axis=1)

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
  #np.random.seed(42)
  #targets[0] = np.random.randint(2, size=(N))

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
  #np.random.seed(42)
  #targets[0] = np.random.randint(2, size=(N))

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

