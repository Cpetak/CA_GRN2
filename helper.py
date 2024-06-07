import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from matplotlib.lines import Line2D
from numba import njit

"""#Helper functions"""

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

#MAKE TARGET DEVELOPMENTAL PATTERN OUT OF CA RULE
def rule2targets_wrapped(r, L, N):
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

def rule2targets_wrapped_multiseeded(r, L, N, seedn):
  base = 2 ** np.arange(3)[::-1]
  rule = np.array([int(v) for v in f"{r:08b}"])[::-1]

  targets = np.zeros((L, N), dtype=np.int32)
  #targets[0][int(N/2)] = 1
  #np.random.seed(42)
  #targets[0] = np.random.randint(2, size=(N))
  start_pattern=np.array([0] * N)
  start_pattern[1::(int(N/seedn))]=1
  start_pattern[1]=0
  start_pattern[-1]=0
  targets[0] = start_pattern

  for i in range(1, L):
    s = np.pad(targets[i - 1], (1, 1), "wrap")
    s = sliding_window_view(s, 3)
    s = (s * base).sum(axis=1)
    s = rule[s]
    targets[i] = s

  return targets.astype(np.float64)

#MAKE TARGET DEVELOPMENTAL PATTERN OUT OF CA RULE
def rule2targets_nowrap(r, L, N):
  base = 2 ** np.arange(3)[::-1]
  rule = np.array([int(v) for v in f"{r:08b}"])[::-1]

  targets = np.zeros((L, N), dtype=np.int32)
  targets[0][int(N/2)] = 1
  #np.random.seed(42)
  #targets[0] = np.random.randint(2, size=(N))

  for i in range(1, L):
    s = np.pad(targets[i - 1], (1, 1))
    s = sliding_window_view(s, 3)
    s = (s * base).sum(axis=1)
    s = rule[s]
    targets[i] = s

  return targets.astype(np.float64)

def rule2next(r, targets):

  base = 2 ** np.arange(3)[::-1]
  rule = np.array([int(v) for v in f"{r:08b}"])[::-1]

  s = np.pad(targets, (1, 1), "wrap")
  s = sliding_window_view(s, 3)
  s = (s * base).sum(axis=1)
  s = rule[s]

  return s

def memory_ca(a, rule, start, num_its):

  hist=[]
  hist.append(start)
  hidden = []
  hidden.append(start)

  next_step=start
  for i in range(num_its):
    #What should it be according to the rule
    temp_next=rule2next(rule, next_step)

    #Add to the hidden history, used for memory of rule output
    hidden.append(temp_next)

    freq_weight = []
    #For each element in the array
    for j in range(len(temp_next)):
      hist1_counter = 0
      hist0_counter = 0
      #for each step in the hidden history, count how many ones
      for h in range(len(hidden)):
        if hidden[len(hidden)-h-1][j] == 1:
          hist1_counter += 1/(h*a+1)
        else:
          hist0_counter += 1/(h*a+1)
      #if 1s are more frequent
      if hist1_counter > hist0_counter:
        freq_weight.append(1)
      elif hist1_counter < hist0_counter:
        freq_weight.append(0)
      else: #if is a tie
        freq_weight.append(temp_next[j])

    #Next step visible to us will depend on the memory
    next_step=freq_weight
    hist.append(next_step)

  return np.array(hist).astype(np.float64)

@njit("f8[:,:](f8[:,:],i8, i8)")
def sigmoid(x,a,c):
  return 1/(1 + np.exp(-a*x+c))

@njit
def sample(fitnesses, N):
  #Fitness proportional selection, "spinning the wheel"
  samples = np.ones(N, dtype=np.int64)
  cum = 0
  cumfit = np.zeros_like(fitnesses)
  for i in range(cumfit.shape[0]):
      cum += fitnesses[i]
      cumfit[i] = cum

  cumfit /= np.max(cumfit)

  for i in range(N):
      p = np.random.rand()
      j = 0
      while cumfit[j] < p:
          j += 1
      samples[i] = j
  return samples

def mynorm01(x):
  return (x-min(x))/(max(x)-min(x))

def plot_best_pheno(pheno, target):
  fig, axs = plt.subplots(ncols=3, figsize=(12, 4))
  #Plot best phenotype
  axs[0].imshow(pheno)
  #Calc fitness
  fitness = -np.abs(pheno - target).sum(axis=1).sum()
  axs[0].set_title(f"Genes fitness: {fitness:.3f}")
  #Plot target
  axs[1].imshow(target)
  #Plot difference
  show_effectors(pheno, target, M=0, ax=axs[2])

  plt.tight_layout()
  plt.show()
