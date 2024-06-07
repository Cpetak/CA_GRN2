import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from numba import njit, prange
import argparse
from pathlib import Path
from time import monotonic

import helper

#DO THE MULTICELLULAR DEVELOPMENT
@njit("f8[:](f8[:], f8[:,:], i8, i8, i8, i8)")
#Make sure that numpy imput in foat64! Otherwise this code breaks
def update_with_grn(padded_gene_values, grn, num_cells, grn_size, residual, alpha):
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
  if residual:
    c = 0
  else:
    c = alpha/2
  next_step = helper.sigmoid(next_step,alpha,c)

  #Returns same shape as padded_gene_values
  return next_step.flatten()

@njit("f8[:](f8[:], f8[:,:], i8, i8, i8, i8)")
#Make sure that numpy imput in foat64! Otherwise this code breaks
def update_internal(padded_gene_values, grn, num_cells, grn_size, residual, alpha):
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
  if residual:
    c = 0
  else:
    c = alpha/2
  next_step = helper.sigmoid(next_step,alpha,c)

  #Returns same shape as padded_gene_values
  return next_step.flatten()

#Might be faster non-parallel depending on how long computing each individual takes!
@njit(f"f8[:,:,:](f8[:,:], f8[:,:,:], i8, i8, i8, i8,i8)", parallel=True)
def develop(
    padded_gene_values,
    grns,
    iters,
    pop_size,
    grn_size,
    num_cells,
    residual
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
  alpha = 10

  #For each individual in parallel
  for i in prange(pop_size):
    #IMPORTANT: ARRAYS IN PROGRMING when "copied" just by assigning to a new variable (eg a=[1,2,3], b = a)
    #Copies location and so b[0]=5 overwrites a[0] too! Need .copy() to copy variable into new memory location
    grn = grns[i]
    state = padded_gene_values[i].copy()
    history[i, 0, :] = state[1:-1].copy() #saving the initial condition
    #For each developmental step
    if residual:
      for t in range(iters):
        #INTERNAL
        next_step = update_internal(state, grn, num_cells, grn_size, residual, alpha)
        next_step=(next_step * 2) - 1
        state[1:-1] += next_step
        state[1:-1]=np.clip(state[1:-1], 0.0, 1.0)
        #To wrap around, change what the pads are
        state[0] = state[-2] #the last element of the output of update_with_grn
        state[-1] = state[1]
        #EXTERNAL
        next_step = update_with_grn(state, grn, num_cells, grn_size, residual, alpha)
        next_step=(next_step * 2) - 1
        state[1:-1] += next_step
        state[1:-1]=np.clip(state[1:-1], 0.0, 1.0)
        #To wrap around, change what the pads are
        state[0] = state[-2] #the last element of the output of update_with_grn
        state[-1] = state[1]
        history[i, t+1, :] = state[1:-1].copy()
    else:
      for t in range(iters):
        #INTERNAL
        state[1:-1] = update_internal(state, grn, num_cells, grn_size, residual, alpha)
        #To wrap around, change what the pads are
        state[0] = state[-2] #the last element of the output of update_with_grn
        state[-1] = state[1]
        #EXTERNAL
        state[1:-1] = update_with_grn(state, grn, num_cells, grn_size, residual, alpha)
        #To wrap around, change what the pads are
        state[0] = state[-2] #the last element of the output of update_with_grn
        state[-1] = state[1]
        history[i, t+1, :] = state[1:-1].copy()
  return history


"""#Evolutionary algorythm"""

def fitness_function_ca(phenos, targ):
  """
  Takes phenos which is pop_size * iters+1 * num_cells and targ which is iters+1 * num_cells
  Returns 1 fitness value for each individual, np array of size pop_size
  """
  return -np.abs(phenos - targ).sum(axis=1).sum(axis=1)

def evolutionary_algorithm(pop_size, grn_size, num_cells, dev_steps, mut_rate, num_generations, selection_prop, rule, mut_size, folder, residual, seedn1, seedn2, season_len, seededness, rule_combo, job_array_id):

  #Setting up
  #Creating population

  #adding a value for each gene in the grn which will be added after matrix multiplication
  #by the rest of the values in the grn
  pop = np.random.randn(pop_size, grn_size+2, grn_size).astype(np.float64)

  curr = 0
  worst= -num_cells*dev_steps
  
  #Creating start expression pattern
  geneid=1 #maternal gene, also the pattern giver, 0 is first but that is used for communication
  inputs= []
  if seededness == "env_seeded":
    #rules=[102, 150, 90]
    #rule=rules[rule_combo] #so that I can reuse the same experiment launcher for the two if branches

    for n in [seedn1, seedn2]:
      start_gene_values = np.zeros((pop_size, int(num_cells * grn_size)))
      start_pattern=np.array([0] * num_cells)
      start_pattern[1::(int(num_cells/n))]=1
      start_pattern[1]=0
      start_pattern[-1]=0
      start_gene_values[:,geneid::grn_size] = start_pattern
      start_padded_gene_values = np.pad(start_gene_values, [(0,0),(1,1)], "wrap")
      start_padded_gene_values = np.float64(start_padded_gene_values)
      inputs.append(start_padded_gene_values)

    #Creating target
    target1 = helper.rule2targets_wrapped_multiseeded(rule, L=dev_steps+1, N=num_cells, seedn=seedn1)
    target2 = helper.rule2targets_wrapped_multiseeded(rule, L=dev_steps+1, N=num_cells, seedn=seedn2)
    targets=[target1, target2]

  elif seededness == "one_hot_seeded":
    start_gene_values = np.zeros((pop_size, int(num_cells * grn_size)))
    middle_idx=int(num_cells/2) #num_cells before middle cell
    geneid=1 #maternal gene, also the pattern giver, 0 is first but that is used for communication
    start_gene_values[:,middle_idx*grn_size+geneid]=1
    start_padded_gene_values = np.pad(start_gene_values, [(0,0),(1,1)], "wrap")
    start_padded_gene_values = np.float64(start_padded_gene_values)
    inputs.append(start_padded_gene_values)
    inputs.append(start_padded_gene_values)

    #Creating target
    rule_combos=[[54,102],[102,94],[54,94]]
    curr_combo=rule_combos[rule_combo]
    target1 = helper.rule2targets_wrapped(rule, L=dev_steps+1, N=num_cells)
    target2 = helper.rule2targets_wrapped(rule, L=dev_steps+1, N=num_cells)
    targets=[target1, target2]

  #Logging targets
  max_fits = []
  ave_fits = []
  kid_stds = []
  save_freq=100
  best_grns = np.zeros((save_freq, grn_size+2, grn_size))
  both_fits_hist = np.zeros((save_freq, 2, pop_size))
  filename = f"{folder}/stats_{seededness}_{residual}_{season_len}_{rule}_{seedn1}_{job_array_id}"

  #Defining variables
  selection_size=int(pop_size*selection_prop)
  num_child = int(pop_size / selection_size) - 1
  tot_children = num_child * selection_size
  num_genes_mutate = int((grn_size + 2) * grn_size * tot_children * mut_rate)

  #when_change= np.arange(300,num_generations, 300)
  #when_change_first_to_second=when_change[::2]
  #points=np.arange(0,len(when_change_first_to_second),len(when_change_first_to_second)//4)
  #points[-1]=len(when_change_first_to_second)-2
  #intgens=np.array([[i-300/2,i-1,i+300/2,i+300-1] for i in when_change_first_to_second[points]])
  #allgen=intgens.flatten().astype(int)
  #all_int_gen=np.sort(allgen)
  #print(all_int_gen)

  evol_st = monotonic()

  # Main for loop
  for gen in range(num_generations):

    # Generating phenotypes
    #Return [pop_size, dev_stepss+1, num_cellsxgrn_size] np.float64 array
    phenos = develop(inputs[curr], pop, dev_steps, pop_size, grn_size, num_cells, residual)
    #get second gene for each cell only, the one I decided will matter for the fitness
    #pop_size, dev_steps, NCxNG
    p=phenos[:,:,1::grn_size]

    #if gen > 0:
    if False:
      child_phenotypes = p[children_locs]
      reshaped=np.reshape(child_phenotypes, (num_child, len(parent_locs), (dev_steps+1)*num_cells))
      stds=np.std(reshaped,axis=0) #one std for each of the parents, so pop_size*trunc_prop now 10
      kid_stds.append(stds.mean(1).mean())

    #Calculating fitnesses
    fitnesses0 = fitness_function_ca(p, targets[0])
    fitnesses0=1-(fitnesses0/worst) #0-1 scaling
    fitnesses1 = fitness_function_ca(p, targets[1])
    fitnesses1=1-(fitnesses1/worst) #0-1 scaling
    fitnesses=np.array([fitnesses0,fitnesses1])
    both_fits_hist[gen % save_freq] = fitnesses

    #Selection
    perm = np.argsort(fitnesses[curr])[::-1]

    #Logging
    best_grn = pop[perm[0]]
    best_grns[gen % save_freq] = best_grn
    
    max_fit=fitnesses[curr].max().item()
    ave_fit=fitnesses[curr].mean().item()
    max_fits.append(max_fit)  # keeping track of max fitness
    ave_fits.append(ave_fit)  # keeping track of average fitness

    # location of top x parents in the array of individuals
    parent_locs = perm[:selection_size]
    # location of individuals that won't survive and hence will be replaced by others' children
    children_locs = perm[selection_size:]

    parents = pop[parent_locs]
    children = np.tile(parents, (num_child, 1, 1))

    #Mutation
    mutations = np.random.randn(num_genes_mutate) * mut_size
    x, y, z = children.shape
    xs = np.random.choice(x, size=num_genes_mutate)
    ys = np.random.choice(y, size=num_genes_mutate)
    zs = np.random.choice(z, size=num_genes_mutate)
    children[xs, ys, zs] = children[xs, ys, zs] + mutations

    pop[children_locs] = children  # put children into population

    #Change environment
    if gen % season_len == season_len - 1: # flip target
      curr = (curr + 1) % 2


    #if gen % save_freq == save_freq - 1: #every hunderd gens save
    if False:
      evol_et = monotonic()
      print(f"gen {gen} took: {evol_et - evol_st:.5f}", flush=True)
      st = monotonic()                                          # START TIME
      best_grns = best_grns.reshape(save_freq,grn_size*(grn_size+2))
      with open(filename+"_best_grn.txt", 'a') as f:
        np.savetxt(f, best_grns, newline=" ")
      best_grns = np.zeros((save_freq, grn_size+2, grn_size))
      both_fits_hist = both_fits_hist.reshape(save_freq,2*pop_size)
      with open(filename+"_both_fits.txt", 'a') as f:
        np.savetxt(f, both_fits_hist, newline=" ")
      both_fits_hist = np.zeros((save_freq, 2, pop_size))
      et = monotonic()                                          # END TIME
      print(f"writing took: {et - st:.5f}s", flush=True)
      evol_st = monotonic()
    # if gen in all_int_gen:
    #   p_to_save=np.reshape(p,(pop_size,num_cells*(dev_steps+1)))
    #   with open(filename+"_phenos"+str(gen)+".txt", 'w') as f:
    #     np.savetxt(f, p_to_save, newline=" ")

  
  with open(filename+"_maxfits.txt", 'w') as f:
    np.savetxt(f, max_fits, newline=" ")
  with open(filename+"_avefits.txt", 'w') as f:
    np.savetxt(f, ave_fits, newline=" ")
  #with open(filename+"_kidstds.txt", 'w') as f:
    #np.savetxt(f, kid_stds, newline=" ")
  #targs=np.array(targets).reshape(2,(dev_steps+1)*num_cells)
  #with open(filename+"_targets.txt", 'w') as f:
    #np.savetxt(f, targs, newline=" ")
    
  #with open(f"{folder}/stats_{rule}.pkl", "wb") as f:
        #pickle.dump(stats, f)

  return max_fit

def prepare_run(entity, project, args, folder_name="final_results_all256_1env"):

    #run = wandb.init(config=args, entity=entity, project=project)

    folder = Path(folder_name) #/ run.name
    folder.mkdir(parents=True, exist_ok=True)

    return folder

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument('--pop_size', type=int, default=100, help="Population size")
  parser.add_argument('--grn_size', type=int, default=12, help="GRN size") 
  parser.add_argument('--num_cells', type=int, default=22, help="Number of cells") 
  parser.add_argument('--dev_steps', type=int, default=22, help="Number of developmental steps") 

  parser.add_argument('--selection_prop', type=float, default=0.1, help="Percent pruncation") 
  parser.add_argument('--mut_rate', type=float, default=0.1, help="Number of mutations") 
  parser.add_argument('--mut_size', type=float, default=1.5, help="Size of mutations") 
  parser.add_argument('--num_generations', type=int, default=100, help="Number of generations")

  #parser.add_argument('--exp_type', type=str, default="postdev", help="vs stepwise") 
  parser.add_argument('--residual', type=int, default=1, help="vs no, 0")
  #parser.add_argument('--alpha', type=int, default=10, help="steepness")
  parser.add_argument('--seedn1', type=int, default=5, help="seedpattern1")
  parser.add_argument('--seedn2', type=int, default=6, help="seedpattern2")
  parser.add_argument('--season_len', type=int, default=10, help="season length")
  #parser.add_argument('--sel_pressure', type=int, default=6, help="Selection pressure")
  #parser.add_argument('--fitness_type', type=str, default="twocomp", help="Selection pressure")

  parser.add_argument('--rule', type=int, default=250, help="CA rule target")

  parser.add_argument('--rule_combo', type=int, default=0, help="Which combination of rules to do")
  parser.add_argument('--seededness', type=str, default="env_seeded", help="How the development should start")
  #parser.add_argument('--embryo_steps', type=int, default=0, help="How many steps won't count in fitness")

  parser.add_argument('--job_array_id', type=int, default=0, help="Job array id to distinguish runs")

  args = parser.parse_args()

  #Writing to file
  folder = prepare_run("molanu", "variable_experiments", args)
  args.folder = folder

  args.num_cells = args.dev_steps

  print("running code", flush=True)
  score=evolutionary_algorithm(**vars(args))
  #wandb.log({'score': score})

  #For sweep, in terminal
  #wandb sweep --project test_for_vari config_vari_env.yaml
  #then
  #wandb agent molanu/sweep_test_CA_GRN/tvkyw75q

  #import pickle
  #file = open('results/cool-sunset-6/stats.pkl', 'rb')
  #data = pickle.load(file)
  #data["max_fits"]
