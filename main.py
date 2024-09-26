import numpy as np
import argparse

import helper

"""#Evolutionary algorythm"""

def evolutionary_algorithm(pop_size, grn_size, num_cells, dev_steps, mut_rate, num_generations, selection_prop, rules, mut_size, folder, seed_ints, season_len, job_array_id):

  #Setting up
  #Creating population

  #adding a value for each gene in the grn which will be added after matrix multiplication
  #by the rest of the values in the grn
  pop = np.random.randn(pop_size, grn_size+2, grn_size).astype(np.float64)

  curr = 0
  worst= -num_cells*dev_steps
  geneid = 1
  
  #Creating start expression pattern
  seeds=[]
  inputs=[]
  for seed_int in seed_ints:
    #Make seeds, 1024 is one-hot
    start_pattern = helper.seedID2string(seed_int, num_cells)
    seeds.append(start_pattern)
    #Make starting expression for whole population
    start_expression = helper.seed2expression(start_pattern, pop_size, num_cells, grn_size, geneid)
    inputs.append(start_expression)

  #Creating targets
  targets=[]
  seeds_ints=[]
  for idx, seed in enumerate(seeds):
    targets.append(helper.rule2targets_wrapped_wstart(int(rules[idx]), L=dev_steps+1, N=num_cells, start_pattern=seed))
    binary_string = ''.join(seed.astype(str))
    seeds_ints.append(int(binary_string, 2)) #for file naming
  seeds_id='-'.join([str(number) for number in seeds_ints]) #id of start pattern for each season
  rules_id='-'.join([str(number) for number in rules])

  #Logging targets
  max_fits = []
  ave_fits = []
  gensin=[10,100,290]
  swichesin=[3,10,15,25,33]
  saveat=[10,30,50]
  #saveat=[s*300+g for s in swichesin for g in gensin]
  #save_freq=int(num_generations/5)
  
  filename = f"{folder}/stats_{season_len}_{rules_id}_{seeds_id}_{job_array_id}"

  #Defining variables
  selection_size=int(pop_size*selection_prop)
  num_child = int(pop_size / selection_size) - 1
  tot_children = num_child * selection_size
  num_genes_mutate = int((grn_size + 2) * grn_size * tot_children * mut_rate)

  # Main for loop
  for gen in range(num_generations):

    # Generating phenotypes
    #Return [pop_size, dev_stepss+1, num_cellsxgrn_size] np.float64 array
    phenos = helper.develop(inputs[curr], pop, dev_steps, pop_size, grn_size, num_cells)
    #get second gene for each cell only, the one I decided will matter for the fitness
    #pop_size, dev_steps, NCxNG
    p=phenos[:,:,1::grn_size]

    #Calculating fitnesses
    fitnesses = []
    for target in targets:
      temp_fitnesses = helper.fitness_function_ca(p, target)
      temp_fitnesses=1-(temp_fitnesses/worst) #0-1 scaling
      fitnesses.append(temp_fitnesses)

    #Selection
    perm = np.argsort(fitnesses[curr])[::-1]

    #Logging
    best_grn = pop[perm[0]]
    
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
      curr = (curr + 1) % len(targets)

    #if gen % save_freq == save_freq - 1:
    if gen in saveat:
      with open(filename+"_best_grn.txt", 'a') as f:
        np.savetxt(f, best_grns, newline=" ")
      with open(filename+"_both_fits.txt", 'a') as f:
        np.savetxt(f, np.array(fitnesses), newline=" ")

  with open(filename+"_maxfits.txt", 'w') as f:
    np.savetxt(f, max_fits, newline=" ")
  with open(filename+"_avefits.txt", 'w') as f:
    np.savetxt(f, ave_fits, newline=" ")

  return max_fit

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument('--pop_size', type=int, default=1000, help="Population size")
  parser.add_argument('--grn_size', type=int, default=22, help="GRN size") 
  parser.add_argument('--num_cells', type=int, default=22, help="Number of cells") 
  parser.add_argument('--dev_steps', type=int, default=22, help="Number of developmental steps") 

  parser.add_argument('--selection_prop', type=float, default=0.1, help="Percent pruncation") 
  parser.add_argument('--mut_rate', type=float, default=0.1, help="Number of mutations") 
  parser.add_argument('--mut_size', type=float, default=0.5, help="Size of mutations") 
  parser.add_argument('--num_generations', type=int, default=9899, help="Number of generations")
  parser.add_argument('--season_len', type=int, default=300, help="season length")

  parser.add_argument('--seed_ints', nargs='+', default=[1024], help='List of seeds in base 10')
  parser.add_argument('--rules', nargs='+', default=[30], help='List of rules')

  parser.add_argument('--job_array_id', type=int, default=0, help="Job array id to distinguish runs")

  args = parser.parse_args()

  #69904,149796
  #1024
  #to_seed = lambda n, N : np.array(list(map(int, format(n, f"0{N}b"))))

  #Writing to file
  folder_name = "results_new_rules"
  folder = helper.prepare_run(folder_name)
  args.folder = folder

  args.num_cells = args.dev_steps

  #Make sure that user provided a rule and a seed for each alternative environment
  assert len(args.rules) == len(args.seed_ints), f"Num rules {len(args.rules)} != num seeds {len(args.seed_ints)}"

  print("running code", flush=True)
  score=evolutionary_algorithm(**vars(args))
  
