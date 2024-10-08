import numpy as np
import argparse
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

import helper

"""#Evolutionary algorythm"""

def evolutionary_algorithm(pop_size, grn_size, num_cells, dev_steps, mut_rate, num_generations, mylambda, selection_prop, rules, mut_size, folder, seed_ints, season_len, job_array_id):

  #Setting up
  rules_str=''.join(str(num) for num in rules)
  seedints_str=''.join(str(num) for num in seed_ints)
  mut_rate_str = str(mut_rate)[0] + str(mut_rate)[-1]
  mut_size_str = str(mut_size)[0] + str(mut_size)[-1]
  selection_prop_str = str(selection_prop)[0] + str(selection_prop)[-1]
  assert (len(str(mut_rate)) == 3) & (len(str(mut_size)) == 3) & (len(str(selection_prop)) == 3), f"mut_rate, mut_size, or selection_prop not in the x.y format"
  
  rand_seed_str = str(pop_size)+str(grn_size)+str(num_cells)+str(dev_steps)+mut_rate_str+mut_size_str+selection_prop_str+str(season_len)+rules_str+seedints_str+str(job_array_id)
  print(int(rand_seed_str))
  rand_seed = helper.map_to_range(int(rand_seed_str))
  print(rand_seed)
  
  np.random.seed(rand_seed)

  with open("experiment_seeds.txt", 'a') as f:
    np.savetxt(f, [np.array([rand_seed_str,str(rand_seed)])], delimiter=",", fmt="%s")
  
  #Creating population

  #adding a value for each gene in the grn which will be added after matrix multiplication
  #by the rest of the values in the grn
  pop = np.random.randn(pop_size, grn_size+2, grn_size).astype(np.float64)

  curr = 0
  worst= -num_cells*dev_steps #(dev_steps+1) if rerunning everything. bug but not important
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
  where_overlap = np.where(targets[0]==targets[1])
  where_no_overlap = np.where(targets[0]!=targets[1])

  #Logging
  max_fits = []
  ave_fits = []
  best_std = []
  pheno_stds = []

  #gensin=[10,100,290]
  #swichesin=[3,10,15,25,32]
  #saveat=[s*300+g for s in swichesin for g in gensin]
  saveat = list(range(num_generations))
  
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

    if gen > 0:
      child_phenotypes = p[children_locs] 
      # inner most list: first: first born of each parent, second: second borns of each parent, etc.
      # so it is NOT all kids of 1 parent, then the other parent, etc.
      reshaped=np.reshape(child_phenotypes, (num_child, len(parent_locs), (dev_steps+1)*num_cells))
      #reshaped is num child per parent, num parents, (dev_steps+1)*num_cells shaped. 
      # so [:,0,:] is all kids of one parent
      pheno_std=np.std(reshaped,axis=0) #one std for each of the parents, so pop_size*trunc_prop now 10
      pheno_std = pheno_std.mean(1).mean() #first averaged across cells, then averaged across individuals in the population
      pheno_stds.append(pheno_std)
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
      best_std.append(np.max(combined_std))
      best_std_id = np.argmax(combined_std)
      best_std_grn = pop[parent_locs[best_std_id]]


    #Calculating fitnesses
    fitnesses = []
    for target in targets:
      temp_fitnesses = helper.fitness_function_ca(p, target)
      temp_fitnesses=1-(temp_fitnesses/worst) #0-1 scaling
      fitnesses.append(temp_fitnesses)
    
    #L1 regularization 
    scaling=0.001 #0.001 makes it into similar range as fitness
    #mylambda = 0.5 #importance of regularization, 1 means that weight sizes are as important as fitness
    pop_abs = np.abs(pop)
    pop_abs = np.reshape(pop_abs, (pop_abs.shape[0],pop_abs.shape[1]*pop_abs.shape[2] ))
    pop_sum = pop_abs.sum(axis=1) * scaling * mylambda

    fitnesses_to_use = fitnesses[curr] #- pop_sum

    #Selection
    perm = np.argsort(fitnesses_to_use)[::-1]

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

    # Logging lineages, forward pointing. 
    # this generation, these are the parents, this is where their kids will go
    edges = []
    for p in parent_locs:
      edges.append([(gen,p),(gen+1,p)]) 
      #each parent stays in the population where it was
    for idx, p in enumerate(np.tile(parent_locs,num_child)):
      #np tile makes it so that it is parent 1, parent 2, parent 1, parent 2, etc.
      edges.append([(gen,p),(gen+1,children_locs[idx])])
      #parent 1 has a kid, then parent 2 has a kid, then parent 1 has a kid, then parent 2 has a kid, etc.
    #edges contains ids, which are also the indicies of the individuals in the population
    #so in the fitnesses variable for example, we will know if generation 2 10th individual (2,10) in the edge list, 
    #someone with no kids, had a good fitness or not, by looking at generation 2, 10th idx in the fitnesses list.

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
        np.savetxt(f, best_grn, newline=" ")
      with open(filename+"_both_fits.txt", 'a') as f:
        np.savetxt(f, np.array(fitnesses), newline=" ")
      save_edges=np.array(edges)
      save_edges=np.reshape(save_edges, (pop_size,4))
      with open(filename+"_edges.txt", 'a') as f:
        np.savetxt(f, save_edges, newline=" ")
      if gen > 0:
        with open(filename+"_best_grn_std.txt", 'a') as f:
          np.savetxt(f, best_std_grn, newline=" ")

  with open(filename+"_maxfits.txt", 'w') as f:
    np.savetxt(f, max_fits, newline=" ")
  with open(filename+"_avefits.txt", 'w') as f:
    np.savetxt(f, ave_fits, newline=" ")
  with open(filename+"_beststd.txt", 'w') as f:
    np.savetxt(f, best_std, newline=" ")
  with open(filename+"_pheno_stds.txt", 'w') as f:
    np.savetxt(f, pheno_stds, newline=" ")
  

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
  parser.add_argument('--num_generations', type=int, default=10, help="Number of generations") #19799
  parser.add_argument('--mylambda', type=float, default = 0.1, help="lambda for L1 or L2 regularization")
  parser.add_argument('--season_len', type=int, default=300, help="season length")

  parser.add_argument('--seed_ints', nargs='+', default=[1024,1024], help='List of seeds in base 10')
  parser.add_argument('--rules', nargs='+', default=[30,30], help='List of rules')

  parser.add_argument('--job_array_id', type=int, default=0, help="Job array id to distinguish runs")

  args = parser.parse_args()

  #69904,149796
  #1024
  #to_seed = lambda n, N : np.array(list(map(int, format(n, f"0{N}b"))))

  #Writing to file
  folder_name = "results_testing_saving"
  folder = helper.prepare_run(folder_name)
  args.folder = folder

  args.num_cells = args.dev_steps

  #Make sure that user provided a rule and a seed for each alternative environment
  assert len(args.rules) == len(args.seed_ints), f"Num rules {len(args.rules)} != num seeds {len(args.seed_ints)}"

  print("running code", flush=True)
  evolutionary_algorithm(**vars(args))

  edges = np.loadtxt("results_testing_saving/stats_300_30-30_1024-1024_0_edges.txt")
  num_generations=int(edges.shape[0]/4/args.pop_size)
  edges = np.reshape(edges, (num_generations*args.pop_size, 2, 2))  

  # ----------------------
  # Parameters
  num_gen_start = 0
  num_gen_stop = 5 #args.num_generations
  num_gens_show = num_gen_stop - num_gen_start
  new_edges = []
  for e in edges:
    if (e[0][0] >= num_gen_start) & (e[0][0] < num_gen_stop):
      #e[0][0] = e[0][0] - num_gen_start
      #e[1][0] = e[1][0] - num_gen_start
      new_edges.append([(e[0][0]- num_gen_start, e[0][1]),(e[1][0]- num_gen_start, e[1][1])])
  edges = new_edges.copy()

  # Set up network
  num_rows = num_gens_show+1
  num_columns = args.pop_size
  G = nx.Graph()
  # Add nodes with specified positions
  pos = {}
  for i in range(num_rows):
      for j in range(num_columns):
          node = (i, j)
          G.add_node(node)
          pos[node] = (j, -i)  # Assigning positions based on rows and columns
  
  # Add edges from the edges variable
  mydic=defaultdict(list) #make dictionary to keep track of OG where it comes from
  for i in range(len(edges)):
    G.add_edge(edges[i][0],edges[i][1])
    if edges[i][0][0] == 0: #if it is the first generation
      mydic[edges[i][0]].append(edges[i][1])
    else:
      for k in mydic.keys():
        if edges[i][0] in mydic[k]:
          mydic[k].append(edges[i][1])

  #coloring
  def generate_colors(n, cmap='viridis'):
    color_map = plt.get_cmap(cmap)
    colors = [color_map(i / n) for i in range(n)]
    return colors
  
  import colorsys
  def generate_colors(n):
    # Generate colors in HSL
    colors = []
    for i in range(n):
        # Generate hue, saturation, and lightness
        hue = i / n  # Normalize hue
        saturation = 0.7  # Set saturation to 70%
        lightness = 0.5  # Set lightness to 50%
        
        # Convert HSL to RGB
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(rgb)
    return colors
  
  colors = generate_colors(args.pop_size)
  node_colors = []
  for c in colors:
    node_colors.append(c) #colors for the first generation
  color_dic = {}
  for idx, node in enumerate(G.nodes()):
    if node[0] == 0:
      color_dic[node] = colors[idx] #color assigned to each original parent
  for node in G.nodes():
    if node[0] != 0:
      for k in mydic.keys():
        if node in mydic[k]:
          node_colors.append(color_dic[k]) #assign color based on original parent

  plt.figure(figsize=(12, 8))

  #Draw network
  nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=20, linewidths=0.0)
  nx.draw_networkx_edges(G, pos, node_size=20, alpha=0.1)
  plt.title("Lineages")
  plt.show()

  #Draw matrix
  i=0
  color_matrix = np.zeros((num_rows, num_columns, 3))
  for r in range(num_rows):
    for c in range(num_columns):
      color_matrix[r, c] = node_colors[i]
      i+=1

  color_matrix = np.repeat(color_matrix, 100, axis=0)
  plt.imshow(color_matrix)
  plt.show()

  #Draw trend, num lineages
  pcs = np.reshape(np.array(node_colors),(num_gens_show+1,args.pop_size,3))
  unique_counts = []
  for subarray in pcs:
    unique_subarrays = np.unique(subarray, axis=0)
    unique_counts.append(len(unique_subarrays))

  plt.plot(unique_counts)
  plt.show()

  
    
