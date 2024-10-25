import os
import argparse
import numpy as np
from pathlib import Path

import helper

class JSONLogger:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def append(self, results: Any) -> None:
        with self.filepath.open("a") as f:
            f.write(json.dumps(results) + "\n")

    def read(self) -> list:
        results = []
        with open(self.filepath) as f:
            results = [json.loads(line) for line in f]
        return results

def score(phenos, target):
    # TODO: phenos has shape 100, 23, 22 sooo... (23 - 1) = 22 dev_steps?
    _, dev_steps, num_cells = phenos.shape
    worst = -num_cells * (dev_steps - 1)
    prefitnesses = helper.fitness_function_ca(phenos, target)
    fitnesses = 1 - (prefitnesses / worst)  # 0-1 scaling
    return fitnesses

def explore_noise(filename, seed, rule, noise_scaling, candidate_idx, grn_size = 22, num_cells = 22, dev_steps = 22, geneid=1, nclones=1000):
    grns = np.loadtxt(filename)
    num_grns = int(grns.shape[0] / (grn_size + 2) / grn_size)
    grns = grns.reshape(num_grns, grn_size + 2, grn_size)

    candidate = grns[candidate_idx]
    clones = np.tile(candidate, (nclones, 1, 1))
    noise = np.random.randn(*clones.shape) * noise_scaling
    clones += noise
    clones[0] = candidate
    target, phenos = helper.get_pop_TPF(
        clones, len(clones), num_cells, grn_size, dev_steps, geneid, rule, seed, seed
    )
    fitnesses = score(phenos, target)
    return phenos, target, fitnesses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--pop_size', type=int, default=1000, help="Population size")
    parser.add_argument('--grn_size', type=int, default=22, help="GRN size") 
    parser.add_argument('--num_cells', type=int, default=22, help="Number of cells") 
    parser.add_argument('--dev_steps', type=int, default=22, help="Number of developmental steps") 

    parser.add_argument('--selection_prop', type=float, default=0.1, help="Percent pruncation") 
    parser.add_argument('--mut_rate', type=float, default=0.1, help="Number of mutations") 
    parser.add_argument('--mut_size', type=float, default=0.5, help="Size of mutations") 
    parser.add_argument('--num_generations', type=int, default=9899, help="Number of generations") #19799
    
    parser.add_argument('--season_len', type=int, default=100000, help="season length")
    parser.add_argument('--seed_ints', nargs='+', default=[69904,149796], help='List of seeds in base 10')
    parser.add_argument('--rules', nargs='+', default=[30,30], help='List of rules')

    parser.add_argument('--job_array_id', type=int, default=0, help="Job array id to distinguish runs")
    parser.add_argument('--exp_type', type=str, default="variable", help="Variable or static")

    args = parser.parse_args()
    root="~/scratch/detailed_save/"

    if args.exp_type == "variable":
        filename = os.path.expanduser(f"{root}/variable/stats_{args.season_len}_{args.rules[0]}-{args.rules[0]}_{args.seed_ints[0]}-{args.seed_ints[1]}_{args.job_array_id}_best_grn.txt")
    else:
        filename = os.path.expanduser(f"{root}/static/stats_{args.season_len}_{args.rules[0]}_{args.seed_ints[0]}_{args.job_array_id}_best_grn.txt")

    rule = args.rules[0]
    nrows = 20
    nclones = nrows * nrows
    phenos, target, fitnesses = explore_noise(
        filename, seed=args.seed_ints[0], noise_scaling=0.5, rule=rule, nclones=nrows * nrows
    )

    M = 0.3
    N = 20
    log = JSONLogger("noise_data.jsonl")
    for filename in tqdm(files, position=0):
        params = GRNFilenameParser.parse(filename)
        data = []
        noise_levels = np.linspace(0, M, N).tolist()
        for noise_scaling in tqdm(noise_levels, position=1, leave=False):
            phenos, target, fitnesses = explore_noise(
                filename,
                seed=params.seeds[0],
                noise_scaling=noise_scaling,
                rule=params.rules[0],
                nclones=nrows * nrows,
            )
            data.append(np.sort(fitnesses).tolist())
        to_log = {
            "file": filename.as_posix(),
            "params": asdict(params),
            "data": data,
            "noise_levels": noise_levels,
        }
        log.append(to_log)