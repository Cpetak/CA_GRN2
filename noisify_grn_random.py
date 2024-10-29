import os
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List
from typing import Any
import json
import re
from tqdm import tqdm

import helper

@dataclass
class GRNFileInfo:
    """Store parsed information from a GRN results filename."""

    iterations: int
    rules: List[int]
    seeds: List[int]
    repetition: int
    filename: str

class GRNFilenameParser:
    @staticmethod
    def parse(filename: Path) -> GRNFileInfo:
        filename_str = filename.as_posix()

        # Try the complex format first (with paired numbers)
        complex_pattern = r"stats_(\d+)_(\d+-\d+)_(\d+-\d+)_(\d+)_best_grn\.txt$"
        complex_match = re.search(complex_pattern, filename_str)

        if complex_match:
            iterations = int(complex_match.group(1))
            rules = [int(x) for x in complex_match.group(2).split("-")]
            seeds = [int(x) for x in complex_match.group(3).split("-")]
            repetition = int(complex_match.group(4))
            return GRNFileInfo(iterations, rules, seeds, repetition, filename_str)

        # Try the simple format
        simple_pattern = r"stats_(\d+)_(\d+)_(\d+)_(\d+)_best_grn\.txt$"
        simple_match = re.search(simple_pattern, filename_str)

        if simple_match:
            iterations = int(simple_match.group(1))
            rules = [int(simple_match.group(2))]
            seeds = [int(simple_match.group(3))]
            repetition = int(simple_match.group(4))
            return GRNFileInfo(iterations, rules, seeds, repetition, filename_str)

        raise ValueError(f"Unrecognized filename format: {filename}")

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
    # TODO: phenos has shape 100, 23, 22 sooo... (23 - 1) = 22 dev_steps? yes because first row is good for sure so dev_step should be 22 in worst calculation
    _, dev_steps, num_cells = phenos.shape
    worst = -num_cells * (dev_steps -1)
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
    target, phenos, fitnesses = helper.get_pop_TPF(
        clones, len(clones), num_cells, grn_size, dev_steps, geneid, rule, seed, seed
    )
    #fitnesses = score(phenos, target)
    return phenos, target, fitnesses

def explore_noise_random(filename, seed, rule, noise_scaling, candidate_idx, grn_size = 22, num_cells = 22, dev_steps = 22, geneid=1, nclones=1000):
    #grns = np.loadtxt(filename)
    grns = np.random.randn(10, grn_size+2, grn_size).astype(np.float64)
    #num_grns = int(grns.shape[0] / (grn_size + 2) / grn_size)
    #grns = grns.reshape(num_grns, grn_size + 2, grn_size)

    candidate = grns[candidate_idx]
    clones = np.tile(candidate, (nclones, 1, 1))
    noise = np.random.randn(*clones.shape) * noise_scaling
    clones += noise
    clones[0] = candidate
    target, phenos, fitnesses = helper.get_pop_TPF(
        clones, len(clones), num_cells, grn_size, dev_steps, geneid, rule, seed, seed
    )
    #fitnesses = score(phenos, target)
    return phenos, target, fitnesses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--rule', type=int, default=30, help='List of rules')    
    parser.add_argument('--exp_type', type=str, default="variable", help="Variable or static")
    parser.add_argument('--candidate_idx', type=str, default="-1", help="Which generation to check")

    args = parser.parse_args()

    if args.exp_type == "variable":
        args.season_len = 300
    else:
        args.season_len = 100_000

    root="~/scratch/detailed_save/"
    dir_path = Path(f"~/scratch/detailed_save/{args.exp_type}/").expanduser()
    print(dir_path)

    print(args.season_len, args.rule)
    files = [file for file in dir_path.iterdir() if file.is_file() and file.name.startswith(f"stats_{args.season_len}_{args.rule}") and file.name.endswith("_best_grn.txt")]
    #out_filename = os.path.expanduser(f"{root}/noise_results/stats_{args.rule}_{args.exp_type}_{args.candidate_idx}_noise_data.jsonl")
    out_filename = os.path.expanduser(f"{root}/noise_results/stats_{args.rule}_randomGRN_noise_data.jsonl")

    print(files)

    M = 0.3 #max noise
    N = 20 #number of noise levels to test
    nrows = 100 #x**2 is number of clones
    log = JSONLogger(out_filename)

    for filename in tqdm(files, position=0):
        print(filename)
        params = GRNFilenameParser.parse(filename)
        data = []
        noise_levels = np.linspace(0, M, N).tolist()
        for noise_scaling in tqdm(noise_levels, position=1, leave=False):
            phenos, target, fitnesses = explore_noise_random(
                filename,
                seed=params.seeds[0],
                noise_scaling=noise_scaling,
                candidate_idx = int(args.candidate_idx),
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