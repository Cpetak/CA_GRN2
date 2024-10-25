#%%

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import re
from typing import List
from typing import Any

from matplotlib.cm import viridis
import matplotlib.pyplot as plt
from numba import njit, prange
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
import torchvision
from tqdm import tqdm

ALPHA = 10


# plotting helper using the convenient make_grid from pytorch
def phenos2grid(numpy_tensor, nrow=10, padding=2):
    # Shape : (100, 23, 22)
    torch_tensor = torch.from_numpy(numpy_tensor)
    # Shape : (100, 1, 23, 22)
    torch_tensor = torch_tensor.unsqueeze(1)
    # Shape : (3, 252, 242) WITH PADDING
    grid_img = torchvision.utils.make_grid(
        torch_tensor, nrow=nrow, padding=padding, pad_value=0.5
    )
    # Shape : (252, 242, 3) to plot with imshow
    result = grid_img.permute(1, 2, 0).numpy()
    return result


@njit("f8[:,:](f8[:,:],i8, i8)")
def sigmoid(x, a, c):
    return 1 / (1 + np.exp(-a * x + c))


def fitness_function_ca(phenos, targ):
    """
    Takes phenos which is pop_size * iters+1 * num_cells and targ which is iters+1 * num_cells
    Returns 1 fitness value for each individual, np array of size pop_size
    """
    return -np.abs(phenos - targ).sum(axis=1).sum(axis=1)


def seedID2string(seed_int, num_cells):
    # takes an integer, turns it into a starting pattern
    binary_string = bin(int(seed_int))[2:]
    binary_list = [int(digit) for digit in binary_string]
    start_pattern = np.array(binary_list)
    start_pattern = np.pad(
        start_pattern,
        (num_cells - len(start_pattern), 0),
        "constant",
        constant_values=(0),
    )
    return start_pattern


def seed2expression(start_pattern, pop_size, num_cells, grn_size, geneid):
    # takes a starting pattern and makes a population of starting gene expressions
    start_gene_values = np.zeros((pop_size, int(num_cells * grn_size)))
    start_gene_values[:, geneid::grn_size] = start_pattern
    start_padded_gene_values = np.pad(start_gene_values, [(0, 0), (1, 1)], "wrap")
    start_padded_gene_values = np.float64(start_padded_gene_values)
    return start_padded_gene_values


# DO THE MULTICELLULAR DEVELOPMENT
@njit("f8[:](f8[:], f8[:,:], i8, i8)")
# Make sure that numpy imput in foat64! Otherwise this code breaks
def update_with_grn(padded_gene_values, grn, num_cells, grn_size):
    """
    Gene expression pattern + grn of a single individual -> Next gene expression pattern
    Takes
    - padded_gene_values: np array with num_genes * num_cells + 2 values
    Gene 1 in cell 1, gene 2 in cell 1, etc then gene 1 in cell 2, gene 2 in cell 2... plus left-right padding
    - grn: np array with num_genes * num_genes +2 values, shape of the GRN
    """
    # This makes it so that each cell is updated simultaneously
    # Accessing gene values in current cell and neighbors
    windows = np.lib.stride_tricks.as_strided(
        padded_gene_values, shape=(num_cells, grn_size + 2), strides=(8 * grn_size, 8)
    )
    # Updating with the grn
    next_step = windows.dot(grn)
    c = ALPHA / 2
    next_step = sigmoid(next_step, ALPHA, c)

    # Returns same shape as padded_gene_values
    return next_step.flatten()


@njit("f8[:](f8[:], f8[:,:], i8, i8)")
# Make sure that numpy imput in foat64! Otherwise this code breaks
def update_internal(padded_gene_values, grn, num_cells, grn_size):
    """
    Gene expression pattern + grn of a single individual -> Next gene expression pattern
    Takes
    - padded_gene_values: np array with num_genes * num_cells + 2 values
    Gene 1 in cell 1, gene 2 in cell 1, etc then gene 1 in cell 2, gene 2 in cell 2... plus left-right padding
    - grn: np array with num_genes * num_genes +2 values, shape of the GRN
    """
    # Updating with the internal grn
    internal_grn = grn[1:-1, :]
    gene_vals = padded_gene_values[1:-1].copy()
    gene_vals = gene_vals.reshape(num_cells, grn_size)
    next_step = gene_vals.dot(internal_grn)
    c = ALPHA / 2
    next_step = sigmoid(next_step, ALPHA, c)

    # Returns same shape as padded_gene_values
    return next_step.flatten()


# Might be faster non-parallel depending on how long computing each individual takes!
@njit(f"f8[:,:,:](f8[:,:], f8[:,:,:], i8, i8, i8, i8)", parallel=True)
def develop(padded_gene_values, grns, iters, pop_size, grn_size, num_cells):
    """
    Starting gene expression pattern + all grns in the population ->
    expression pattern throughout development for each cell for each individual
    DOES NOT assume that the starting gene expression pattern is the same for everyone
    returns tensor of shape: [POP_SIZE, N_ITERS+1, num_cellsxgrn_size]
    N_ITERS in num developmental steps not including the initial step
    """
    NCxNGplus2 = padded_gene_values.shape[1]
    history = np.zeros((pop_size, iters + 1, NCxNGplus2 - 2), dtype=np.float64)

    # For each individual in parallel
    for i in prange(pop_size):
        # IMPORTANT: ARRAYS IN PROGRMING when "copied" just by assigning to a new variable (eg a=[1,2,3], b = a)
        # Copies location and so b[0]=5 overwrites a[0] too! Need .copy() to copy variable into new memory location
        grn = grns[i]
        state = padded_gene_values[i].copy()
        history[i, 0, :] = state[1:-1].copy()  # saving the initial condition
        # For each developmental step
        for t in range(iters):
            # INTERNAL
            state[1:-1] = update_internal(state, grn, num_cells, grn_size)
            # To wrap around, change what the pads are
            state[0] = state[-2]  # the last element of the output of update_with_grn
            state[-1] = state[1]
            # EXTERNAL
            state[1:-1] = update_with_grn(state, grn, num_cells, grn_size)
            # To wrap around, change what the pads are
            state[0] = state[-2]  # the last element of the output of update_with_grn
            state[-1] = state[1]
            history[i, t + 1, :] = state[1:-1].copy()
    return history


# WITH CUSTOM START
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


def get_pop_TPF(pop, pop_size, num_cells, grn_size, dev_steps, geneid, rule, seed_int):
    start_pattern = seedID2string(seed_int, num_cells)
    start_expression = seed2expression(
        start_pattern, pop_size, num_cells, grn_size, geneid
    )

    target = rule2targets_wrapped_wstart(
        int(rule), L=dev_steps + 1, N=num_cells, start_pattern=start_pattern
    )

    all_phenos = develop(
        start_expression, pop, dev_steps, pop_size, grn_size, num_cells
    )
    phenos = all_phenos[:, :, geneid::grn_size]

    return target, phenos


def score(phenos, target):
    # TODO: phenos has shape 100, 23, 22 sooo... (23 - 1) = 22 dev_steps?
    _, dev_steps, num_cells = phenos.shape
    worst = -num_cells * (dev_steps - 1)
    prefitnesses = fitness_function_ca(phenos, target)
    fitnesses = 1 - (prefitnesses / worst)  # 0-1 scaling
    return fitnesses


# %%

import os

# Parameters
rule = 22
grn_size = 22
num_cells = 22
dev_steps = 22
geneid = 1  # which gene was used to get fitness


def explore_noise(filename, seed, rule, noise_scaling, nclones=1000):
    grns = np.loadtxt(filename)
    num_grns = int(grns.shape[0] / (grn_size + 2) / grn_size)
    grns = grns.reshape(num_grns, grn_size + 2, grn_size)

    candidate = grns[-1]
    clones = np.tile(candidate, (nclones, 1, 1))
    noise = np.random.randn(*clones.shape) * noise_scaling
    clones += noise
    clones[0] = candidate
    target, phenos = get_pop_TPF(
        clones, len(clones), num_cells, grn_size, dev_steps, geneid, rule, seed
    )
    fitnesses = score(phenos, target)
    return phenos, target, fitnesses


# static if 100000
# filename = "./best_grn_results/stats_100000_102_149796_1_best_grn.txt"
# variable if 300
root="~/scratch/detailed_save/variable/"
filename = os.path.expanduser(f"{root}stats_300_102-102_69904-149796_9_best_grn.txt")
# filename = "./best_grn_results/stats_300_102-102_69904-149796_5_best_grn.txt"
seed_int = 69904
rule = 102
nrows = 20
nclones = nrows * nrows
phenos, target, fitnesses = explore_noise(
    filename, seed=seed_int, noise_scaling=0.5, rule=rule, nclones=nrows * nrows
)

grid = phenos2grid(phenos, nrow=nrows)
plt.figure(figsize=(10, 10))
plt.imshow(grid, cmap="viridis")
plt.axis("off")  # Hide the axis
plt.tight_layout()
plt.show()

plt.plot(np.sort(fitnesses), ".-")
plt.axhline(fitnesses[0], color="red", lw=0.5)
plt.grid()
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# %%


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


def test_parser(files):
    """Test the parser on sample filenames and print a summary."""
    results = []
    for file in files:
        try:
            info = GRNFilenameParser.parse(file)
            results.append((True, file))
        except ValueError as e:
            results.append((False, f"Error parsing {file}: {str(e)}"))

    # Print summary
    successful = sum(1 for r in results if r[0])
    print(f"Successfully parsed {successful}/{len(results)} files")

    # Print any failures
    failures = [r[1] for r in results if not r[0]]
    if failures:
        print("\nFailed to parse:")
        for failure in failures:
            print(f"  {failure}")

    # Print example parsed data
    if successful > 0:
        print("\nExample parsed data:")
        success_file = next(r[1] for r in results if r[0])
        info = GRNFilenameParser.parse(success_file)
        print(f"Filename: {success_file}")
        print(f"Iterations: {info.iterations}")
        print(f"Rules: {info.rules}")
        print(f"Seeds: {info.seeds}")
        print(f"Repetition: {info.repetition}")


files = list(Path("./best_grn_results").iterdir())

# Run the test
test_parser(files)

# %%


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


# %%


def custom_legend(ax, title, n=5):
    """
    Reduce the number of entries in the Legend,
    by only showing first,last and every n-th
    """
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) <= n + 2:
        return ax.legend(title=title)
    new_handles = [handles[0]] + handles[1:-1:n] + [handles[-1]]
    new_labels = [labels[0]] + labels[1:-1:n] + [labels[-1]]
    return ax.legend(new_handles, new_labels, title=title)


all_data = log.read()
for exp in all_data:
    data = exp["data"]
    name = Path(exp["file"]).name
    fig, ax = plt.subplots()
    for i, (fitnesses, noise) in enumerate(zip(data, np.linspace(0, M, N))):
        ax.plot(fitnesses, color=viridis((i + 1) / N), label=f"{noise:.3f}")
    # plt.axhline(fitnesses[0], color="red", lw=0.5)
    ax.set_ylim(0, 1.1)
    plt.grid()
    plt.ylim(0, 1.1)
    plt.ylabel("Fitness")
    plt.xlabel("Samples sorted by Fitness")
    custom_legend(ax, title="Noise scale:", n=3)
    plt.title(name)
    plt.tight_layout()
    # plt.savefig("plots/" + name + ".fitnesses.png", dpi=300)
    # plt.close()
    plt.show()

    fig, ax = plt.subplots()
    for i, fitnesses in enumerate(data):
        # label=f"{np.linspace(0,M,N)[i]:.3f}"
        ax.plot(np.array(fitnesses) - i / 80, color=viridis((i + 1) / N))
    # ax.annotate(
    #     "different levels of noise\n(displaced for visibility)",
    #     xy=(2300, 0.75),
    #     xytext=(1500, 0.3),
    #     arrowprops=dict(fc="gray", ec="None", shrink=0.01),
    # )
    plt.grid()
    plt.yticks([])
    plt.xticks([])
    plt.ylabel("Fitness")
    plt.xlabel("Samples sorted by Fitness")
    # plt.legend()
    plt.title(name)
    plt.tight_layout()
    # plt.savefig("plots/" + name + ".stagger.png", dpi=300)
    # plt.close()
    plt.show()
