# Learning Market Equilibria from Samples

This repository contains code for the paper ["The Price is (Probably) Right: Learning Market Equilibria from Samples"](https://arxiv.org/abs/2012.14838)

## Market Generation

The [Data Creation](DataCreation) folder contains all the code needed to generate the different markets. It generates both txt and csv versions of the data files which contains the value each player has for each good. The code uses the ratings.csv file from the [MovieLens dataset](https://grouplens.org/datasets/movielens/).

- data1 contains data for the Sellers' Market (n = 50, k = 30)
- data2 contains data for the Buyers' Market (n = 30, k = 50)
- data3 contains data for the Balanced Market (n = 40, k = 40)

## General Simulation Information

All of the code uses [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) to speed up simulations. However, the functions themselves do not use this library. So, the functions can be used in other serial code to compute market outcomes.

The data structure used to represent allocations and samples is a simple binary matrix.

The [Unit Demand](UnitDemand), [Additve](Additive) and [Submod](Submod) folders contain all the code needed to compute equilibria as well as summarise the computed equilibria for unit demand markets, additive markets and submodular markets respectively. The program files used for each market are unitdemand_parallel.py, additive_parallel.py and submod_parallel.py. These files take as input the following (in this specific order)

- Number of players
- Number of goods
- No. of iterations
- Number of randomized schedules to average results for (Only for submod_parallel.py)
- Maximum number of samples
- Binary variable asking for budget normalisation of valuations (0 for False, 1 for True)
- Threshold Value (Only for submod_parallel.py)
- Input Data File Address
- Output File Address

The program generates an output text file summarising the allocation output by the algorithms used in the paper. This output file is then used by the Graph.ipynb file in each folder to create graphs.

In case of any queries, feel free to contact vviswanathan (at) umass.edu
