# Information Retrieval and Search Engines Project 2020-2021

This repository contains my solutions for the project of the course IRSE,
part of the Master of Engineering: Computer Science program at KU Leuven.
It includes the source code of my solution, as well as the assignment and 
the report of my solution. Both parts of the project are seperated on two
git branches: part1 & part2

# Running

## Dataset

Add the dataset to the project as instructed in the assignment to a
directory `NUS-WIDE-Lite/`. Then run the following python script to save
the dataset. It will create and store 3 subsets (train, val and test) of 
the dataset in `datasets/`.

```
python dataset_saver.py
```

## Training

Train a model and save the experiment with the a name of choice
(e.g. "my-experiment"). The model's weights and other relevant information 
of your experiment will be saved in `exp/my-experiment/`.

```
python train.py --exp "my-experiment"
```

## Testing

Calculate the MRR of your trained model. This script will also precompute 
all projections required for retrieval and store it in `exp/my-experiment/`.

```
python test.py --exp "my-experiment"
```

## Retrieval

Retrieve the 10 most relevant images for a given query string 
(e.g. "yellow flower") with your trained model. Make sure to first run the
testing script.

```
python retrieve.py "yellow flower" --exp "my-experiment"
```
