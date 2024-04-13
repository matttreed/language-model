# CS336 Spring 2024 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2024_assignment1_basics.pdf](./cs336_spring2024_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

0. Set up a conda environment and install packages:

``` sh
conda create -n cs336_basics python=3.10 --yes
conda activate cs336_basics
pip install -e .'[test]'
```

1. Run unit tests:

``` sh
pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

2. Download the TinyStories data and a subsample of OpenWebText:

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
log in to head node
ssh c-mattreed@ad12a3ca-hn.cloud.together.ai

run debug interactive shell
srun --partition=interactive --pty bash

run cpu batch job 
sbatch scripts/serialize_data.sh 

monitor jobs
squeue -u $USER

cancel job
scancel <jobid>


Examples of flags you can pass to srun or put in your sbatch script:
--gpus=1 to request a GPU
--mem=50G to request 50GB of memory, or --mem=100G to request 100GB of memory
--time=00:20:00 to request 20 minutes for your job.
--cpus-per-task=8 to request 8 CPU cores.



unfortunately, i don't think there's anything we can do on our side to enable private forking---I think this is something that github just doesn't support (let me know if i'm wrong about that). The steps at https://gist.github.com/0xjac/85097472043b697ab57ba1b1c7530274 should be suitable, though. In a nutshell:
Make an empty private repo
it should allow you to import code from another repo. do that and paste in the URL of the assignment.
clone your new private repo locally to your desktop. e.g., if my repo is nfliu/a1 , git remote -v  after cloning would show one remote named "origin" pointing to nfliu/a1
now, add a new remote for the upstream repository: git remote add upstream git@github.com:stanford-cs336/spring2024-assignment1-basics.git  (you may need to edit this command if you use HTTPS instead of SSH)
at this point, if i have pushed a bunch of changes to my nfliu/a1  main branch, i can stack those changes on top of any changes that might come in on the upstream with git fetch upstream  followed by git rebase upstream/master . i expect that you won't need to solve many (if any) merge conflicts, since the changes we make are localized to test code that you probably aren't touching anyway.