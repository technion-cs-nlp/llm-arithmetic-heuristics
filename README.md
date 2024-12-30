# llm-arithmetic-heuristics

This repository contains the code for the experiments (and [website](https://technion-cs-nlp.github.io/llm-arithmetic-heuristics/)) of the llm arithmetic heuristics project.


## Repository structure
* The notebook files contain experimentation code and code to generate the data, results and figures for the paper. Specifically, `llm-arithmetic-analysis-main.ipynb` contains most of the code for general LLM arithmetic analysis presented in the paper, and `pythia-heuristics-analysis-notebook.ipynb` contains the relevant code for experiments across checkpoints, presented in section 5 of the paper.
* All script files (`script_.*.py`) contain a separate-file version of some of the code from the notebooks, to run as GPU jobs.
* Other files contain processes used in experiments (activation patching, faithfulness evaluations, heuristic classification algorithm, etc).
* `docs` contains code for the project website.
