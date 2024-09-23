# Code Structure

```shell
.
├── Bonus Task
├── config
│   └── param.yaml
├── data // data files should be put here
│   ├── BonusTask
│   ├── Task 1
│   └── Task 2
├── Report
├── Task1
│   ├── results
│   ├── task1.ipynb
│   └── task1.py
├── Task2
│   ├── results
│   ├── task2.ipynb
│   └── task2.py
├── README.md
├── requirements.txt
└── utils.py // utility functions
```

The results from previous runs can be directly accessed in the `Task1/results/archive` and `Task2/results/archive` directories.

# Installation

```shell
python3 -m venv .venv
source .venv/bin/activate # for linux or macos
# source .venv/bin/activate.ps1 # for windows
pip install -r requirements.txt
```

# Quick Start

1. Jupyter Notebook
   
   Open and run. Remeber to select the created virtual environment as the kernel.

2. Python file
   
   ```shell
   python3 Task1/task1.py
   python3 Task2/task2.py
   ```

3. Bonus Task
   
   ```shell
   # cd to ./Bonus Task
   ./start.sh <sequence_number>
   # for example:
   # ./start.sh 1
   ```
