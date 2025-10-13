# THPrompt

We provide the code (in PyTorch) for our paper "Dual-Prompt Tuning for Spatial-Temporal Graphs".

## Usage

### 1\. Download Data

From [JODIE dataset website](https://snap.stanford.edu/jodie/), download the required data files (e.g., `wikipedia.csv`, `reddit.csv`). Place the downloaded files into the `processed/` directory.

### 2\. Process Data

Run the following scripts to preprocess the raw data.

```bash
# The --data argument can be changed to 'reddit' or other datasets
python utils/preprocess_data.py --data wikipedia --bipartite
python utils/downstream_process.py
```

The pre-training part of the code references the paper "Node-Time Conditional Prompt Learning in Dynamic Graphs".

### 3\. Pre-train the Model

Execute the following command to pre-train the model.

```bash
python pre-training.py --use_memory
```

### 4\. Run Downstream Task

Use the pre-trained model to perform the downstream task.

```bash
python downstream_task.py --use_memory
```
