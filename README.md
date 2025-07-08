# FREDA-CV

This repository contains the source code of our research article:

**Privacy Preserving Federated Unsupervised Domain Adaptation with Application to Age Prediction from DNA Methylation Data**

[FREDA-CV](https://github.com/mdppml/FREDA_CV) is a simplified and task-agnostic implementation of our federated domain adaptation framework originally introduced in the [FREDA](https://github.com/mdppml/FREDA) repository.

---

### Usage

The following arguments can be configured when running the `main.py` script:

| Argument                    | Description                                                                             | Default Value       |
|-----------------------------|-----------------------------------------------------------------------------------------|---------------------|
| `--setup`                   | Number of source clients to simulate.                                                   | `2`                 |
| `--use_precomputed_confs`   | Whether to use precomputed confidence scores.                                           | `True`              |
| `--cv_folds`                | Number of Folds to run cross validation for lambda prediction.                          | `5`                 |
| `--use_precomputed_lambdas` | Whether to use precomputed optimal lambdas.                                             | `True`              |
| `--lambda_path`             | Path to a text file containing lambda values. If not provided, default values are used. | `None`              |
| `--home_path`               | Root directory for the project. Can be set to any desired path.                         | `Current directory` |
| `--alpha`                   | Weighting factor for the loss function.                                                 | `0.8`               |
| `--epochs`                  | Number of local training epochs.                                                        | `20`                |
| `--global_iterations`       | Number of global iterations.                                                            | `100`               |
| `--lr_init`                 | Initial learning rate.                                                                  | `0.0001`            |
| `--lr_final`                | Final learning rate.                                                                    | `0.00001`           |
| `--k_value`                 | Exponent of the weight function for transforming confidences into weights.              | `3`                 |

### Example Command

Here’s an example of how to run the experiment with sample arguments:

```bash
python main.py --setup 2 --use_precomputed_confs False --cv_folds 5 --use_precomputed_lambdas False --lambda_path ./lambdas.txt --home_path ./FREDA-CV/ --alpha 0.8 --epochs 20 --global_iterations 100 --lr_init 0.0001 --lr_final 0.00001 --k_value 3
```

### Data

This project includes a utility to prepare federated domain adaptation datasets from any tabular `.csv` file or `pandas.DataFrame`.

You can use the `generate_federated_simulation_data` function (in [`generate_federated_data.py`](generate_federated_data.py)) to split your dataset across multiple source clients and a single target client. The data is saved in individual folders in plain `.txt` format.

> ⚠️ **Note**: Your dataset **must include a column specifying the domain** of each sample (e.g., region, hospital, tissues).

#### How to use

```python
from generate_federated_data import generate_federated_simulation_data
import pandas as pd

# Load your dataset
df = pd.read_csv("your_dataset.csv")

# domain column must be available
df["domain"] = ...  # e.g., region, hospital, tissues, etc.

# Generate federated simulation data
generate_federated_simulation_data(
    df=df,
    label_column="your_label_column",
    domain_column="domain",
    num_clients=3,
    output_dir="data",
    target_domains=["target_domain_1"] # one or more domain values to act as the target
)
```

This will create the following folder structure:
```
data/
├── 0/
│   ├── x_train.txt
│   └── y_train.txt
├── 1/
│   └── ...
├── 2/
├── target/
    ├── x_train.txt
    └── y_train.txt
```









