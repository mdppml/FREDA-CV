import argparse
from aggregator import Aggregator
import datetime
import time
import os
from clients import SourceClient, TargetClient
import pandas as pd
from generate_federated_data import generate_federated_simulation_data

parser = argparse.ArgumentParser()

parser.add_argument('--setup', type=int, default=4,
                    help="Client number.")

parser.add_argument('--use_precomputed_confs', type=bool, default=False,
                    help="Whether to use pre-computed confidences.")

parser.add_argument('--cv_folds', type=int, default=5,
                    help="Number of Folds to run cross validation for lambda prediction.")

parser.add_argument('--use_precomputed_lambdas', type=bool, default=False,
                    help="Whether to use pre-computed optimal lambdas.")

parser.add_argument('--lambda_path', type=str, default=None,
                    help="Path to a text file containing lambda values, one per line.")

parser.add_argument('--alpha', type=float, default=0.8,
                    help="Weighting factor for the loss function.")

parser.add_argument('--epochs', type=int, default=5,
                    help="Number of local training epochs.")

parser.add_argument('--global_iterations', type=int, default=5,
                    help="Number of global iterations.")

parser.add_argument('--lr_init', type=float, default=1e-4,
                    help="Initial learning rate.")

parser.add_argument('--lr_final', type=float, default=1e-5,
                    help="Final learning rate.")

parser.add_argument('--k_value', type=int, default=3,
                    help="Exponent of the weight function for transforming confidences into weights.")

parser.add_argument('--home_path', type=str, default=os.getcwd(),
                    help="Root directory for the project (default: current working directory).")

args = parser.parse_args()

no_clients = args.setup

use_precomputed_confs = args.use_precomputed_confs
use_precomputed_lambdas = args.use_precomputed_lambdas
lambda_path_file = args.lambda_path

# Hyperparameters
alpha = args.alpha
epochs = args.epochs
global_iterations = args.global_iterations
lr_init = args.lr_init
lr_final = args.lr_final
k_value = args.k_value

cv_folds = args.cv_folds

home_path = args.home_path


def load_lambda_values(file_path):
    """
    Load lambda values from a specified text file.
    Each line in the file should contain one numeric value.
    """
    try:
        with open(file_path, 'r') as f:
            lambdas = [float(line.strip()) for line in f]
        print(f"Loaded lambda values from {file_path}: {lambdas}")
        return lambdas
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        exit(1)
    except ValueError:
        print(f"Error: Invalid format in file {file_path}. Ensure it contains only numeric values, one per line.")
        exit(1)


def main():
    # Print time stamp
    print("Program started:", "{:%d.%m.%Y %H:%M:%S}".format(datetime.datetime.now()))

    random_seed = 36  # Random common seed across clients for masking and secure aggregation

    start_time = time.time()

    # Load your own DataFrame and specify label/domain columns
    df = pd.read_csv("your_dataset.csv")
    df['domain'] = ...  # define or load a domain column if available

    domain_indices, target_domains = generate_federated_simulation_data(
        df=df,
        label_column="target_column",
        domain_column="domain",
        num_clients=3,
        output_dir="data",
        target_domains=["example_target"]
    )

    source_clients = [SourceClient(home_path, no_clients, cid, random_seed, cv_folds) for cid in range(no_clients)]

    target_client = TargetClient(home_path, random_seed, k_value, domain_indices, target_domains)

    server = Aggregator(home_path, no_clients, source_clients, target_client, cv_folds)

    if use_precomputed_confs:
        print("Using pre-computed confidences...")
        server.use_precomputed_confidences()
    else:
        print("Performing federated hyper-parameter optimization...")
        server.compute_global_hyperparameters_secure()

        print("Computing confidence scores...")
        server.compute_federated_confidence_scores()

    if use_precomputed_lambdas:
        print("Using pre-computed lambda values...")
        server.use_precomputed_best_lambdas()
    else:
        if lambda_path_file:
            lambda_path = load_lambda_values(lambda_path_file)
        else:
            lambda_path = [0.1, 0.2, 0.3]

        print("Computing best lambdas...")
        server.compute_lambda_cv(lambda_path, alpha, epochs, global_iterations, lr_init, lr_final)

    print("Training adaptive models with optimal lambda prediction...")
    server.train_final_adaptive_models(alpha, epochs, global_iterations, lr_init, lr_final)

    end_time = time.time()
    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Program finished in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()
