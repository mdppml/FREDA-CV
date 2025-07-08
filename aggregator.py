import os
import numpy as np
from models import *
import pandas as pd


class Aggregator:

    def __init__(self, home_path, setup, source_clients, target_client, cv_folds):
        """
        The aggregator class which orchestrates the federated learning simulation
        :param home_path: Current working directory.
        :param setup: The number of source clients.
        :param source_clients: A list of source client objects.
        :param target_client: A target client object.
        :param cv_folds: Number of cv folds to use for finding the best lambda.
        """
        self.no_clients = setup

        self.dist_dir = os.path.join(home_path, 'data/')

        self.source_clients = source_clients
        self.target_client = target_client

        self.source_client_sample_sizes = [client.get_no_samples() for client in self.source_clients]

        self.total_source_samples = sum(self.source_client_sample_sizes)

        self.cv_folds = cv_folds

        self.kernel_sig = None
        self.noise_sig = None

        self.confidences = None
        self.target_groups = self.target_client.get_groups()

        self.best_lambdas = None

    def compute_global_hyperparameters_secure(self):
        if self.kernel_sig and self.noise_sig:
            print("Hyperparameters already computed.")
            return
        elif self.confidences:
            print("Confidences already computed, skipping hyperparameters.")
            return

        kernel_sig = []
        noise_sig = []

        no_features = self.target_client.get_no_features()

        for i in range(no_features):
            for client in self.source_clients:
                client.compute_masked_hyperparameters(feature=i)

            ks, ns = zip(*(client.get_masked_hyperparameters() for client in self.source_clients))
            kernel_sig.append(sum(ks) / self.no_clients)
            noise_sig.append(sum(ns) / self.no_clients)

            print(f"Securely aggregated hyperparameters for feature {i}")

        self.kernel_sig = kernel_sig
        self.noise_sig = noise_sig

        return kernel_sig, noise_sig

    def compute_federated_confidence_scores(self):
        """
        When called, performs the federated GPR training as well as the confidence score computation. The federated GPR
        training involves both the source clients and the target client. Randomized encoding and masking is used during
        the computation to protect the privacy of local data of the clients. The computed closed form solution of the
        GPR model is directly the predicted distribution. this distribution is then sent to the target client where
        the target client computes the confidence score.
        :return: None
        """
        if self.confidences is not None:
            print("Confidences already computed.")
            self.target_client.set_confidences(self.confidences)
            return

        if self.noise_sig is None or self.kernel_sig is None:
            raise ValueError(
                "Hyperparameters (noise_sig and kernel_sig) are not set. Cannot compute federated confidence scores.")

        preds_mean, preds_var = self.predict_with_federated_GPRs(self.kernel_sig, self.noise_sig)

        confidences = []

        no_features = self.target_client.get_no_features()

        for i in range(no_features):
            confs = self.target_client.compute_confidence_score(i, preds_mean[i], preds_var[i])
            confidences.append(confs)
            print(f"Confidence computed for feature: {i}")

        confidences = np.column_stack(confidences)
        np.savetxt(os.path.join(self.dist_dir, "all_confidences.csv"), confidences, delimiter=';')

        self.confidences = confidences
        self.target_client.set_confidences(self.confidences)

    def predict_with_federated_GPRs(self, global_kernel_sig, global_noise_sig):
        """
        Performs federated GPR training with the computed hyperparameters.
        :param global_kernel_sig: The global sigma of the linear kernel computed during federated hyperparameter
        optimization
        :param global_noise_sig: The global sigma of the noise of the GPR formula computed during federated
        hyperparameter optimization
        :return: The mean and variance of the predicted distribution
        """

        prediction_means = []
        prediction_vars = []

        no_features = self.target_client.get_no_features()

        for i in range(no_features):
            masked_source_data = [client.generate_masked_data(i) for client in self.source_clients]
            masked_target_data = self.target_client.generate_masked_data(i)

            rows = [np.concatenate([masked_data @ Y.T for Y in masked_source_data], axis=1) for masked_data in
                    masked_source_data]

            # Concatenate all rows vertically to form the Gram matrix
            gram_matrix = np.concatenate(rows, axis=0)

            K = global_kernel_sig[i] * gram_matrix

            K_star = global_kernel_sig[i] * np.concatenate([masked_target_data @ X.T for X in masked_source_data],
                                                           axis=1)

            K_star_star = global_kernel_sig[i] * (masked_target_data @ masked_target_data.T)

            # Computing the intermediate matrix products

            inverse_of_covariance_matrix_of_input = np.linalg.inv(
                K + global_noise_sig[i] * np.eye(self.total_source_samples))

            intermediate_mean_vector = np.dot(K_star, inverse_of_covariance_matrix_of_input)  # This is K* @ K^-1

            # now splitting the intermediate result between the source clients
            client_means = []
            start_idx = 0

            for idx in range(self.no_clients):
                # Calculate the end index for the current client based on the number of samples they have
                end_idx = start_idx + self.source_client_sample_sizes[idx]

                # Slice the mean matrix from start_idx to end_idx for this client
                client_means.append(intermediate_mean_vector[:, start_idx:end_idx])

                # Update the start index for the next client
                start_idx = end_idx

            # Each client computes the dot product between their slice and their label vector (current feature vector)

            final_mean_parts = [client.compute_mean_piece(mean_piece, i) for mean_piece, client in
                                zip(client_means, self.source_clients)]

            final_mean = sum(final_mean_parts).flatten()

            # computing the variance is straightforward
            cov = np.diag(K_star_star) - np.dot(K_star, np.dot(inverse_of_covariance_matrix_of_input, K_star.T))
            var = np.sqrt(np.diag(cov))

            # computed mean and var are then sent to the target for confidence score computation

            prediction_means.append(final_mean)
            prediction_vars.append(var)

        return prediction_means, prediction_vars

    def compute_lambda_cv(self, lambda_path, alpha, epochs, global_iterations, lr_init,
                          lr_final):
        """
        Computes the best lambda for each domain using cross validation. For a given lambda path, all source clients
        train  WEN models in a federated learning setting for each target domain. All these models are then evaluated
        using k-fold cross validation on the local source domains to find the best lambda.
        :param lambda_path: A list of lambda values to perform training with
        :param alpha: Weighting factor for the loss function.
        :param epochs: Number of local training epochs.
        :param global_iterations: Number of global iterations.
        :param lr_init: Initial learning rate.
        :param lr_final: Final learning rate.
        :return: None
        """

        all_results = dict()

        domain_weights = self.target_client.compute_weights()

        for lam in lambda_path:
            print(f"Training for Lambda = {lam}")

            for domain in domain_weights:
                domain_scores = []

                for fold in range(self.cv_folds):
                    # Train model for this fold
                    global_model = self.train_federated_WEN(
                        lam, domain_weights[domain], alpha, epochs,
                        global_iterations, lr_init, lr_final, fold_no=fold
                    )

                    # Evaluate each source on this fold
                    scores = [source.evaluate_cv_model(global_model, fold) for source in self.source_clients]

                    avg_score = np.mean(scores)
                    domain_scores.append(avg_score)

                # Save average score across folds for this domain and lambda
                mean_cv_score = np.mean(domain_scores)
                if domain not in all_results:
                    all_results[domain] = {}
                all_results[domain][lam] = mean_cv_score

        # Convert results into DataFrame
        df = pd.DataFrame(all_results).sort_index()
        df.index.name = "Lambda"
        df.to_csv(os.path.join(self.dist_dir, "all_results.csv"))
        print("Saved cross-validation results to all_results.csv")

        # Save best lambda per domain
        best_lambdas = {
            domain: max(scores, key=scores.get)
            for domain, scores in all_results.items()
        }

        # Convert to DataFrame
        best_lambda_df = pd.DataFrame([
            {"Domain": domain, "Best Lambda": lam}
            for domain, lam in best_lambdas.items()
        ])

        # Save to CSV
        best_lambda_df.to_csv(os.path.join(self.dist_dir, "best_lambdas.csv"), index=False)
        print("Saved best lambda values to best_lambdas.csv")

        self.best_lambdas = best_lambdas

    def use_precomputed_best_lambdas(self):
        """
        Reads precomputed lambda values from the working directory if they exist. The file should be named
        best_lambdas.csv and should be inside the current distribution directory
        :return: None
        """
        file_path = os.path.join(self.dist_dir, "best_lambdas.csv")

        if os.path.exists(file_path):  # Check if precomputed lambdas exist
            # Read the CSV file into a DataFrame
            best_lambdas_df = pd.read_csv(file_path)

            # Convert the DataFrame to a dictionary (if needed)
            self.best_lambdas = best_lambdas_df.set_index('Domain')['Best Lambda'].to_dict()
        else:
            # Raise an error if the file does not exist
            raise FileNotFoundError(f"Could not find 'best_lambdas.csv' in the directory: {self.dist_dir}")

    def use_precomputed_confidences(self):
        """
        Reads precomputed confidence scores from the working directory if they exist. The file should be named
        all_confidences.csv and should be inside the current distribution directory
        :return: None
        """
        file_path = os.path.join(self.dist_dir, "all_confidences.csv")

        if os.path.exists(file_path):  # Check if precomputed confidences exist
            # Load precomputed confidences
            confidences = np.loadtxt(file_path, delimiter=';')
            self.confidences = np.asfortranarray(confidences)
            self.target_client.set_confidences(self.confidences)
        else:
            # Raise an error if the file does not exist
            raise FileNotFoundError(f"Could not find 'all_confidences.csv' in the directory: {self.dist_dir}")

    def train_federated_WEN(self, lam, feature_weights, alpha=0.8, epochs=100, global_iterations=10,
                            lr_init=1e-4,
                            lr_final=1e-5, fold_no=None):
        """
        Performs federated Weighted Elastic Net (WEN) training with the source clients for a given set of parameters.
        :param lam: The regularization parameter to use
        :param feature_weights: The feature weights computed using the confidence scores
        :param alpha: Weighting factor for the loss function
        :param epochs: Number of local training epochs
        :param global_iterations: Number of global iterations
        :param lr_init: Initial learning rate
        :param lr_final: Final learning rate
        :param fold_no: Current fold of the cross validation
        :return: The global model after the federated training is completed
        """

        if self.confidences is None:
            raise ValueError("Cannot perform training without confidences.")

        no_features = self.target_client.get_no_features()

        global_model = create_model(no_features, alpha, lam, feature_weights, lr_init)

        for iteration in range(global_iterations):
            print(".", end="")

            current_lr = lr_schedule(iteration, global_iterations, lr_init, lr_final)

            global_model_weights = global_model.get_weights()

            local_updates = []

            for client in self.source_clients:
                local_update = client.train_WEN_locally(global_model_weights, feature_weights, alpha, lam, current_lr,
                                                        epochs, fold_no=fold_no)
                local_updates.append(local_update)

            # Average the weights
            average_weights = [np.mean(weight_list, axis=0) for weight_list in zip(*local_updates)]

            global_model.set_weights(average_weights)

        return global_model

    def train_final_adaptive_models(self, alpha, epochs, global_iterations, lr_init,
                                    lr_final):
        """
        Trains and prints the performance of the final adaptive models using the best lambda values for each domain
        :param alpha: Weighting factor for the loss function
        :param epochs: Number of local training epochs
        :param global_iterations: Number of global iterations
        :param lr_init: Initial learning rate
        :param lr_final: Final learning rate
        :return: None
        """

        domain_weights = self.target_client.compute_weights()

        # set up tables for hold-out errors

        for domain in domain_weights:
            global_model = self.train_federated_WEN(self.best_lambdas[domain], domain_weights[domain], alpha, epochs,
                                                    global_iterations,
                                                    lr_init,
                                                    lr_final)

            self.target_client.eval_model(domain, global_model)
            print()
