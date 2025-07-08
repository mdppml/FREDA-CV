import os
import numpy as np
from GPR import GPR
from scipy.linalg import sqrtm
from scipy.stats import norm

from models import *
import util

import keras.backend as K
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


class SourceClient:

    def __init__(self, home_path, total_clients, client_id, seed, cv_folds):
        """
        Source client class for FREDA.
        :param home_path: Working directory
        :param total_clients: Total number of source clients in the system.
        :param client_id: id of the client
        :param seed: Random seed for common random mask generation
        """
        setup_dir = os.path.join(home_path, 'data/')
        self.data_dir = os.path.join(setup_dir, f'{client_id}/')

        self.total_clients = total_clients
        self.id = client_id

        self.X = np.loadtxt(os.path.join(self.data_dir, "x_train.txt"))
        self.Y = np.loadtxt(os.path.join(self.data_dir, "y_train.txt"))

        self.noises = list()
        self.kernel_vars = list()

        self.local_noise = None
        self.local_kernel = None

        self.local_WEN_model = None
        self.local_masked_WEN_model = None

        self.seed = seed
        np.random.seed(seed)

        self.no_features = self.X.shape[1]
        self.k = self.no_features + 1  # Can be set as high as desired depending on the privacy requirements

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        self.train_folds_X = []
        self.train_folds_Y = []
        self.val_folds_X = []
        self.val_folds_Y = []

        for train_idx, val_idx in kf.split(self.X):
            self.train_folds_X.append(self.X[train_idx])
            self.train_folds_Y.append(self.Y[train_idx])
            self.val_folds_X.append(self.X[val_idx])
            self.val_folds_Y.append(self.Y[val_idx])

    def get_no_samples(self):
        return self.X.shape[0]

    def compute_masked_hyperparameters(self, feature):
        """
        Computes masked kernel and noise sigmas for a given feature using zero-sum masking.
        :param feature: the index of the feature
        :return: masked kernel sigma, masked noise sigma, masks to share (for debugging/logging)
        """
        kernel_sig, noise_sig = self.compute_hyperparameters(feature)

        total_mask_kernel = 0.0
        total_mask_noise = 0.0
        shared_masks_kernel = {}
        shared_masks_noise = {}

        for peer_id in range(self.total_clients):
            if peer_id == self.id:
                continue

            rng = np.random.default_rng(self.seed + feature)  # deterministic for reproducibility
            r_kernel = rng.normal()
            r_noise = rng.normal()

            # Enforce consistent masking direction
            if self.id < peer_id:
                # We subtract this mask (sending)
                total_mask_kernel += r_kernel
                total_mask_noise += r_noise
                shared_masks_kernel[peer_id] = r_kernel
                shared_masks_noise[peer_id] = r_noise
            else:
                # We add this mask (receiving)
                total_mask_kernel -= r_kernel
                total_mask_noise -= r_noise
                shared_masks_kernel[peer_id] = -r_kernel
                shared_masks_noise[peer_id] = -r_noise

        # Final masked hyperparameters
        self.local_kernel = kernel_sig - total_mask_kernel
        self.local_noise = noise_sig - total_mask_noise

        return shared_masks_kernel, shared_masks_noise

    def get_masked_hyperparameters(self):
        return self.local_kernel, self.local_noise

    def compute_hyperparameters(self, feature):
        """
        Computes the optimal sigma for the linear kernel as well as the optimal sigma for the noise for a given feature
        based on own local data.
        :param feature: The feature for which to compute the kernel's sigma and the noise's sigma
        :return: the optimal kernel and noise sigmas based on own local data
        """
        # Paths to precomputed hyperparameter files
        kernel_sig_path = os.path.join(self.data_dir, "x_kernel_sig.txt")
        noise_sig_path = os.path.join(self.data_dir, "x_noise_sig.txt")

        # Check if precomputed hyperparameters exist
        if os.path.exists(kernel_sig_path) and os.path.exists(noise_sig_path):
            # Load precomputed hyperparameters
            kernel_sig = np.loadtxt(kernel_sig_path)
            noise_sig = np.loadtxt(noise_sig_path)
            return kernel_sig[feature], noise_sig[feature]

        # Compute hyperparameters from scratch if not precomputed
        is_target_feature = np.in1d(np.arange(self.X.shape[1]), feature)
        data_train_x = self.X[:, ~is_target_feature]  # Input features
        data_train_y = self.X[:, is_target_feature]  # Target feature

        gpr = GPR(data_train_x, data_train_y)
        kernel_sig, noise_sig = gpr.optimize()

        return kernel_sig, noise_sig

    def generate_masked_data(self, feature):
        """
        Generates the masked version of the data matrix after taking out the given feature
        :param feature: which feature to extract before masking
        :return: the masked data
        """
        is_feature = np.in1d(np.arange(self.no_features), feature)

        N = np.random.rand(self.k, self.no_features - 1)  # random common matrix for masking generated by all clients
        L = np.linalg.pinv(N)  # left inverse of the random matrix N

        data_train_x = self.X[:, ~is_feature]

        masked_data = data_train_x @ L @ np.real(sqrtm(N @ N.T))

        return masked_data

    def compute_mean_piece(self, mean_piece, feature):
        """
        Computes the dot product between the piece of the mean matrix and own local feature slice.
        :param mean_piece: A sub-matrix of the mean matrix with a dimension of
        N x M where N is the number of local samples
        :param feature: The index of the feature slice
        :return: the resulting vector after taking the dot product
        """
        is_feature = np.in1d(np.arange(self.no_features), feature)

        data_train_y = self.X[:, is_feature]

        return np.dot(mean_piece, data_train_y).flatten()

    def compute_masked_weights(self, weights):
        """
        Applies zero-sum masking to WEN model weights.
        :param weights: List of numpy arrays (model weights)
        :return: masked weights (with this client's mask applied)
        """

        masked_weights = []

        for i, weight in enumerate(weights):
            total_mask = np.zeros_like(weight)

            for peer_id in range(self.total_clients):
                if peer_id == self.id:
                    continue

                # Use deterministic seed per layer for reproducibility
                rng = np.random.default_rng(self.seed + i)

                mask = rng.normal(size=weight.shape)

                if self.id < peer_id:
                    total_mask += mask
                else:
                    total_mask -= mask

            masked_weight = weight - total_mask
            masked_weights.append(masked_weight)

        return masked_weights

    def train_WEN_locally(self, global_model_weights, feature_weights, alpha, lam, current_lr, epochs,
                          fold_no=None):
        """
        Performs local training of the Weighted Elastic Net model for the federated learning system. Takes as input
        the hyperparameters for the training and the current global model weights. Updates the weights of the current
        global model for a given number of epochs on own local data and gives back the updated model weights.

        :param global_model_weights: the initial global model weights for the current iteration
        :param feature_weights: weight vector of the weighted elastic net computed from the confidence scores
        :param alpha: penalties of the WEN
        :param lam: the regularization parameter to use during training
        :param current_lr: current global learning rate to use
        :param epochs: how many epochs to run on the local data
        :param fold_no: If fold_no is provided, trains on the training part of that fold. Otherwise, uses all data.
        :return: the updated model weights after training
        """

        # Determine training data
        if fold_no is None:
            X_train = self.X
            Y_train = self.Y
        else:
            X_train = self.train_folds_X[fold_no]
            Y_train = self.train_folds_Y[fold_no]

        if not self.local_WEN_model:
            self.local_WEN_model = create_model(self.no_features, alpha, lam, feature_weights, current_lr)

        self.local_WEN_model.set_weights(global_model_weights)

        K.set_value(self.local_WEN_model.optimizer.lr, current_lr)

        self.local_WEN_model.fit(X_train,
                                 Y_train,
                                 epochs=epochs,
                                 batch_size=32,
                                 verbose=0,
                                 shuffle=False
                                 )

        updated_weights = self.local_WEN_model.get_weights()

        masked_weights = self.compute_masked_weights(updated_weights)

        return masked_weights


class TargetClient:

    def __init__(self, home_path, seed, k_value, domain_indices, target_domains):
        """
        Target client class for FREDA.
        :param home_path: Working directory
        :param seed: Random seed for common random mask generation
        :param k_value: Exponent of the weight function for transforming confidences into weights.
        """
        self.home_path = home_path
        data_dir = os.path.join(home_path, f'data/target/')

        self.X = np.loadtxt(os.path.join(data_dir, "x_train.txt"))
        self.Y = np.loadtxt(os.path.join(data_dir, "y_train.txt"))

        self.k_value = k_value

        self.weighted_elastic_net = None
        self.confidences = None

        self.seed = seed
        np.random.seed(seed)

        self.no_features = self.X.shape[1]
        self.k = self.no_features + 1  # Can be set as high as desired depending on the privacy requirements

        self.domain_indices = domain_indices
        self.domains = target_domains

    def set_confidences(self, confidences):
        self.confidences = confidences

    def get_groups(self):
        return self.domains

    def get_no_features(self):
        return self.no_features

    def weight_func(self, x):
        return np.power(1 - x, self.k_value)

    def compute_weights(self):
        """
        computes the weights for the WEN training. Can only be called if confidences are already computed
        :return: the weight vectors for each domain in the target data computed from the confidences scores
        """
        domain_weights = dict()

        for domain in self.domains:
            feature_weights = self.confidences[np.array(domain == self.domain_indices)]
            feature_weights = self.weight_func(np.mean(feature_weights, axis=0))

            current_sum = np.sum(feature_weights)

            # Calculate the scaling factor
            scaling_factor = self.no_features / current_sum
            # Scale each element in the list
            weights = [element * scaling_factor for element in feature_weights]
            domain_weights[domain] = weights

        return domain_weights

    def generate_masked_data(self, feature):
        """
        Generates the masked version of the data matrix after taking out the given feature
        :param feature: which feature to extract before masking
        :return: the masked data
        """

        is_feature = np.in1d(np.arange(self.no_features), feature)

        N = np.random.rand(self.k, self.no_features - 1)  # random common matrix for masking generated by all clients
        L = np.linalg.pinv(N)  # left inverse of the random matrix N

        data_train_x = self.X[:, ~is_feature]

        masked_data = data_train_x @ L @ np.real(sqrtm(N @ N.T))

        return masked_data

    def compute_confidence_score(self, feature, mean, var):
        """
        Computes the confidence score for the current feature
        :param feature: The feature to compute the confidence score on
        :param mean: the mean of the predicted GPR distribution
        :param var: The variance of the predicted GPR distribution
        :return: The confidence vector
        """
        is_feature = np.in1d(np.arange(self.no_features), feature)

        target_label = self.X[:, is_feature].ravel()

        res_normed = (target_label - mean) / var
        confs = (1 - abs(norm.cdf(res_normed) - norm.cdf(-res_normed)))

        return confs

    def eval_model(self, domain, model):
        """
        Evaluates the given model on the target domain specified and prints the performance of the model
        :param domain: The domain data to use
        :param model: The model to evaluate
        :return: the performance of the model on the given domain's samples in a list
        """

        predictions = model.predict(self.X[self.domain_indices == domain, :]).ravel()
        errors_t = util.printTestErrors(predictions, self.Y[self.domain_indices == domain],
                                        "Performance on {}:".format(domain),
                                        indent=4)

        return errors_t
