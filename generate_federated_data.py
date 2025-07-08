import os
import numpy as np
from sklearn.preprocessing import StandardScaler


def generate_federated_simulation_data(
        df,
        label_column,
        domain_column=None,
        num_clients=3,
        output_dir="data",
        scale=True,
        target_domains=None
):
    """
    Generate federated simulation data from a general-purpose dataset.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - label_column (str): Name of the target variable column.
    - domain_column (str, optional): Name of the domain column. If None, random split is used.
    - num_clients (int): Number of source clients to simulate.
    - output_dir (str): Base directory where client data folders will be stored.
    - scale (bool): Whether to standardize features.
    - domain_split_strategy (str): "equal" or "custom". Currently only "equal" is implemented.
    - target_domains (list, optional): List of domains to group together as the target client.

    Returns:
    - target_grouping (pd.Series or None): The domain assignment of each row if domain_column is provided.
    - target_groups (np.ndarray or None): Unique domain values if domain_column is provided.
    """
    os.makedirs(output_dir, exist_ok=True)

    drop_columns = [label_column]
    if domain_column:
        drop_columns.append(domain_column)

    features = df.drop(columns=drop_columns)
    labels = df[label_column].values.reshape(-1, 1)

    if scale:
        scaler_x = StandardScaler()
        features = scaler_x.fit_transform(features)
        scaler_y = StandardScaler()
        labels = scaler_y.fit_transform(labels)

    assert domain_column in df.columns, "Provided domain_column does not exist in DataFrame."
    target_grouping = df[domain_column]
    target_groups = np.unique(target_grouping)

    # Separate source and target data
    target_mask = df[domain_column].isin(target_domains) if target_domains else np.zeros(len(df), dtype=bool)
    source_mask = ~target_mask

    x_target = features[target_mask]
    y_target = labels[target_mask]

    x_source = features[source_mask]
    y_source = labels[source_mask]

    print("Source data shape:", x_source.shape)
    print("Target data shape:", x_target.shape)

    # Shuffle and partition source data randomly across clients
    rng = np.random.RandomState(seed=42)
    indices = rng.permutation(len(x_source))
    chunks = np.array_split(indices, num_clients)

    for i, idx in enumerate(chunks):
        x_client = x_source[idx]
        y_client = y_source[idx]

        print(f"Client {i}: {len(x_client)} samples")

        client_dir = os.path.join(output_dir, f"client_{i}")
        os.makedirs(client_dir, exist_ok=True)
        np.savetxt(os.path.join(client_dir, "x_train.txt"), x_client, fmt='%.6f')
        np.savetxt(os.path.join(client_dir, "y_train.txt"), y_client, fmt='%.6f')

    # Save target client data
    if target_domains:
        print(f"Target client: {len(x_target)} samples from domains {target_domains}")
        target_dir = os.path.join(output_dir, "target")
        os.makedirs(target_dir, exist_ok=True)
        np.savetxt(os.path.join(target_dir, "x_train.txt"), x_target, fmt='%.6f')
        np.savetxt(os.path.join(target_dir, "y_train.txt"), y_target, fmt='%.6f')

    return target_grouping, target_groups
