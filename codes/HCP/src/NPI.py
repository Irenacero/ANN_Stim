"""
NPI.py — Surrogate "neural process inference" models and utilities.

Notation for shapes used throughout:
- T: number of time points (time samples) in an input time series
- N: number of brain regions / nodes (features per time point)
- S: number of past steps concatenated to predict the next step (window length)
- B: batch size
- C: channels for CNN (here, conceptually equals N)
- L: sequence length for 1D convs/RNNs (here, equals S)

Key shapes:
- Raw time series:                    (T, N)
- Sliding-window inputs (multi2one):  X: (T - S, S * N)
- Sliding-window targets (multi2one): Y: (T - S, N)

All models map an input vector of length S*N to an output vector of length N,
except ANN_CNN and ANN_RNN which internally reshape for temporal processing.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

# Select GPU if available, otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Feedforward MLP surrogate
# -----------------------------------------------------------------------------
class ANN_MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) used as a surrogate brain model.

    Expected forward I/O:
      - Input x:  shape (B, S*N)  — S time steps concatenated across N regions
      - Output:   shape (B, N)    — next-time-point prediction for all regions
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, output_dim: int):
        """
        Args:
            input_dim:  S*N (windowed steps times number of regions)
            hidden_dim: size of first hidden layer
            latent_dim: size of second hidden layer (bottleneck)
            output_dim: N (number of regions)
        """
        super().__init__()
        self.func = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # (B, S*N) -> (B, hidden_dim)
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim), # (B, hidden_dim) -> (B, latent_dim)
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim), # (B, latent_dim) -> (B, N)
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S*N)
        returns: (B, N)
        """
        return self.func(x)


# -----------------------------------------------------------------------------
# Temporal CNN surrogate
# -----------------------------------------------------------------------------
class ANN_CNN(nn.Module):
    """
    1D CNN used as a surrogate brain model.

    Internally, the flat input (B, S*N) is reshaped into a temporal stack:
      - Reshape to (B, S, N) then permute to (B, C=N, L=S) for Conv1d
    The Conv1d operates across the temporal axis (length S).

    Expected forward I/O:
      - Input x:   (B, S*N)
      - Output:    (B, N)
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, data_length: int):
        """
        Args:
            in_channels:   N (number of regions; becomes channels for Conv1d)
            hidden_channels: channels in the intermediate conv layer
            out_channels:  channels after the final conv layer
            data_length:   S (number of time steps per input window)
        """
        super().__init__()
        self.in_channels = in_channels   # N
        self.data_length = data_length   # S

        # Conv1d expects shape (B, C, L) -> (B, hidden_channels, L-1) after kernel_size=2
        self.CNN = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=2),  # (B, N, S) -> (B, hidden_channels, S-1)
            nn.ReLU(),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=2), # (B, hidden_channels, S-1) -> (B, out_channels, S-2)
        ).to(device)

        # After two convs with kernel_size=2 (no padding/stride), length reduces by 2 -> (S - 2)
        # We flatten (B, out_channels, S-2) -> (B, out_channels*(S-2)), then map to N
        self.Linear = nn.Linear(out_channels * (data_length - 2), in_channels).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S*N)
        returns: (B, N)
        """
        # Reshape flat window to (B, S, N)
        x = x.view(-1, self.data_length, self.in_channels)  # (B, S, N)
        # Permute to channels-first for Conv1d: (B, N, S)
        x = x.permute(0, 2, 1)

        # Temporal convolution across the S axis
        pred = self.CNN(x)  # (B, out_channels, S-2)

        # torch.squeeze() can drop dims of size 1; be careful with B=1.
        # In the original code, squeeze is used; we keep it for parity.
        pred = torch.squeeze(pred)  # (B, out_channels, S-2) or (out_channels, S-2) if B==1

        # Ensure 2D for the Linear layer: if batch was squeezed out, unsqueeze it back.
        if pred.dim() == 2:
            # Good case: (B, out_channels, S-2) -> flatten next
            pass
        else:
            # If pred is (out_channels, S-2), insert batch dim
            pred = pred.unsqueeze(0)

        # Flatten temporal features and map to N
        pred = pred.reshape(pred.shape[0], -1)   # (B, out_channels*(S-2))
        pred = self.Linear(pred)                 # (B, N)
        return pred


# -----------------------------------------------------------------------------
# RNN surrogate
# -----------------------------------------------------------------------------
class ANN_RNN(nn.Module):
    """
    Simple RNN-based surrogate.

    Pipeline:
      - Input x (B, S*N) -> reshape to (B, S, N)
      - Encode each time step's N-dim vector with a small MLP to a latent_dim vector
      - Feed a dummy sequence (length 1) into an RNN whose initial hidden state is the
        time-major encoding of the last S encodings (S, B, latent_dim). (This matches
        your original code's use of encodes[:, S-1:, :] permuted to (1, B, latent_dim).)
      - Map the RNN output at that dummy step to (B, N)

    Expected forward I/O:
      - Input x:  (B, S*N)
      - Output:   (B, N)
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, output_dim: int, data_length: int):
        """
        Args:
            input_dim:   N  (features per time step)
            hidden_dim:  size of hidden layer in the encoder MLP
            latent_dim:  latent dimension (also RNN hidden size)
            output_dim:  N
            data_length: S
        """
        super().__init__()
        self.input_dim = input_dim   # N
        self.data_length = data_length  # S

        # Per-time-step encoder: (B, S, N) -> (B, S, latent_dim)
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # (B, S, N) -> (B, S, hidden_dim)
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim), # (B, S, hidden_dim) -> (B, S, latent_dim)
        ).to(device)

        # RNN that will consume a dummy input of shape (B, 1, 1), with initial hidden state
        # set to the last S encodings permuted to (num_layers=1, B, latent_dim).
        self.rnn = nn.RNN(input_size=1, hidden_size=latent_dim, batch_first=True).to(device)

        # Map the RNN hidden/state to N outputs
        self.output = nn.Linear(latent_dim, output_dim).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S*N)
        returns: (B, N)
        """
        # Reshape to time-major blocks per batch: (B, S, N)
        x = x.view(-1, self.data_length, self.input_dim)

        # Encode each time step independently: (B, S, latent_dim)
        encodes = self.enc(x)

        # Construct a dummy 1-step sequence input for the RNN: zeros of shape (B, 1, 1)
        dummy_in = torch.zeros((x.shape[0], 1, 1), device=device)  # (B, 1, 1)

        # Use the last S encodings as the initial hidden state.
        # Original code uses encodes[:, self.data_length-1:, :] -> (B, 1, latent_dim),
        # then permutes to (1, B, latent_dim) for h0. We preserve that behavior.
        h0 = torch.permute(encodes[:, self.data_length - 1 :, :], (1, 0, 2)).contiguous()  # (1, B, latent_dim)

        # Run the RNN for one step; ht: (B, 1, latent_dim)
        ht, _ = self.rnn(dummy_in, h0)

        # Map the single-step output to N: output(ht) -> (B, 1, N) then select step 0 -> (B, N)
        return self.output(ht)[:, 0, :]


# -----------------------------------------------------------------------------
# Linear VAR surrogate
# -----------------------------------------------------------------------------
class ANN_VAR(nn.Module):
    """
    Linear model equivalent to a single-layer VAR (no bias on lag structure explicitly).
    It simply learns a linear map from the flattened S*N input to N outputs.

    Expected forward I/O:
      - Input x:  (B, S*N)
      - Output:   (B, N)
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.func = nn.Linear(input_dim, output_dim).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S*N)
        returns: (B, N)
        """
        return self.func(x)


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------
def multi2one(time_series: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window input/output pairs from a multivariate time series.

    Args:
        time_series: array of shape (T, N)
        steps:       S, number of past time steps to use as input

    Returns:
        input_X:  shape (T - S, S*N)  — concatenation of S consecutive time points
        target_Y: shape (T - S, N)    — the next time point following each window
    """
    n_area = time_series.shape[1]  # N
    n_step = time_series.shape[0]  # T

    input_X = np.zeros((n_step - steps, n_area * steps), dtype=float)  # (T-S, S*N)
    target_Y = np.zeros((n_step - steps, n_area), dtype=float)         # (T-S, N)

    for i in range(n_step - steps):
        # Window: time_series[i : i+S] has shape (S, N) -> flatten to (S*N,)
        input_X[i] = time_series[i : steps + i].flatten()
        # Target: next time point (N,)
        target_Y[i] = time_series[steps + i].flatten()

    return np.array(input_X), np.array(target_Y)


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
def train_NN(
    model: nn.Module,
    input_X: np.ndarray,
    target_Y: np.ndarray,
    batch_size: int = 50,
    train_set_proportion: float = 0.8,
    num_epochs: int = 100,
    lr: float = 1e-3,
    l2: float = 0.0,
):
    """
    Train a surrogate model on (X, Y) pairs.

    Args:
        model:                 ANN_MLP / ANN_CNN / ANN_RNN / ANN_VAR
        input_X:               (T-S, S*N)
        target_Y:              (T-S, N)
        batch_size:            B
        train_set_proportion:  proportion of samples used for training
        num_epochs:            training epochs
        lr:                    learning rate
        l2:                    weight decay (L2 regularization)

    Returns:
        model:             trained model (on current device)
        train_epoch_loss:  list[float] of length num_epochs
        test_epoch_loss:   list[float] of length num_epochs
    """
    # Split train/test by proportion along the sample axis (first dimension)
    split_idx = int(train_set_proportion * input_X.shape[0])

    train_inputs = torch.tensor(input_X[:split_idx], dtype=torch.float, device=device)  # (n_train, S*N)
    train_targets = torch.tensor(target_Y[:split_idx], dtype=torch.float, device=device)  # (n_train, N)

    test_inputs = torch.tensor(input_X[split_idx:], dtype=torch.float, device=device)   # (n_test, S*N)
    test_targets = torch.tensor(target_Y[split_idx:], dtype=torch.float, device=device) # (n_test, N)

    # Datasets and loaders
    train_dataset = data.TensorDataset(train_inputs, train_targets)
    test_dataset = data.TensorDataset(test_inputs, test_targets)
    train_iter = data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_iter = data.DataLoader(test_dataset, batch_size, shuffle=False)

    # Optimizer/loss
    loss_fn = nn.MSELoss()
    trainer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    train_epoch_loss = []
    test_epoch_loss = []

    for _ in range(num_epochs):
        # ---- Train ----
        model.train()
        for Xb, yb in train_iter:
            y_hat = model(Xb)              # (B, N)
            l = loss_fn(y_hat, yb)         # scalar
            trainer.zero_grad()
            l.backward()
            trainer.step()

        # ---- Evaluate ----
        model.eval()
        with torch.no_grad():
            # Compute average train loss over all train batches
            total_loss = 0.0
            total_num = 0
            for Xb, yb in train_iter:
                y_hat = model(Xb)
                l = loss_fn(y_hat, yb)
                total_loss += l * yb.shape[0]
                total_num += yb.shape[0]
            train_epoch_loss.append(float(total_loss / total_num))

            # Compute average test loss over all test batches
            total_loss = 0.0
            total_num = 0
            for Xb, yb in test_iter:
                y_hat = model(Xb)
                l = loss_fn(y_hat, yb)
                total_loss += l * yb.shape[0]
                total_num += yb.shape[0]
            test_epoch_loss.append(float(total_loss / total_num))

    return model, train_epoch_loss, test_epoch_loss


# -----------------------------------------------------------------------------
# Model builder utility
# -----------------------------------------------------------------------------
def build_model(method: str, ROI_num: int, using_steps: int):
    """
    Build and return an ANN model based on the selected architecture.

    Args:
        method:       'MLP', 'CNN', 'RNN', or 'VAR'
        ROI_num:      number of brain regions (N)
        using_steps:  number of time steps in each input window (S)
    """
    method = method.upper()
    if method == "MLP":
        return ANN_MLP(
            input_dim=using_steps * ROI_num,
            hidden_dim=2 * ROI_num,
            latent_dim=int(0.8 * ROI_num),
            output_dim=ROI_num,
        )
    elif method == "CNN":
        return ANN_CNN(
            in_channels=ROI_num,
            hidden_channels=3 * ROI_num,
            out_channels=ROI_num,
            data_length=using_steps,
        )
    elif method == "RNN":
        return ANN_RNN(
            input_dim=ROI_num,
            hidden_dim=int(2.5 * ROI_num),
            latent_dim=int(2.5 * ROI_num),
            output_dim=ROI_num,
            data_length=using_steps,
        )
    elif method == "VAR":
        return ANN_VAR(
            input_dim=using_steps * ROI_num,
            output_dim=ROI_num,
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'MLP', 'CNN', 'RNN', or 'VAR'.")



# -----------------------------------------------------------------------------
# Connectivity helpers
# -----------------------------------------------------------------------------
def corrcoef(signals: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation-based functional connectivity (FC) matrix.

    Args:
        signals: (T, N) time-by-region matrix

    Returns:
        FC: (N, N) correlation matrix
    """
    # torch.corrcoef expects variables in rows: provide signals.T -> (N, T)
    return (
        torch.corrcoef(torch.tensor(signals.T, dtype=torch.float, device=device))
        .detach()
        .cpu()
        .numpy()
    )


def model_FC(model: nn.Module, node_num: int, steps: int) -> np.ndarray:
    """
    Simulate surrogate data by iteratively feeding back model outputs with small noise,
    then compute the FC of the simulated time series.

    Process:
      - Maintain a rolling buffer of the last S states, initialized to zeros.
      - At each iteration, create input = concat(last S states) + Gaussian noise
      - Forward through the model to get next state (N,)
      - Append to the simulation buffer

    Args:
        model:     surrogate model
        node_num:  N
        steps:     S (window length)

    Returns:
        FC: (N, N) correlation matrix of simulated series
    """
    NN_sim = []
    # Seed the initial S states with zeros: S entries of (N,)
    for _ in range(steps):
        NN_sim.append(np.zeros(node_num))

    # Generate ~1200 predicted steps
    for _ in range(1200):
        noise = 0.5 * np.random.randn(steps * node_num)  # (S*N,)
        model_input = np.array(NN_sim[-steps:]).flatten() + noise  # (S*N,)

        if isinstance(model, ANN_RNN):
            # RNN forward returns (1, N) when given a single sample -> select [0]
            out = model(torch.tensor(model_input, dtype=torch.float, device=device)).detach().cpu().numpy()[0]  # (N,)
        else:
            out = model(torch.tensor(model_input, dtype=torch.float, device=device)).detach().cpu().numpy()     # (N,)
        NN_sim.append(out)

    NN_sim = np.array(NN_sim)  # shape ~ (S + 1200, N)
    return corrcoef(NN_sim)


def model_EC(model: nn.Module, input_X: np.ndarray, target_Y: np.ndarray, pert_strength: float) -> np.ndarray:
    """
    Infer an effective connectivity (EC)-like influence matrix via perturbation.

    For each node j (column), add a small perturbation of size 'pert_strength' to the
    last time step in the S-step input window (i.e., only at the most recent time),
    feed both perturbed and unperturbed inputs through the model, and average the
    output differences across samples. Row j of the returned matrix is the effect of
    perturbing node j on all nodes.

    Args:
        model:        trained surrogate
        input_X:      (M, S*N) input windows for M samples
        target_Y:     (M, N)   (not used for loss here; used for N)
        pert_strength: scalar perturbation magnitude

    Returns:
        NPI_EC: (N, N) where row j is effect of perturbing node j on all nodes
    """
    node_num = target_Y.shape[1]           # N
    steps = int(input_X.shape[1] / node_num)  # S from S*N

    NPI_EC = np.zeros((node_num, node_num), dtype=float)  # (N, N)

    for node in range(node_num):
        # Unperturbed outputs across all M samples: (M, N)
        unperturbed_output = (
            model(torch.tensor(input_X, dtype=torch.float, device=device)).detach().cpu().numpy()
        )

        # Build a perturbation that only hits the last step and a single node: (S, N)
        perturbation = np.zeros((steps, node_num), dtype=float)
        perturbation[-1, node] = pert_strength
        perturb_flat = perturbation.flatten()  # (S*N,)

        # Perturbed outputs: (M, N)
        perturbed_output = (
            model(torch.tensor(input_X + perturb_flat, dtype=torch.float, device=device))
            .detach()
            .cpu()
            .numpy()
        )

        # Average difference across M samples -> row for this perturbed node
        NPI_EC[node] = np.mean(perturbed_output - unperturbed_output, axis=0)  # (N,)

    return NPI_EC



def model_Jacobian(model: nn.Module, input_X: np.ndarray, steps: int) -> np.ndarray:
    """
    Estimate the (average) Jacobian of the model's mapping with respect to its inputs,
    then extract the block that corresponds to the last time step -> all outputs,
    and transpose to form an EC-like matrix.

    For non-recurrent models:
      - J(x) has shape (N, S*N). We take the last N columns (corresponding to the
        most recent time step in the window) -> (N, N). Accumulate over samples and
        average.

    For ANN_RNN (due to its batch/dummy-step structure):
      - torch.autograd.functional.jacobian returns a tensor where we index
        [0, :, -N:] to get the slice corresponding to the single dummy time step's
        output and the input block for the last step.

    Args:
        model:    trained surrogate model
        input_X:  (M, S*N) input windows
        steps:    S

    Returns:
        jacobian_EC: (N, N) average of the per-sample Jacobian blocks (transposed)
                     i.e., rows are "source" nodes (inputs at last step),
                           columns are "target" nodes (outputs)
    """
    node_num = int(input_X.shape[1] / steps)  # N
    jacobian = np.zeros((node_num, node_num), dtype=float)  # accumulator in output-space layout

    model.train()  # Jacobian requires grad
    for i in range(input_X.shape[0]):
        x_i = torch.tensor(input_X[i], dtype=torch.float, device=device)  # (S*N,)

        if isinstance(model, ANN_RNN):
            # For RNN, the jacobian shape depends on autograd expansion;
            # the original code indexes [0, :, -N:] to select the correct slice.
            # Resulting slice: (N_out, N_in_last_step) == (N, N)
            J = torch.autograd.functional.jacobian(model, x_i).cpu().detach().numpy()
            jacobian += J[0, :, -node_num:]
        else:
            # For feedforward models: J has shape (N, S*N); take last N columns -> (N, N)
            J = torch.autograd.functional.jacobian(model, x_i).cpu().detach().numpy()  # (N, S*N)
            jacobian += J[:, -node_num:]

    model.eval()

    # Average over samples, then transpose to match EC convention in this codebase
    jacobian_EC = jacobian.T / input_X.shape[0]  # (N, N)
    return jacobian_EC





# ######################################
# ######################################
# ######################################
# ######################################
# # GIOVANNI's CODE
# # BRAIN STATE DEPENDENCE OF STIMULATION



def model_time_series(model: nn.Module, initial_state, tlen:int, noise_strength:float) -> np.ndarray:
    """
    Simulate surrogate data by starting from initial state and iteratively feeding back model outputs with small noise,
    then compute the simulated time series.

    Process:
      - Maintain a rolling buffer of the last S states, initialized to zeros.
      - At each iteration, create input = concat(last S states) + Gaussian noise
      - Forward through the model to get next state (N,)
      - Append to the simulation buffer

    Args:
        model:     surrogate model
        initial_state: array of size (S,N)~(window length, number of ROIs)
        tlen: duration of the simulated activity
        noise_strength: noise magnitude

    Returns:
        modeled time series
    """

    S=initial_state.shape[0]
    N=initial_state.shape[1]
    
    NN_sim = list(initial_state)

    # Generate tlen predicted steps
    for _ in range(tlen):
        noise = noise_strength * np.random.randn(S * N)  # (S*N,)
        model_input = np.array(NN_sim[-S:]).flatten() + noise  # (S*N,)

        if isinstance(model, ANN_RNN):
            # RNN forward returns (1, N) when given a single sample -> select [0]
            out = model(torch.tensor(model_input, dtype=torch.float, device=device)).detach().cpu().numpy()[0]  # (N,)
        else:
            out = model(torch.tensor(model_input, dtype=torch.float, device=device)).detach().cpu().numpy()     # (N,)
        NN_sim.append(out)

    return np.array(NN_sim)  # shape ~ (S + tlen, N)


def model_Jacobian_timewise(model: nn.Module, input_X: np.ndarray, steps: int) -> np.ndarray:
    """
    Compute the Jacobian for each input window separately.

    Args:
        model: trained surrogate model
        input_X: (M, S*N) input windows (each row is one time step)
        steps: S (window length)

    Returns:
        jacobians: (M, N, N) Jacobian slice per time step
                   where jacobians[m] = d y / d x_laststep  at window m
    """
    node_num = int(input_X.shape[1] / steps)  # N
    M = input_X.shape[0]
    jacobians = np.zeros((M, node_num, node_num), dtype=float)

    model.train()  # gradients needed
    for i in range(M):
        x_i = torch.tensor(input_X[i], dtype=torch.float, device=device)

        if isinstance(model, ANN_RNN):
            J = torch.autograd.functional.jacobian(model, x_i).cpu().detach().numpy()
            jacobians[i] = J[0, :, -node_num:]  # output→last-step input
        else:
            J = torch.autograd.functional.jacobian(model, x_i).cpu().detach().numpy()  # (N, S*N)
            jacobians[i] = J[:, -node_num:]

    model.eval()
    # Optionally transpose each Jacobian if you follow EC convention
    return jacobians.transpose(0, 2, 1)  # shape (M, N, N)



def model_ECt(model: nn.Module, input_X: np.ndarray, target_Y: np.ndarray, pert_strength: float) -> np.ndarray:
    """
    Compute effective connectivity (EC)-like influence matrix via single-node perturbation at the last timestep.

    For each node j in the network, compute how stimulating that node at the final timestep affects
    all nodes' outputs, repeated across M different initial conditions.

    For each node j:
      1. Get unperturbed model predictions for all M initial conditions
      2. Create a perturbation (stimulation) affecting only node j at the final timestep
      3. Get model predictions with stimulation applied to all M initial conditions
      4. Record the difference in predictions (perturbed - unperturbed) for each initial condition

    Args:
        model:          trained surrogate neural network
        input_X:        (M, S*N) array of M flattened input windows, where each row represents
                        a different initial condition. S is the number of past timesteps and N is the
                        number of nodes. Each input window contains S consecutive timesteps
                        (e.g., [t, t-1, t-2, ...]) flattened from (S, N) to (S*N).
                        The "final timestep" refers to the most recent state (t) in this window.
        target_Y:       (M, N) array of target outputs (used only to extract N, the number of nodes)
        pert_strength:  scalar magnitude of perturbation/stimulation to apply

    Returns:
        NPI_ECt: (M, N, N) array where NPI_ECt[i, j, :] is the output difference for initial
                 condition i when node j is stimulated at the last timestep.
                 The second axis j indexes which node was stimulated; the last axis indexes the
                 resulting effect on all N nodes.
    """


    node_num = target_Y.shape[1]           # N
    steps = int(input_X.shape[1] / node_num)  # S from S*N
    M=input_X.shape[0]

    NPI_ECt = np.zeros((M, node_num, node_num), dtype=float)  # (N, N)

    for node in range(node_num):
        # Unperturbed outputs across all M samples: (M, N)
        unperturbed_output = (
            model(torch.tensor(input_X, dtype=torch.float, device=device)).detach().cpu().numpy()
        )

        # Build a perturbation that only hits the last step and a single node: (S, N)
        perturbation = np.zeros((steps, node_num), dtype=float)
        perturbation[-1, node] = pert_strength
        perturb_flat = perturbation.flatten()  # (S*N,)

        # Perturbed outputs: (M, N)
        perturbed_output = (
            model(torch.tensor(input_X + perturb_flat, dtype=torch.float, device=device))
            .detach()
            .cpu()
            .numpy()
        )

        # Average difference across M samples -> row for this perturbed node
        NPI_ECt[:, node, :] = perturbed_output - unperturbed_output  # (M, N, N)

    return NPI_ECt


def state_distance(x, x_s, metric='l2'):
    """
    Compute distance or angle between two state vectors x and x_s.
    Works for high-dimensional vectors.
    """
    x = np.ravel(x)
    x_s = np.ravel(x_s)

    if metric == 'l2':
        return np.linalg.norm(x_s - x)

    nx, nxs = np.linalg.norm(x), np.linalg.norm(x_s)
    if nx == 0 or nxs == 0:
        return np.nan

    if metric == 'cosine_dist':
        cos_sim = np.dot(x, x_s) / (nx * nxs)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        return 1.0 - cos_sim

    elif metric == 'angle':
        cos_theta = np.dot(x, x_s) / (nx * nxs)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)          # radians, [0, π]
        # map to [-π, π] (unsigned → signed mapping not meaningful in >3D, but safe)
        if theta > np.pi:
            theta -= 2 * np.pi
        return theta

    else:
        raise ValueError("metric must be one of ['l2', 'cosine_dist', 'angle']")





def model_BECt(model: nn.Module, input_X: np.ndarray, target_Y: np.ndarray, pert_strength: float, metric='l2') -> np.ndarray:
    """
    Infer bifocal effective connectivity (BEC) via dual-region perturbation at each time point.

    For each pair of nodes (i, j), add a perturbation to both regions simultaneously at the
    last time step, feed through the model, and measure the output distance compared to
    unperturbed. Results are symmetric: BEC[i,j] = BEC[j,i].

    Args:
        model:          trained surrogate (ANN_MLP / ANN_CNN / ANN_RNN / ANN_VAR)
        input_X:        (M, S*N) input windows for M samples
        target_Y:       (M, N)   used only for shapes; N = number of regions
        pert_strength:  scalar amplitude; each region gets pert_strength/√2 for equalized total
        metric:         'l2', 'cosine_dist', or 'angle' to measure perturbed vs unperturbed distance

    Returns:
        NPI_BECt: (M, N, N) symmetric matrix where BEC[t, i, j] = distance when perturbing nodes i+j
                  (only upper triangle + diagonal computed; lower triangle mirrored for efficiency)
    """
    node_num = target_Y.shape[1]           # N
    steps = int(input_X.shape[1] / node_num)  # S from S*N
    M = input_X.shape[0]

    NPI_BECt = np.zeros((M, node_num, node_num), dtype=float)

    # Compute unperturbed outputs ONCE (outside loops)
    with torch.no_grad():
        unperturbed_output = (
            model(torch.tensor(input_X, dtype=torch.float, device=device))
            .detach()
            .cpu()
            .numpy()
        )  # (M, N)

    # Iterate over all pairs (i, j) with i <= j for efficiency
    for node_i in range(node_num):
        for node_j in range(node_i, node_num):
            # Build perturbation
            perturbation = np.zeros((steps, node_num), dtype=float)
            
            if node_i == node_j:
                # Single node: full strength
                perturbation[-1, node_i] = pert_strength
            else:
                # Bifocal: split equally to maintain total magnitude
                # Total L2 magnitude: √(2 × (pert_strength/√2)²) = pert_strength
                perturbation[-1, node_i] = pert_strength / np.sqrt(2)
                perturbation[-1, node_j] = pert_strength / np.sqrt(2)
            
            perturb_flat = perturbation.flatten()  # (S*N,)

            # Perturbed outputs
            with torch.no_grad():
                perturbed_output = (
                    model(torch.tensor(input_X + perturb_flat, dtype=torch.float, device=device))
                    .detach()
                    .cpu()
                    .numpy()
                )  # (M, N)

            # Compute state distance for each time point
            distances = np.array([
                state_distance(perturbed_output[m, :], unperturbed_output[m, :], metric=metric)
                for m in range(M)
            ])  # (M,)

            # Store symmetrically
            NPI_BECt[:, node_i, node_j] = distances
            if node_i != node_j:
                NPI_BECt[:, node_j, node_i] = distances

    return NPI_BECt



def collect_state_effect_pairs(
    model: nn.Module,
    input_X: np.ndarray,
    target_Y: np.ndarray,
    pert_strength: float = 0.1,
    use_model_next: bool = True,
):
    """
    Collect (x_t baseline, x_{t+Δt} next, x_{t+Δt}^{stim}) triplets for *every* time window
    and *every* stimulated region.

    Args:
        model:          Trained surrogate (ANN_MLP / ANN_CNN / ANN_RNN / ANN_VAR).
        input_X:        (W, S*N) sliding-window inputs. Each row ends at time t.
        target_Y:       (W, N) targets (empirical next step at t+Δt). Used for shapes (and
                        can optionally serve as x_{t+Δt} if use_model_next=False).
        pert_strength:  Scalar amplitude of the stimulation applied to the *last* step
                        (time t) and a single region j.
        use_model_next: If True, x_{t+Δt} (unperturbed) comes from the model forward pass.
                        If False, it uses target_Y (empirical next step).

    Returns:
        result: dict with
            - 'xt_baseline':   (W, N) baseline states at time t (last step of each window)
            - 'xt_next':       (W, N) unperturbed next states at t+Δt
                               (model outputs if use_model_next=True, else target_Y)
            - 'xt_stim_next':  (N, W, N) for each stimulated region j:
                                  xt_stim_next[j, w, :] = model( window_w with stim at node j )
            - 'meta':          dict of helper info (N, S, W, pert_strength)

    Notes:
        Shapes:
            - W: number of windows (sliding windows extracted from data)
            - S: number of past steps per window
            - N: number of regions (nodes)
        Stimulation is injected at the *last* step of each window (time t).
    """
    model.eval()

    W = input_X.shape[0]
    N = target_Y.shape[1]
    S = input_X.shape[1] // N

    # ---- x_t baseline: last-step slice from each window ----
    xt_baseline = input_X.reshape(W, S, N)[:, -1, :].copy()  # (W, N)

    # ---- x_{t+Δt} unperturbed ----
    if use_model_next:
        with torch.no_grad():
            xt_next = (
                model(torch.tensor(input_X, dtype=torch.float, device=device))
                .detach()
                .cpu()
                .numpy()
            )  # (W, N)
    else:
        xt_next = np.asarray(target_Y, dtype=float)  # (W, N)

    # ---- x_{t+Δt}^{stim} for each stimulated region j ----
    xt_stim_next = np.zeros((N, W, N), dtype=float)

    for j in range(N):
        perturb = np.zeros((S, N), dtype=float)
        perturb[-1, j] = pert_strength
        perturb_flat = perturb.flatten()  # (S*N,)

        with torch.no_grad():
            stim_out = (
                model(torch.tensor(input_X + perturb_flat, dtype=torch.float, device=device))
                .detach()
                .cpu()
                .numpy()
            )  # (W, N)

        xt_stim_next[j] = stim_out

    result = {
        "xt_baseline": xt_baseline,     # (W, N)
        "xt_next": xt_next,             # (W, N)
        "xt_stim_next": xt_stim_next,   # (N, W, N)
        "meta": {"W": W, "S": S, "N": N, "pert_strength": pert_strength},
    }
    return result

########################################
# # Simulated timeseries 

# def simulate_ann_timeseries(
#     model: torch.nn.Module,
#     steps_out: int,
#     S: int,
#     seed_series: np.ndarray | None = None,
#     noise_std: float = 0.0,
#     device: torch.device | None = None,
# ) -> np.ndarray:
#     """
#     Generate a synthetic BOLD-like time series by 'closing the loop' on a fitted ANN.

#     Args:
#         model:       trained ANN that maps (B, S*N) -> (B, N)
#         steps_out:   number of future time points to generate (T_out)
#         S:           window length used during training (past steps concatenated)
#         seed_series: optional (T_seed, N) array to initialize the buffer.
#                      If provided and T_seed >= S, the last S rows are used.
#                      If provided and 0 < T_seed < S, it will be padded with zeros.
#                      If None, initialization is zeros for S steps.
#         noise_std:   std of i.i.d. Gaussian noise added to each input vector (S*N)
#                      at each step (helps avoid fixed-point collapse). Default 0.0.
#         device:      torch device; if None, uses the device of the model parameters.

#     Returns:
#         sim: (steps_out, N) numpy array with the generated time series.
#     """
#     model.eval()
#     # Infer device from model if not given
#     if device is None:
#         try:
#             device = next(model.parameters()).device
#         except StopIteration:
#             device = torch.device("cpu")

#     # Try to infer N (output dimension)
#     try:
#         N = model.func[-1].out_features  # ANN_MLP as defined in your file
#     except Exception:
#         # Fallback: do a tiny dry run with zeros to infer N
#         # We need input dim; try to infer from first Linear layer
#         try:
#             in_dim = model.func[0].in_features
#         except Exception as e:
#             raise ValueError(
#                 "Could not infer model input/output sizes. "
#                 "Please pass a seed_series with known N."
#             ) from e
#         N = None  # will be inferred after first forward

#     # Build initial rolling buffer of S states (each of shape (N,))
#     if seed_series is not None:
#         seed_series = np.asarray(seed_series, dtype=float)
#         if seed_series.ndim != 2:
#             raise ValueError("seed_series must be 2D (T_seed, N).")
#         T_seed, N_seed = seed_series.shape
#         if T_seed >= S:
#             buf = [seed_series[T_seed - S + i].copy() for i in range(S)]  # last S rows
#         else:
#             # pad with zeros to reach S
#             pad = [np.zeros(N_seed, dtype=float) for _ in range(S - T_seed)]
#             buf = pad + [seed_series[i].copy() for i in range(T_seed)]
#         N_from_seed = buf[0].shape[0]
#         N = N if N is not None else N_from_seed
#         if N is not None and N != N_from_seed:
#             raise ValueError(f"Mismatch between inferred N={N} and seed N={N_from_seed}.")
#     else:
#         if N is None:
#             raise ValueError(
#                 "Could not infer N. Provide a seed_series or use an ANN_MLP with model.func[-1].out_features."
#             )
#         buf = [np.zeros(N, dtype=float) for _ in range(S)]

#     out_list = []
#     with torch.no_grad():
#         for _ in range(steps_out):
#             x_in = np.array(buf[-S:], dtype=float).reshape(-1)  # (S*N,)
#             if noise_std > 0.0:
#                 x_in = x_in + noise_std * np.random.randn(x_in.size)

#             x_in_t = torch.tensor(x_in, dtype=torch.float32, device=device).unsqueeze(0)  # (1, S*N)
#             x_next_t = model(x_in_t)  # (1, N)
#             if N is None:
#                 N = x_next_t.shape[-1]
#             x_next = x_next_t.squeeze(0).detach().cpu().numpy()  # (N,)

#             out_list.append(x_next)
#             buf.append(x_next)
#             # keep last S states in the buffer
#             if len(buf) > S:
#                 buf = buf[-S:]

#     sim = np.stack(out_list, axis=0)  # (steps_out, N)
#     return sim


# ##########################################
# # Simulated Timeseries from snapshot

# def _rollout_from_snapshot(
#     model: torch.nn.Module,
#     init_buffer: np.ndarray,     # (S, N) — pre/post snapshot buffer ending at x_t
#     first_next: np.ndarray,      # (N,)   — x_{t+Δt} (unperturbed or stimulated)
#     horizon: int,                # H      — how many future steps to generate
#     noise_std: float = 0.0,
#     device: torch.device | None = None,
# ) -> np.ndarray:
#     """
#     Roll out the ANN from a known pre-pulse buffer and the first next state.

#     Returns:
#         traj: (H, N) predicted sequence starting at index 0 == first_next
#     """
#     model.eval()
#     if device is None:
#         try:
#             device = next(model.parameters()).device
#         except StopIteration:
#             device = torch.device("cpu")

#     init_buffer = np.asarray(init_buffer, dtype=float)  # (S, N)
#     first_next  = np.asarray(first_next, dtype=float)   # (N,)
#     S, N = init_buffer.shape

#     # Rolling buffer holds the last S states; after we have first_next, we proceed
#     buf = [init_buffer[i].copy() for i in range(S)]
#     traj = [first_next.copy()]  # time step 1 = x_{t+Δt}

#     with torch.no_grad():
#         # Now generate H-1 additional steps by closing the loop
#         while len(traj) < horizon:
#             # Compose model input = concat(last S states)
#             x_in = np.array(buf[-S:], dtype=float).reshape(-1)  # (S*N,)
#             if noise_std > 0:
#                 x_in = x_in + noise_std * np.random.randn(x_in.size)
#             x_in_t = torch.tensor(x_in, dtype=torch.float32, device=device).unsqueeze(0)  # (1, S*N)

#             # Predict next state
#             x_next_t = model(x_in_t)  # (1, N)
#             x_next = x_next_t.squeeze(0).detach().cpu().numpy()  # (N,)

#             traj.append(x_next)
#             buf.append(x_next)
#             if len(buf) > S:
#                 buf = buf[-S:]

#     return np.stack(traj, axis=0)  # (H, N)


# def simulate_stim_responses(
#     model: torch.nn.Module,
#     xt_baseline: np.ndarray,   # (W, N)   — x_t
#     xt_next: np.ndarray,       # (W, N)   — unperturbed x_{t+Δt}
#     xt_stim_next: np.ndarray,  # (N, W, N) — stimulated x^{(j)}_{t+Δt}
#     S: int,                    # window length used to train the ANN
#     H: int,                    # rollout horizon (number of post-pulse steps)
#     prev_stack: np.ndarray | None = None,  # optional (W, S-1, N) pre-pulse frames
#     noise_std: float = 0.0,
#     device: torch.device | None = None,
# ):
#     """
#     Produce full post-pulse time series for unperturbed and stimulated branches.

#     Returns:
#         unperturbed: (W, H, N)
#         stimulated:  (N, W, H, N) — per target j
#         delta:       (N, W, H, N) = stimulated - unperturbed (aligned in time)
#     """
#     xt_baseline = np.asarray(xt_baseline, dtype=float)   # (W, N)
#     xt_next     = np.asarray(xt_next, dtype=float)       # (W, N)
#     xt_stim_next = np.asarray(xt_stim_next, dtype=float) # (N, W, N)

#     W, N = xt_baseline.shape
#     J = xt_stim_next.shape[0]
#     assert xt_next.shape == (W, N), "xt_next must be (W, N)"
#     assert xt_stim_next.shape == (J, W, N), "xt_stim_next must be (N_targets, W, N)"

#     if prev_stack is not None:
#         prev_stack = np.asarray(prev_stack, dtype=float)
#         assert prev_stack.shape == (W, S-1, N), f"prev_stack must be (W, S-1, N); got {prev_stack.shape}"

#     # Allocate outputs
#     unperturbed = np.empty((W, H, N), dtype=float)
#     stimulated  = np.empty((J, W, H, N), dtype=float)

#     # Loop over windows
#     for w in range(W):
#         # Build the S-length buffer that ends at x_t
#         if prev_stack is not None:
#             # concat the S-1 previous frames with x_t
#             init_buffer = np.vstack([prev_stack[w], xt_baseline[w][None, :]])  # (S, N)
#         else:
#             # zero-pad the S-1 previous frames
#             init_buffer = np.vstack([np.zeros((S-1, N), dtype=float), xt_baseline[w][None, :]])  # (S, N)

#         # Unperturbed rollout: first_next = xt_next[w]
#         unperturbed[w] = _rollout_from_snapshot(
#             model=model,
#             init_buffer=init_buffer,
#             first_next=xt_next[w],
#             horizon=H,
#             noise_std=noise_std,
#             device=device,
#         )  # (H, N)

#         # Stimulated rollouts for each target j
#         for j in range(J):
#             stimulated[j, w] = _rollout_from_snapshot(
#                 model=model,
#                 init_buffer=init_buffer,
#                 first_next=xt_stim_next[j, w],
#                 horizon=H,
#                 noise_std=noise_std,
#                 device=device,
#             )  # (H, N)

#     # Delta response = stimulated - unperturbed (broadcast over W,H,N)
#     delta = stimulated - unperturbed[None, :, :, :]  # (J, W, H, N)

#     return {
#         "unperturbed": unperturbed,  # (W, H, N)
#         "stimulated": stimulated,    # (J, W, H, N)
#         "delta": delta,              # (J, W, H, N)
#     }



# #########################################
# # AB Stimulation 

# def collect_pairwise_state_effects(
#     model: nn.Module,
#     input_X: np.ndarray,      # (W, S*N)
#     target_Y: np.ndarray,     # (W, N)  (used for shapes or as xt_next if use_model_next=False)
#     pert_strength: float = 0.1,
#     use_model_next: bool = True,
# ):
#     """
#     Collect (x_t baseline, x_{t+Δt} next, x_{t+Δt}^{stim, A+B}) triplets for every time window
#     and for every unordered pair of target regions (A, B), A < B.

#     Stimulus is injected at the *last* step of each window (time t) on *both* A and B.

#     Args:
#         model:          Trained surrogate (ANN_MLP / ANN_CNN / ANN_RNN / ANN_VAR).
#         input_X:        (W, S*N) sliding-window inputs. Each row ends at time t.
#         target_Y:       (W, N) targets (empirical next step at t+Δt). Used for shapes (and
#                         can optionally serve as x_{t+Δt} if use_model_next=False).
#         pert_strength:  Scalar amplitude of the stimulation applied to the last step at nodes A and B.
#         use_model_next: If True, x_{t+Δt} (unperturbed) comes from the model forward pass.
#                         If False, it uses target_Y (empirical next step).

#     Returns:
#         result: dict with
#             - 'xt_baseline':   (W, N) baseline states at time t (last step of each window)
#             - 'xt_next':       (W, N) unperturbed next states at t+Δt
#                                (model outputs if use_model_next=True, else target_Y)
#             - 'xt_stim_next':  (E, W, N) stimulated next states for each edge (A,B), A<B
#             - 'meta':          dict with:
#                                 * 'W', 'S', 'N', 'E', 'pert_strength'
#                                 * 'edges' : list of tuples [(A,B), ...] length E;
#                                             xt_stim_next[k] corresponds to edges[k]
#     """
#     model.eval()

#     # Dimensions
#     W = input_X.shape[0]
#     N = target_Y.shape[1]
#     S = input_X.shape[1] // N

#     # Baseline at time t (last step in each window)
#     xt_baseline = input_X.reshape(W, S, N)[:, -1, :].copy()  # (W, N)

#     # Unperturbed next
#     if use_model_next:
#         with torch.no_grad():
#             xt_next = (
#                 model(torch.tensor(input_X, dtype=torch.float, device=device))
#                 .detach()
#                 .cpu()
#                 .numpy()
#             )  # (W, N)
#     else:
#         xt_next = np.asarray(target_Y, dtype=float)  # (W, N)

#     # Build list of unordered edges (A,B) with A < B
#     edges = [(a, b) for a in range(N) for b in range(a + 1, N)]
#     E = len(edges)

#     # Prepare output container
#     xt_stim_next = np.zeros((E, W, N), dtype=float)

#     # Pre-build the (S, N) perturbation template (we'll set two entries per edge)
#     base_pert = np.zeros((S, N), dtype=float)
#     # only last step receives the stimulation
#     # we'll copy and fill positions [-1, A] and [-1, B] for each edge

#     # Iterate over edges and simulate
#     for k, (A, B) in enumerate(edges):
#         pert = base_pert.copy()
#         pert[-1, A] = pert_strength
#         pert[-1, B] = pert_strength
#         pert_flat = pert.flatten()  # (S*N,)

#         with torch.no_grad():
#             stim_out = (
#                 model(torch.tensor(input_X + pert_flat, dtype=torch.float, device=device))
#                 .detach()
#                 .cpu()
#                 .numpy()
#             )  # (W, N)

#         xt_stim_next[k] = stim_out

#     result = {
#         "xt_baseline": xt_baseline,     # (W, N)
#         "xt_next": xt_next,             # (W, N)
#         "xt_stim_next": xt_stim_next,   # (E, W, N)
#         "meta": {
#             "W": W, "S": S, "N": N, "E": E,
#             "pert_strength": pert_strength,
#             "edges": edges,             # list of (A,B) with A<B
#         },
#     }
#     return result

# # ---------- helpers -----------------------------------------------------------
# _EPS = 1e-12

# def _rowwise_norm(X):
#     """Normalize rows to unit L2 norm; safe for zero rows."""
#     norms = np.linalg.norm(X, axis=1, keepdims=True)
#     norms = np.where(norms < _EPS, 1.0, norms)
#     return X / norms

# def cosine_similarity_matrix(X):
#     """
#     X: (W, N) rows are effect vectors at different times for a fixed target
#     returns: (W, W) cosine similarity matrix
#     """
#     Xn = _rowwise_norm(X)
#     return Xn @ Xn.T

# def pearson_similarity_matrix(X):
#     """
#     Pearson correlation across features N, between *rows* (windows).
#     X: (W, N)
#     returns: (W, W) correlation matrix
#     """
#     # subtract row means?
#     # For Pearson across features, we z-score each row vector
#     Xm = X - X.mean(axis=1, keepdims=True)
#     std = X.std(axis=1, keepdims=True) + _EPS
#     Xz = Xm / std
#     return (Xz @ Xz.T) / X.shape[1]

# def make_effect_similarity_single(xt_next, xt_stim_next, metric="cosine"):
#     """
#     Single-focus:
#       xt_next:      (W, N)
#       xt_stim_next: (N, W, N)  stim on A -> next state
#     returns: dict A -> (W, W) similarity across windows
#     """
#     N = xt_stim_next.shape[0]
#     sim_single = {}

#     for A in range(N):
#         # effects across time for this target A: (W, N)
#         E_A = xt_stim_next[A] - xt_next

#         if metric == "cosine":
#             S = cosine_similarity_matrix(E_A)
#         elif metric == "pearson":
#             S = pearson_similarity_matrix(E_A)
#         else:
#             raise ValueError("metric must be 'cosine' or 'pearson'")

#         sim_single[A] = S

#     return sim_single

# def make_effect_similarity_pairs(xt_next, xt_stimAB_next, edges, metric="cosine"):
#     """
#     Bifocal pairs:
#       xt_next:        (W, N)
#       xt_stimAB_next: (E, W, N)
#       edges:          list of tuples [(A,B), ...] with A < B, length E
#     returns: nested dict A -> { B -> (W, W) similarity }
#     """
#     sim_pairs = {}
#     E = len(edges)

#     for k in range(E):
#         A, B = edges[k]
#         # effects (W, N) for pair (A,B)
#         E_AB = xt_stimAB_next[k] - xt_next

#         if metric == "cosine":
#             S = cosine_similarity_matrix(E_AB)
#         elif metric == "pearson":
#             S = pearson_similarity_matrix(E_AB)
#         else:
#             raise ValueError("metric must be 'cosine' or 'pearson'")

#         sim_pairs.setdefault(A, {})[B] = S

#     return sim_pairs

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import pearsonr

# # --- helper: variability from similarity matrix ------------------------------
# def variability_from_similarity(S: np.ndarray) -> float:
#     """
#     Compute variability = 1 - mean(upper triangular part of similarity matrix).
#     S: (W, W) symmetric similarity matrix
#     Returns: scalar
#     """
#     iu = np.triu_indices_from(S, k=1)
#     mean_sim = np.mean(S[iu])
#     return 1.0 - mean_sim

# # --- build variability DataFrame ---------------------------------------------
# def build_variability_df(sim_single: dict, sim_pairs: dict) -> pd.DataFrame:
#     """
#     sim_single: dict A -> (W, W)
#     sim_pairs:  dict A -> dict B -> (W, W), with A<B

#     Returns: pd.DataFrame with columns ['A','B','varA','varB','varAB']
#     """
#     rows = []
#     N = len(sim_single)

#     # Precompute single variabilities
#     var_single = {A: variability_from_similarity(sim_single[A]) for A in range(N)}

#     for A, dict_B in sim_pairs.items():
#         for B, S_AB in dict_B.items():
#             varA = var_single[A]
#             varB = var_single[B]
#             varAB = variability_from_similarity(S_AB)
#             rows.append(dict(A=A, B=B, varA=varA, varB=varB, varAB=varAB))

#     return pd.DataFrame(rows)

# # --- Example: distribution of varAB for fixed A ------------------------------
# def plot_varAB_distribution(df_var: pd.DataFrame, A: int):
#     """
#     Given a variability DataFrame and fixed A, plot distribution of varAB across B.
#     """
#     sub = df_var[df_var['A'] == A]

#     plt.figure(figsize=(6,4))
#     sns.histplot(sub['varAB'], bins=20, color="steelblue", alpha=0.7)
#     plt.axvline(sub['varA'].iloc[0], color="red", linestyle="--", lw=2, label=f"varA={sub['varA'].iloc[0]:.3f}")
#     plt.xlabel("Variability of AB effects")
#     plt.ylabel("Count of B")
#     plt.title(f"Distribution of AB variability for primary A={A}")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# # --- Score and correlation with in-strength ----------------------------------
# def correlate_score_instrength(df_var: pd.DataFrame, in_strength: np.ndarray, A: int):
#     """
#     For fixed A, compute score = varAB - varA across all B.
#     Correlate this array with in_strength[B].
#     """
#     sub = df_var[df_var['A'] == A].copy()
#     sub['score'] = sub['varAB'] - sub['varA']

#     r, p = pearsonr(sub['score'], in_strength[sub['B']])
#     print(f"A={A} | Pearson r={r:.3f}, p={p:.3g}")

#     # Scatter
#     plt.figure(figsize=(5,4))
#     sns.regplot(
#         x=in_strength[sub['B']], y=sub['score'],
#         scatter_kws=dict(s=40, alpha=0.7),
#         line_kws=dict(color="red")
#     )
#     plt.xlabel("In-strength of B")
#     plt.ylabel("Score (varAB - varA)")
#     plt.title(f"A={A}: score vs in-strength of B\nr={r:.2f}, p={p:.3g}")
#     plt.tight_layout()
#     plt.show()

#     return sub


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import pearsonr

# # ------------------------ helpers ------------------------
# def variability_from_similarity(S: np.ndarray) -> float:
#     """Variability = 1 - mean(upper-triangular similarity)."""
#     iu = np.triu_indices_from(S, k=1)
#     return 1.0 - float(np.mean(S[iu]))

# def build_variability_df(sim_single: dict, sim_pairs: dict) -> pd.DataFrame:
#     """
#     sim_single: A -> (W,W)
#     sim_pairs:  A -> {B -> (W,W)} with A < B
#     Returns rows ['A','B','varA','varB','varAB','delta'] with delta = varAB - varA
#     """
#     N = len(sim_single)
#     varA = {A: variability_from_similarity(sim_single[A]) for A in range(N)}
#     rows = []
#     for A, Bs in sim_pairs.items():
#         for B, S_AB in Bs.items():
#             vA = varA[A]
#             vB = varA[B]  # variability of single-focus B
#             vAB = variability_from_similarity(S_AB)
#             rows.append(dict(A=A, B=B, varA=vA, varB=vB, varAB=vAB, delta=vAB - vA))
#     return pd.DataFrame(rows)

# def perA_fraction_reduced(df_var: pd.DataFrame, in_strength: np.ndarray) -> pd.DataFrame:
#     """
#     Symmetrize pairs so every node appears as primary A.
#     For each A, compute the fraction of 'other' nodes where varAB < varA.
#     Returns perA DataFrame with columns ['A','n_pairs','frac_reduced','in_strength'].
#     """
#     rows = []
#     for _, row in df_var.iterrows():
#         A, B, varA, varB, varAB = int(row['A']), int(row['B']), row['varA'], row['varB'], row['varAB']
#         rows.append(dict(A=A, other=B, var_self=varA, varAB=varAB, reduced=(varAB < varA)))
#         rows.append(dict(A=B, other=A, var_self=varB, varAB=varAB, reduced=(varAB < varB)))
#     df_sym = pd.DataFrame(rows)
#     perA = (
#         df_sym.groupby('A')
#         .agg(n_pairs=('reduced','size'),
#              frac_reduced=('reduced','mean'))
#         .reset_index()
#     )
#     perA['in_strength'] = in_strength[perA['A'].values]
#     return perA


# import numpy as np

# def mean_cosine_distance(perturbed_output, unperturbed_output, eps=1e-12):
#     """
#     Compute mean cosine distance across time (axis=0 over M steps).
#     perturbed_output, unperturbed_output : arrays (M, N)
#     """
#     u = np.asarray(perturbed_output, float)
#     v = np.asarray(unperturbed_output, float)

#     dot = np.sum(u * v, axis=1)
#     uu = np.linalg.norm(u, axis=1)
#     vv = np.linalg.norm(v, axis=1)
#     cos_sim = dot / (uu * vv + eps)
#     cos_sim = np.clip(cos_sim, -1.0, 1.0)

#     cos_dist = 1.0 - cos_sim
#     return np.nanmean(cos_dist)

# def mean_signed_L2_change(perturbed_output, unperturbed_output, eps=1e-12):
#     """
#     Compute mean signed L2 change between two trajectories.
#     Sign is positive if the perturbation moves along the unperturbed direction,
#     negative if it moves against it.

#     Args:
#         perturbed_output, unperturbed_output : arrays (M, N)
#         eps : small constant to avoid division by zero

#     Returns:
#         scalar : mean signed L2 change across all time steps
#     """
#     u = np.asarray(unperturbed_output, float)
#     v = np.asarray(perturbed_output, float)

#     diff = v - u                         # (M, N)
#     l2_mag = np.linalg.norm(diff, axis=1)    # magnitude of change per time step
#     l2_mag = np.where(l2_mag < eps, 0, l2_mag)

#     # sign from projection of the change vector onto the baseline direction
#     proj = np.sum(diff * u, axis=1)
#     sign = np.sign(proj)                 # +1 if along, -1 if opposite

#     signed_l2 = sign * l2_mag
#     return np.nanmean(signed_l2)


# def model_EABC_L2(model: nn.Module, input_X: np.ndarray, target_Y: np.ndarray, pert_strength: float) -> np.ndarray:
#     """
#     Infer an effective connectivity (EC)-like influence matrix via perturbation.

#     For each node j (column), add a small perturbation of size 'pert_strength' to the
#     last time step in the S-step input window (i.e., only at the most recent time),
#     feed both perturbed and unperturbed inputs through the model, and average the
#     output differences across samples. Row j of the returned matrix is the effect of
#     perturbing node j on all nodes.

#     Args:
#         model:        trained surrogate
#         input_X:      (M, S*N) input windows for M samples
#         target_Y:     (M, N)   (not used for loss here; used for N)
#         pert_strength: scalar perturbation magnitude

#     Returns:
#         NPI_EABC: (N, N) where element i,j is the effect of perturbing node i and j in terms of cosine distance
#     """
#     node_num = target_Y.shape[1]           # N
#     steps = int(input_X.shape[1] / node_num)  # S from S*N

#     NPI_EABC_L2 = np.zeros((node_num, node_num), dtype=float)  # (N, N)

#     for node_i in range(node_num):
#         for node_j in range(node_num):
#             if node_j>node_i:
#                 # Unperturbed outputs across all M samples: (M, N)
#                 unperturbed_output = (
#                     model(torch.tensor(input_X, dtype=torch.float, device=device)).detach().cpu().numpy()
#                 )
        
#                 # Build a perturbation that only hits the last step and a single node: (S, N)
#                 perturbation = np.zeros((steps, node_num), dtype=float)
#                 perturbation[-1, node_i] = pert_strength
#                 perturbation[-1, node_j] = pert_strength
#                 perturb_flat = perturbation.flatten()  # (S*N,)
        
#                 # Perturbed outputs: (M, N)
#                 perturbed_output = (
#                     model(torch.tensor(input_X + perturb_flat, dtype=torch.float, device=device))
#                     .detach()
#                     .cpu()
#                     .numpy()
#                 )
        
#                 # Average difference across M samples -> row for this perturbed node
#                 NPI_EABC_L2[node_i,node_j] = mean_signed_L2_change(perturbed_output, unperturbed_output)

#     return NPI_EABC_L2+NPI_EABC_L2.T
    

# def model_EABC(model: nn.Module, input_X: np.ndarray, target_Y: np.ndarray, pert_strength: float) -> np.ndarray:
#     """
#     Infer an effective connectivity (EC)-like influence matrix via perturbation.

#     For each node j (column), add a small perturbation of size 'pert_strength' to the
#     last time step in the S-step input window (i.e., only at the most recent time),
#     feed both perturbed and unperturbed inputs through the model, and average the
#     output differences across samples. Row j of the returned matrix is the effect of
#     perturbing node j on all nodes.

#     Args:
#         model:        trained surrogate
#         input_X:      (M, S*N) input windows for M samples
#         target_Y:     (M, N)   (not used for loss here; used for N)
#         pert_strength: scalar perturbation magnitude

#     Returns:
#         NPI_EABC: (N, N) where element i,j is the effect of perturbing node i and j in terms of cosine distance or l2
#     """
#     node_num = target_Y.shape[1]           # N
#     steps = int(input_X.shape[1] / node_num)  # S from S*N

#     NPI_EABC_L2 = np.zeros((node_num, node_num), dtype=float)  # (N, N)
#     NPI_EABC_Cosine = np.zeros((node_num, node_num), dtype=float)  # (N, N)

#     for node_i in range(node_num):
#         for node_j in range(node_num):
#             if node_j>node_i:
#                 # Unperturbed outputs across all M samples: (M, N)
#                 unperturbed_output = (
#                     model(torch.tensor(input_X, dtype=torch.float, device=device)).detach().cpu().numpy()
#                 )
        
#                 # Build a perturbation that only hits the last step and a single node: (S, N)
#                 perturbation = np.zeros((steps, node_num), dtype=float)
#                 perturbation[-1, node_i] = pert_strength
#                 perturbation[-1, node_j] = pert_strength
#                 perturb_flat = perturbation.flatten()  # (S*N,)
        
#                 # Perturbed outputs: (M, N)
#                 perturbed_output = (
#                     model(torch.tensor(input_X + perturb_flat, dtype=torch.float, device=device))
#                     .detach()
#                     .cpu()
#                     .numpy()
#                 )
        
#                 # Average difference across M samples -> row for this perturbed node
#                 NPI_EABC_Cosine[node_i,node_j] = mean_cosine_distance(perturbed_output, unperturbed_output)
#                 NPI_EABC_L2[node_i,node_j] =  mean_signed_L2_change(perturbed_output, unperturbed_output)

#     return [NPI_EABC_Cosine+NPI_EABC_Cosine.T,NPI_EABC_L2+NPI_EABC_L2.T]
    

# def model_EC_reduced(model: nn.Module, input_X: np.ndarray, target_Y: np.ndarray, pert_strength: float) -> np.ndarray:
#     """
#     Infer an effective connectivity (EC)-like influence matrix via perturbation.

#     For each node j (column), add a small perturbation of size 'pert_strength' to the
#     last time step in the S-step input window (i.e., only at the most recent time),
#     feed both perturbed and unperturbed inputs through the model, and average the
#     output differences across samples. Row j of the returned matrix is the effect of
#     perturbing node j on all nodes.

#     Args:
#         model:        trained surrogate
#         input_X:      (M, S*N) input windows for M samples
#         target_Y:     (M, N)   (not used for loss here; used for N)
#         pert_strength: scalar perturbation magnitude

#     Returns:
#         NPI_EC: (N, N) where row j is effect of perturbing node j on all nodes
#     """
#     M=target_Y.shape[0]
#     node_num = target_Y.shape[1]           # N
#     steps = int(input_X.shape[1] / node_num)  # S from S*N

#     NPI_EC_min = np.zeros((node_num, node_num), dtype=float)  # (N, N)
#     NPI_EC_max = np.zeros((node_num, node_num), dtype=float)  # (N, N)
#     NPI_EC_rand = np.zeros((node_num, node_num), dtype=float)  # (N, N)

#     # --- compute energy of the most recent state (first N columns) ---
#     energy_input = np.sum(input_X[:, :node_num]**2, axis=1)   # (M,)
    
#     # --- number of samples in the top/bottom 25% ---
#     n_sel = int(0.05 * M)
    
#     # --- sort indices by energy ---
#     sorted_idx = np.argsort(energy_input)
    
#     # lowest and highest 25%
#     M_ids_min = sorted_idx[:n_sel]
#     M_ids_max = sorted_idx[-n_sel:]
#     # --- random selection (same count, drawn without replacement) ---
#     rng = np.random.default_rng(seed=42)
#     M_ids_rand = rng.choice(M, size=n_sel, replace=False)

#     input_X_min=input_X[M_ids_min,:]
#     input_X_max=input_X[M_ids_max,:]
#     input_X_rand=input_X[M_ids_rand,:]

#     for node in range(node_num):

#         # Build a perturbation that only hits the last step and a single node: (S, N)
#         perturbation = np.zeros((steps, node_num), dtype=float)
#         perturbation[-1, node] = pert_strength
#         perturb_flat = perturbation.flatten()  # (S*N,)
        
#         # Unperturbed outputs across all M/4 samples: (M/4, N)
#         unperturbed_output_min = (
#             model(torch.tensor(input_X_min, dtype=torch.float, device=device)).detach().cpu().numpy()
#         )

#         # Perturbed outputs: (M/4, N)
#         perturbed_output_min = (
#             model(torch.tensor(input_X_min + perturb_flat, dtype=torch.float, device=device))
#             .detach()
#             .cpu()
#             .numpy()
#         )

#                 # Unperturbed outputs across all M/4 samples: (M/4, N)
#         unperturbed_output_max = (
#             model(torch.tensor(input_X_max, dtype=torch.float, device=device)).detach().cpu().numpy()
#         )

#         # Perturbed outputs: (M/4, N)
#         perturbed_output_max = (
#             model(torch.tensor(input_X_max + perturb_flat, dtype=torch.float, device=device))
#             .detach()
#             .cpu()
#             .numpy()
#         )

#                 # Unperturbed outputs across all M/4 samples: (M/4, N)
#         unperturbed_output_rand = (
#             model(torch.tensor(input_X_rand, dtype=torch.float, device=device)).detach().cpu().numpy()
#         )

#         # Perturbed outputs: (M/4, N)
#         perturbed_output_rand = (
#             model(torch.tensor(input_X_rand + perturb_flat, dtype=torch.float, device=device))
#             .detach()
#             .cpu()
#             .numpy()
#         )

#         # Average difference across M samples -> row for this perturbed node
#         NPI_EC_min[node] = np.mean(perturbed_output_min - unperturbed_output_min, axis=0)  # (N,)
#         NPI_EC_max[node] = np.mean(perturbed_output_max - unperturbed_output_max, axis=0)  # (N,)
#         NPI_EC_rand[node] = np.mean(perturbed_output_rand - unperturbed_output_rand, axis=0)  # (N,)

#     return [NPI_EC_min,NPI_EC_max,NPI_EC_rand]
    
