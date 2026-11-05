# =============================================================================
# WHOLE-TRAJECTORY SUFFIX-EMBEDDING MEMBERSHIP BENCHMARK
#
# PUBLICATION-ORIENTED POSITIVE-CONTROL BENCHMARK
# =============================================================================
#
# PURPOSE
# -------
# Evaluate whether a whole-trajectory original-vs-generated classifier can
# distinguish:
#
#     1. strongly memorized member texts
#     2. ordinary member texts
#     3. non-member texts
#
#
# CORE IDEA
# ---------
#
# The language model is trained ONLY on member texts:
#
#     ordinary members:
#         each text appears once
#
#     strongly memorized members:
#         each text is deliberately repeated STRONG_REPEAT times
#
# The classifier is trained on completely separate texts.
#
# It receives:
#
#     original trajectory -> class 0
#     generated trajectory -> class 1
#
# It NEVER receives membership labels.
#
# For every text and prefix:
#
#     prefix + suffix
#
# We compute a suffix representation using contextual hidden states.
#
# K prefixes produce:
#
#     trajectory = K x D
#
# The CNN processes the complete trajectory jointly.
#
#
# FINAL MEMBERSHIP SCORE
# ----------------------
#
# For an evaluation text:
#
#     p_orig(original trajectory)
#     p_orig(generated trajectory)
#
# Score:
#
#     S = p_orig(original) - p_orig(generated)
#
# Larger S means:
#
#     original looks more like an original trajectory
#     AND
#     generated looks more like generated data
#
# Membership is evaluated ONLY after classifier training.
#
#
# IMPORTANT
# ---------
#
# This is a deliberately artificial POSITIVE CONTROL.
#
# STRONG_REPEAT does NOT represent realistic LM training.
#
# It is designed to establish whether the proposed method can detect
# increasing memorization under a controlled experimental manipulation.
#
# =============================================================================


import os
import csv
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 1234

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 110)
print("WHOLE-TRAJECTORY SUFFIX-EMBEDDING MEMBERSHIP BENCHMARK")
print("=" * 110)
print(f"Device = {DEVICE}")


# =============================================================================
# DATA
# =============================================================================

# LM training members
N_ORDINARY_MEMBERS = 80
N_STRONG_MEMBERS = 10

# Independent classifier corpus
N_CLASSIFIER_TRAIN = 30
N_CLASSIFIER_VAL = 10

# Final membership evaluation
N_EVAL_ORDINARY = 40
N_EVAL_STRONG = 10
N_NONMEMBERS = 80


# =============================================================================
# TEXT
# =============================================================================

SEQ_LEN = 48


# =============================================================================
# LANGUAGE MODEL
# =============================================================================

EMBED_DIM = 24
HIDDEN_DIM = 48

LM_EPOCHS = 15
LM_LR = 2e-3


# =============================================================================
# WHOLE TRAJECTORY
# =============================================================================

PREFIX_LENGTHS = [
    8,
    16,
    24,
    32,
]

K = len(PREFIX_LENGTHS)


# =============================================================================
# GENERATION
# =============================================================================

GEN_LEN = 8
TEMPERATURE = 1.15


# =============================================================================
# MEMORIZATION POSITIVE CONTROL
# =============================================================================

STRONG_REPEAT = 15


# =============================================================================
# TRAJECTORY CLASSIFIER
# =============================================================================

CNN_CHANNELS = 24
MLP_HIDDEN = 48

CLS_MAX_EPOCHS = 80
CLS_LR = 2e-3

WEIGHT_DECAY = 1e-4
DROPOUT = 0.10

BATCH_SIZE = 16

# Early stopping
PATIENCE = 10


# =============================================================================
# REPEATS
# =============================================================================

N_REPEATS = 5


# =============================================================================
# BOOTSTRAP
# =============================================================================

N_BOOTSTRAP = 2000


# =============================================================================
# OUTPUT
# =============================================================================

RESULTS_CSV = "whole_trajectory_membership_results.csv"


# =============================================================================
# SEED
# =============================================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# SYNTHETIC VOCABULARY
# =============================================================================

VOCAB_WORDS = [
    "patient",
    "clinical",
    "study",
    "treatment",
    "response",
    "analysis",
    "measurement",
    "baseline",
    "followup",
    "result",
    "model",
    "risk",
    "outcome",
    "variable",
    "estimate",
    "confidence",
    "interval",
    "sample",
    "group",
    "effect",
    "observed",
    "change",
    "significant",
    "consistent",
    "additional",
    "evaluation",
    "evidence",
    "association",
    "prediction",
    "assessment",
    "longitudinal",
    "temporal",
    "biomarker",
    "laboratory",
    "imaging",
    "protocol",
    "comparison",
    "distribution",
    "variance",
    "probability",
    "classification",
    "validation",
    "performance",
    "accuracy",
    "specificity",
    "sensitivity",
    "method",
    "dataset",
    "feature",
    "trajectory",
    "observation",
    "signal",
    "pattern",
]


# =============================================================================
# MEMBER DOCUMENT
# =============================================================================

def make_document(i, length=SEQ_LEN):

    tokens = []

    offset = i % len(VOCAB_WORDS)

    for j in range(length):

        idx = (
            offset
            + j * 7
            + i * 13
            + j * j
        ) % len(VOCAB_WORDS)

        tokens.append(
            VOCAB_WORDS[idx]
        )

    return " ".join(tokens)


# =============================================================================
# NON-MEMBER DOCUMENT
#
# Same vocabulary/domain.
# Independently generated.
# Never used for LM training.
# =============================================================================

def make_nonmember_document(i, length=SEQ_LEN):

    rng = random.Random(
        900000 + i
    )

    tokens = []

    for j in range(length):

        idx = (
            i * 17
            + j * 11
            + j * j * 3
            + rng.randint(0, 10)
        ) % len(VOCAB_WORDS)

        tokens.append(
            VOCAB_WORDS[idx]
        )

    return " ".join(tokens)


# =============================================================================
# TOKENIZATION
# =============================================================================

def tokenize(text):

    return text.lower().split()


# =============================================================================
# VOCABULARY
#
# IMPORTANT:
#
# The vocabulary is constructed independently of the LM parameters.
# This avoids unknown-token artifacts during evaluation.
# =============================================================================

def build_vocabulary(texts):

    counts = {}

    for text in texts:

        for token in tokenize(text):

            counts[token] = (
                counts.get(token, 0) + 1
            )

    ordered = sorted(
        counts.items(),
        key=lambda x: (-x[1], x[0])
    )

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for token, _ in ordered:

        if token not in vocab:

            vocab[token] = len(vocab)

    return vocab


# =============================================================================
# ENCODE
# =============================================================================

def encode_text(
    text,
    vocab
):

    ids = [
        vocab.get(
            token,
            vocab["<UNK>"]
        )
        for token in tokenize(text)
    ]

    if len(ids) < SEQ_LEN:

        ids += (
            [vocab["<PAD>"]]
            * (SEQ_LEN - len(ids))
        )

    else:

        ids = ids[:SEQ_LEN]

    return torch.tensor(
        ids,
        dtype=torch.long
    )


# =============================================================================
# LANGUAGE MODEL
# =============================================================================

class TinyLM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.output = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, x):

        emb = self.embedding(x)

        hidden, _ = self.gru(
            emb
        )

        logits = self.output(
            hidden
        )

        return logits


# =============================================================================
# TRAIN LM
# =============================================================================

def train_lm(
    model,
    ordinary_sequences,
    strong_sequences,
    vocab_size
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LM_LR
    )

    training_sequences = []

    # Ordinary members: once
    for seq in ordinary_sequences:

        training_sequences.append(seq)

    # Strong members: repeated
    for seq in strong_sequences:

        for _ in range(
            STRONG_REPEAT
        ):

            training_sequences.append(seq)

    print()
    print(
        f"LM training sequences = "
        f"{len(training_sequences)}"
    )

    print(
        f"  ordinary = "
        f"{len(ordinary_sequences)}"
    )

    print(
        f"  strong = "
        f"{len(strong_sequences)} "
        f"x {STRONG_REPEAT}"
    )

    model.train()

    for epoch in range(
        1,
        LM_EPOCHS + 1
    ):

        order = list(
            range(
                len(training_sequences)
            )
        )

        random.shuffle(order)

        losses = []

        for idx in order:

            seq = training_sequences[idx]

            x = (
                seq[:-1]
                .unsqueeze(0)
                .to(DEVICE)
            )

            y = (
                seq[1:]
                .unsqueeze(0)
                .to(DEVICE)
            )

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    vocab_size
                ),
                y.reshape(-1)
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            losses.append(
                float(loss.item())
            )

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == LM_EPOCHS
        ):

            print(
                f"LM epoch "
                f"{epoch:02d}/{LM_EPOCHS} "
                f"loss = "
                f"{np.mean(losses):.4f}"
            )


# =============================================================================
# GET CONTEXTUAL HIDDEN STATES
# =============================================================================
#
# This function deliberately separates:
#
#     context processing
#     suffix pooling
#
# The returned representation is ALWAYS one-dimensional.
# =============================================================================

@torch.no_grad()
def get_hidden_states(
    model,
    sequence
):

    model.eval()

    sequence = (
        sequence
        .to(DEVICE)
        .reshape(-1)
    )

    if sequence.numel() < 2:

        raise RuntimeError(
            "Sequence must contain at least "
            "two tokens."
        )

    x = sequence[:-1].unsqueeze(0)

    embeddings = model.embedding(x)

    hidden, _ = model.gru(
        embeddings
    )

    hidden = hidden.squeeze(0)

    if hidden.ndim != 2:

        raise RuntimeError(
            "Expected hidden states with "
            "shape [T,D], got "
            f"{tuple(hidden.shape)}"
        )

    if hidden.shape[1] != HIDDEN_DIM:

        raise RuntimeError(
            "Unexpected hidden dimension: "
            f"{hidden.shape[1]} "
            f"(expected {HIDDEN_DIM})"
        )

    return hidden.cpu()


# =============================================================================
# SUFFIX EMBEDDING
# =============================================================================
#
# IMPORTANT:
#
# The prefix determines the contextual state.
# Only states from the suffix are pooled.
#
# The output is guaranteed to be [HIDDEN_DIM].
# =============================================================================

@torch.no_grad()
def get_suffix_embedding(
    model,
    sequence,
    prefix_length
):

    hidden = get_hidden_states(
        model,
        sequence
    )

    T = hidden.shape[0]

    # Hidden state index t corresponds to token t.
    #
    # Prefix occupies:
    #
    #   0 ... prefix_length-1
    #
    # Therefore suffix begins at prefix_length.
    #
    suffix_start = min(
        max(prefix_length, 0),
        T
    )

    suffix_states = hidden[
        suffix_start:
    ]

    if suffix_states.numel() == 0:

        # Robust fallback.
        # This should not occur with the configured
        # prefix lengths and sequence length.
        representation = hidden[-1]

    else:

        representation = (
            suffix_states
            .mean(dim=0)
        )

    representation = (
        representation
        .reshape(-1)
        .float()
        .cpu()
    )

    if representation.ndim != 1:

        raise RuntimeError(
            "Suffix embedding must be 1-D. "
            f"Got shape "
            f"{tuple(representation.shape)}"
        )

    if representation.numel() != HIDDEN_DIM:

        raise RuntimeError(
            "Wrong suffix embedding dimension. "
            f"Expected {HIDDEN_DIM}, "
            f"got {representation.numel()}."
        )

    return representation


# =============================================================================
# GENERATE CONTINUATION
# =============================================================================

@torch.no_grad()
def generate_suffix(
    model,
    prefix,
    gen_len,
    temperature
):

    model.eval()

    generated = (
        prefix
        .reshape(-1)
        .to(DEVICE)
    )

    for _ in range(gen_len):

        logits = model(
            generated.unsqueeze(0)
        )

        next_logits = (
            logits[0, -1]
            / max(
                temperature,
                1e-6
            )
        )

        probs = F.softmax(
            next_logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probs,
            1
        )

        generated = torch.cat(
            [
                generated,
                next_token
            ]
        )

    return generated.cpu()


# =============================================================================
# BUILD ORIGINAL TRAJECTORY
# =============================================================================

@torch.no_grad()
def build_original_trajectory(
    model,
    sequence
):

    rows = []

    for prefix_length in PREFIX_LENGTHS:

        prefix_length = min(
            prefix_length,
            len(sequence) - 1
        )

        embedding = (
            get_suffix_embedding(
                model,
                sequence,
                prefix_length
            )
        )

        embedding = (
            embedding
            .reshape(-1)
        )

        if embedding.numel() != HIDDEN_DIM:

            raise RuntimeError(
                "Invalid original trajectory row."
            )

        rows.append(
            embedding
        )

    trajectory = torch.stack(
        rows,
        dim=0
    )

    expected = (
        len(PREFIX_LENGTHS),
        HIDDEN_DIM
    )

    if tuple(trajectory.shape) != expected:

        raise RuntimeError(
            "Original trajectory has wrong "
            f"shape {tuple(trajectory.shape)}; "
            f"expected {expected}."
        )

    return trajectory


# =============================================================================
# BUILD GENERATED TRAJECTORY
# =============================================================================

@torch.no_grad()
def build_generated_trajectory(
    model,
    sequence
):

    rows = []

    for prefix_length in PREFIX_LENGTHS:

        prefix_length = min(
            prefix_length,
            len(sequence) - 1
        )

        prefix = sequence[
            :prefix_length
        ]

        generated = (
            generate_suffix(
                model,
                prefix,
                GEN_LEN,
                TEMPERATURE
            )
        )

        embedding = (
            get_suffix_embedding(
                model,
                generated,
                prefix_length
            )
        )

        embedding = (
            embedding
            .reshape(-1)
        )

        if embedding.numel() != HIDDEN_DIM:

            raise RuntimeError(
                "Invalid generated trajectory row."
            )

        rows.append(
            embedding
        )

    trajectory = torch.stack(
        rows,
        dim=0
    )

    expected = (
        len(PREFIX_LENGTHS),
        HIDDEN_DIM
    )

    if tuple(trajectory.shape) != expected:

        raise RuntimeError(
            "Generated trajectory has wrong "
            f"shape {tuple(trajectory.shape)}; "
            f"expected {expected}."
        )

    return trajectory


# =============================================================================
# BUILD TRAJECTORY PAIR
# =============================================================================

@torch.no_grad()
def build_trajectory_pair(
    model,
    sequence
):

    original = (
        build_original_trajectory(
            model,
            sequence
        )
    )

    generated = (
        build_generated_trajectory(
            model,
            sequence
        )
    )

    expected = (
        len(PREFIX_LENGTHS),
        HIDDEN_DIM
    )

    if tuple(original.shape) != expected:

        raise RuntimeError(
            f"Original trajectory shape "
            f"{tuple(original.shape)} != "
            f"{expected}"
        )

    if tuple(generated.shape) != expected:

        raise RuntimeError(
            f"Generated trajectory shape "
            f"{tuple(generated.shape)} != "
            f"{expected}"
        )

    return (
        original,
        generated
    )


# =============================================================================
# TRAJECTORY CLASSIFIER
# =============================================================================
#
# Input:
#
#     batch x K x D
#
# The trajectory is NOT flattened.
#
# Conv1D operates along K, i.e. across prefix positions.
#
# =============================================================================

class TrajectoryCNN(nn.Module):

    def __init__(
        self,
        embedding_dim
    ):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=CNN_CHANNELS,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv1d(
            in_channels=CNN_CHANNELS,
            out_channels=CNN_CHANNELS,
            kernel_size=3,
            padding=1
        )

        self.norm1 = nn.BatchNorm1d(
            CNN_CHANNELS
        )

        self.norm2 = nn.BatchNorm1d(
            CNN_CHANNELS
        )

        self.pool = nn.AdaptiveAvgPool1d(
            1
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                CNN_CHANNELS,
                MLP_HIDDEN
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                MLP_HIDDEN,
                2
            )
        )

    def forward(self, x):

        if x.ndim != 3:

            raise RuntimeError(
                "Trajectory classifier expects "
                "[batch,K,D], got "
                f"{tuple(x.shape)}"
            )

        # [B,K,D] -> [B,D,K]
        x = x.transpose(
            1,
            2
        )

        x = self.conv1(x)

        x = self.norm1(x)

        x = F.relu(x)

        x = self.conv2(x)

        x = self.norm2(x)

        x = F.relu(x)

        x = self.pool(x)

        x = x.squeeze(-1)

        return self.classifier(x)


# =============================================================================
# BUILD CLASSIFIER DATASET
# =============================================================================

@torch.no_grad()
def build_trajectory_classifier_dataset(
    model,
    sequences
):

    X = []
    y = []

    print()
    print("=" * 110)
    print(
        "BUILD WHOLE-TRAJECTORY CLASSIFIER DATA"
    )
    print("=" * 110)

    for i, sequence in enumerate(
        sequences
    ):

        original, generated = (
            build_trajectory_pair(
                model,
                sequence
            )
        )

        X.append(
            original
        )

        y.append(0)

        X.append(
            generated
        )

        y.append(1)

        print(
            f"\rMatrices: "
            f"{i+1}/{len(sequences)}",
            end="",
            flush=True
        )

    print()

    X = torch.stack(
        X,
        dim=0
    )

    y = torch.tensor(
        y,
        dtype=torch.long
    )

    expected = (
        len(sequences) * 2,
        len(PREFIX_LENGTHS),
        HIDDEN_DIM
    )

    if tuple(X.shape) != expected:

        raise RuntimeError(
            "Classifier tensor has wrong shape: "
            f"{tuple(X.shape)}; "
            f"expected {expected}"
        )

    return X, y


# =============================================================================
# TRAIN CLASSIFIER WITH EARLY STOPPING
# =============================================================================

def train_classifier(
    X_train,
    y_train,
    X_val,
    y_val
):

    print()
    print("=" * 110)
    print(
        "TRAIN WHOLE-TRAJECTORY "
        "ORIGINAL-vs-GENERATED CLASSIFIER"
    )
    print("=" * 110)

    print(
        f"Training matrices = "
        f"{len(X_train)}"
    )

    print(
        f"Validation matrices = "
        f"{len(X_val)}"
    )

    print(
        f"Trajectory shape = "
        f"{tuple(X_train.shape[1:])}"
    )

    model = TrajectoryCNN(
        embedding_dim=HIDDEN_DIM
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CLS_LR,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val_loss = float("inf")

    epochs_without_improvement = 0

    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)

    X_val = X_val.to(DEVICE)
    y_val = y_val.to(DEVICE)

    for epoch in range(
        1,
        CLS_MAX_EPOCHS + 1
    ):

        # ---------------------------------------------------------
        # TRAIN
        # ---------------------------------------------------------

        model.train()

        permutation = torch.randperm(
            len(X_train),
            device=DEVICE
        )

        total_loss = 0.0
        correct = 0

        for start in range(
            0,
            len(X_train),
            BATCH_SIZE
        ):

            idx = permutation[
                start:
                start + BATCH_SIZE
            ]

            xb = X_train[idx]
            yb = y_train[idx]

            logits = model(xb)

            loss = criterion(
                logits,
                yb
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            total_loss += (
                loss.item()
                * len(idx)
            )

            pred = logits.argmax(
                dim=1
            )

            correct += int(
                (pred == yb).sum()
            )

        train_loss = (
            total_loss
            / len(X_train)
        )

        train_acc = (
            correct
            / len(X_train)
        )

        # ---------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_logits = model(
                X_val
            )

            val_loss = criterion(
                val_logits,
                y_val
            ).item()

            val_pred = (
                val_logits.argmax(
                    dim=1
                )
            )

            val_acc = float(
                (
                    val_pred == y_val
                ).float().mean()
            )

        # ---------------------------------------------------------
        # EARLY STOPPING
        # ---------------------------------------------------------

        improved = (
            val_loss
            < best_val_loss - 1e-6
        )

        if improved:

            best_val_loss = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epoch == 1
            or epoch % 10 == 0
            or improved
        ):

            print(
                f"Epoch "
                f"{epoch:03d}/{CLS_MAX_EPOCHS} | "
                f"train loss = "
                f"{train_loss:.4f} | "
                f"train acc = "
                f"{train_acc:.4f} | "
                f"val loss = "
                f"{val_loss:.4f} | "
                f"val acc = "
                f"{val_acc:.4f}"
            )

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                f"Early stopping at epoch "
                f"{epoch}."
            )

            break

    # ---------------------------------------------------------
    # Restore best validation model
    # ---------------------------------------------------------

    if best_state is None:

        raise RuntimeError(
            "No valid classifier checkpoint "
            "was created."
        )

    model.load_state_dict(
        best_state
    )

    model.to(DEVICE)

    model.eval()

    print(
        f"Best validation loss = "
        f"{best_val_loss:.4f}"
    )

    return model


# =============================================================================
# EVALUATE ONE TEXT
# =============================================================================

@torch.no_grad()
def evaluate_text(
    lm,
    classifier,
    sequence
):

    original, generated = (
        build_trajectory_pair(
            lm,
            sequence
        )
    )

    original_logits = classifier(
        original.unsqueeze(0).to(DEVICE)
    )

    generated_logits = classifier(
        generated.unsqueeze(0).to(DEVICE)
    )

    original_probs = F.softmax(
        original_logits,
        dim=1
    )[0]

    generated_probs = F.softmax(
        generated_logits,
        dim=1
    )[0]

    p_original_original = float(
        original_probs[0]
    )

    p_original_generated = float(
        generated_probs[0]
    )

    separation = (
        p_original_original
        - p_original_generated
    )

    # Prefix-wise Euclidean distance
    distances = torch.norm(
        original - generated,
        dim=1
    ).numpy()

    return {
        "p_original_original":
            p_original_original,

        "p_original_generated":
            p_original_generated,

        "separation":
            separation,

        "original":
            original,

        "generated":
            generated,

        "distances":
            distances,
    }


# =============================================================================
# AUC
# =============================================================================

def auc_score(
    positive_scores,
    negative_scores
):

    positive_scores = np.asarray(
        positive_scores,
        dtype=float
    )

    negative_scores = np.asarray(
        negative_scores,
        dtype=float
    )

    if (
        len(positive_scores) == 0
        or len(negative_scores) == 0
    ):

        return np.nan

    scores = np.concatenate(
        [
            positive_scores,
            negative_scores
        ]
    )

    labels = np.concatenate(
        [
            np.ones(
                len(positive_scores)
            ),
            np.zeros(
                len(negative_scores)
            )
        ]
    )

    order = np.argsort(
        scores
    )

    sorted_scores = scores[
        order
    ]

    sorted_labels = labels[
        order
    ]

    ranks = np.empty(
        len(sorted_scores),
        dtype=float
    )

    i = 0

    while i < len(sorted_scores):

        j = i + 1

        while (
            j < len(sorted_scores)
            and sorted_scores[j]
            == sorted_scores[i]
        ):

            j += 1

        ranks[i:j] = (
            (i + 1 + j)
            / 2.0
        )

        i = j

    positive_ranks = ranks[
        sorted_labels == 1
    ]

    n_pos = len(
        positive_scores
    )

    n_neg = len(
        negative_scores
    )

    return float(
        (
            positive_ranks.sum()
            - n_pos * (n_pos + 1) / 2
        )
        / (
            n_pos * n_neg
        )
    )


# =============================================================================
# BOOTSTRAP AUC CI
# =============================================================================

def bootstrap_auc_ci(
    positive_scores,
    negative_scores,
    n_bootstrap=N_BOOTSTRAP,
    seed=12345
):

    positive_scores = np.asarray(
        positive_scores,
        dtype=float
    )

    negative_scores = np.asarray(
        negative_scores,
        dtype=float
    )

    rng = np.random.default_rng(
        seed
    )

    observed = auc_score(
        positive_scores,
        negative_scores
    )

    boot = []

    for _ in range(
        n_bootstrap
    ):

        pos = rng.choice(
            positive_scores,
            size=len(positive_scores),
            replace=True
        )

        neg = rng.choice(
            negative_scores,
            size=len(negative_scores),
            replace=True
        )

        value = auc_score(
            pos,
            neg
        )

        if np.isfinite(value):

            boot.append(value)

    if len(boot) == 0:

        return (
            observed,
            np.nan,
            np.nan
        )

    lower = np.percentile(
        boot,
        2.5
    )

    upper = np.percentile(
        boot,
        97.5
    )

    return (
        observed,
        float(lower),
        float(upper)
    )


# =============================================================================
# COHEN'S D
# =============================================================================

def cohens_d(
    x,
    y
):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    nx = len(x)
    ny = len(y)

    vx = np.var(
        x,
        ddof=1
    )

    vy = np.var(
        y,
        ddof=1
    )

    pooled_sd = np.sqrt(
        (
            (nx - 1) * vx
            + (ny - 1) * vy
        )
        / (
            nx + ny - 2
        )
    )

    if pooled_sd == 0:

        return np.nan

    return float(
        (
            np.mean(x)
            - np.mean(y)
        )
        / pooled_sd
    )


# =============================================================================
# EVALUATE GROUP
# =============================================================================

def evaluate_group(
    name,
    lm,
    classifier,
    sequences
):

    print()
    print("=" * 110)
    print(
        f"EVALUATING {name}"
    )
    print("=" * 110)

    results = []

    for i, sequence in enumerate(
        sequences
    ):

        result = evaluate_text(
            lm,
            classifier,
            sequence
        )

        results.append(
            result
        )

        print(
            f"\r{name}: "
            f"{i+1}/{len(sequences)}",
            end="",
            flush=True
        )

    print()

    scores = np.array(
        [
            x["separation"]
            for x in results
        ],
        dtype=float
    )

    p_original_original = np.array(
        [
            x["p_original_original"]
            for x in results
        ],
        dtype=float
    )

    p_original_generated = np.array(
        [
            x["p_original_generated"]
            for x in results
        ],
        dtype=float
    )

    distances = np.stack(
        [
            x["distances"]
            for x in results
        ],
        axis=0
    )

    return {
        "results": results,
        "scores": scores,
        "p_original_original":
            p_original_original,
        "p_original_generated":
            p_original_generated,
        "distances":
            distances,
    }


# =============================================================================
# TRAJECTORY SHAPE SANITY CHECK
# =============================================================================

def trajectory_sanity_check(
    lm,
    sequence
):

    print()
    print("=" * 110)
    print(
        "TRAJECTORY SHAPE SANITY CHECK"
    )
    print("=" * 110)

    original, generated = (
        build_trajectory_pair(
            lm,
            sequence
        )
    )

    expected = (
        len(PREFIX_LENGTHS),
        HIDDEN_DIM
    )

    print(
        f"Original trajectory shape = "
        f"{tuple(original.shape)}"
    )

    print(
        f"Generated trajectory shape = "
        f"{tuple(generated.shape)}"
    )

    print(
        f"Expected trajectory shape = "
        f"{expected}"
    )

    if tuple(original.shape) != expected:

        raise RuntimeError(
            "Original trajectory shape check failed."
        )

    if tuple(generated.shape) != expected:

        raise RuntimeError(
            "Generated trajectory shape check failed."
        )

    print(
        "✓ Trajectory shape check passed."
    )


# =============================================================================
# ONE EXPERIMENT
# =============================================================================

def run_experiment(
    repeat
):

    start_time = time.time()

    print()
    print("#" * 110)
    print(
        f"REPEAT {repeat}/{N_REPEATS}"
    )
    print("#" * 110)

    set_seed(
        SEED + repeat
    )

    # =========================================================================
    # CREATE CORPUS
    # =========================================================================

    print()
    print(
        "Creating synthetic corpus..."
    )

    ordinary_texts = [
        make_document(i)
        for i in range(
            N_ORDINARY_MEMBERS
        )
    ]

    strong_texts = [
        make_document(
            1000 + i
        )
        for i in range(
            N_STRONG_MEMBERS
        )
    ]

    classifier_train_texts = [
        make_document(
            10000 + i
        )
        for i in range(
            N_CLASSIFIER_TRAIN
        )
    ]

    classifier_val_texts = [
        make_document(
            20000 + i
        )
        for i in range(
            N_CLASSIFIER_VAL
        )
    ]

    nonmember_texts = [
        make_nonmember_document(i)
        for i in range(
            N_NONMEMBERS
        )
    ]

    # =========================================================================
    # EVALUATION
    # =========================================================================

    eval_ordinary_texts = (
        ordinary_texts[
            :N_EVAL_ORDINARY
        ]
    )

    eval_strong_texts = (
        strong_texts
    )

    # =========================================================================
    # VOCABULARY
    # =========================================================================

    print(
        "Building vocabulary..."
    )

    all_texts = (
        ordinary_texts
        + strong_texts
        + classifier_train_texts
        + classifier_val_texts
        + nonmember_texts
    )

    vocab = build_vocabulary(
        all_texts
    )

    print(
        f"Vocabulary size = "
        f"{len(vocab)}"
    )

    # =========================================================================
    # ENCODE
    # =========================================================================

    print(
        "Encoding texts..."
    )

    ordinary_sequences = [
        encode_text(
            x,
            vocab
        )
        for x in ordinary_texts
    ]

    strong_sequences = [
        encode_text(
            x,
            vocab
        )
        for x in strong_texts
    ]

    classifier_train_sequences = [
        encode_text(
            x,
            vocab
        )
        for x in classifier_train_texts
    ]

    classifier_val_sequences = [
        encode_text(
            x,
            vocab
        )
        for x in classifier_val_texts
    ]

    eval_ordinary_sequences = [
        encode_text(
            x,
            vocab
        )
        for x in eval_ordinary_texts
    ]

    eval_strong_sequences = [
        encode_text(
            x,
            vocab
        )
        for x in eval_strong_texts
    ]

    nonmember_sequences = [
        encode_text(
            x,
            vocab
        )
        for x in nonmember_texts
    ]

    # =========================================================================
    # STEP 1: TRAIN LM
    # =========================================================================

    print()
    print("=" * 110)
    print(
        "STEP 1: TRAIN LANGUAGE MODEL"
    )
    print("=" * 110)

    print(
        "The LM sees:"
    )

    print(
        f"  ordinary members = "
        f"{len(ordinary_sequences)}"
    )

    print(
        f"  strongly memorized = "
        f"{len(strong_sequences)}"
    )

    print(
        f"  repetitions per strong text = "
        f"{STRONG_REPEAT}"
    )

    print(
        "The LM does NOT see:"
    )

    print(
        "  classifier validation texts"
    )

    print(
        "  non-members"
    )

    lm = TinyLM(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    train_lm(
        lm,
        ordinary_sequences,
        strong_sequences,
        len(vocab)
    )

    # =========================================================================
    # SHAPE CHECK
    # =========================================================================

    trajectory_sanity_check(
        lm,
        classifier_train_sequences[0]
    )

    # =========================================================================
    # STEP 2: CLASSIFIER DATA
    # =========================================================================

    print()
    print("=" * 110)
    print(
        "STEP 2: BUILD WHOLE-TRAJECTORY "
        "CLASSIFIER DATA"
    )
    print("=" * 110)

    X_train, y_train = (
        build_trajectory_classifier_dataset(
            lm,
            classifier_train_sequences
        )
    )

    X_val, y_val = (
        build_trajectory_classifier_dataset(
            lm,
            classifier_val_sequences
        )
    )

    print()
    print(
        f"Training trajectory tensor = "
        f"{tuple(X_train.shape)}"
    )

    print(
        f"Validation trajectory tensor = "
        f"{tuple(X_val.shape)}"
    )

    # =========================================================================
    # STEP 3: TRAIN CLASSIFIER
    # =========================================================================

    classifier = train_classifier(
        X_train,
        y_train,
        X_val,
        y_val
    )

    # =========================================================================
    # STEP 4: FINAL EVALUATION
    # =========================================================================

    strong_eval = evaluate_group(
        "STRONGLY MEMORIZED",
        lm,
        classifier,
        eval_strong_sequences
    )

    ordinary_eval = evaluate_group(
        "ORDINARY MEMBERS",
        lm,
        classifier,
        eval_ordinary_sequences
    )

    nonmember_eval = evaluate_group(
        "NON-MEMBERS",
        lm,
        classifier,
        nonmember_sequences
    )

    # =========================================================================
    # SCORES
    # =========================================================================

    strong_scores = (
        strong_eval["scores"]
    )

    ordinary_scores = (
        ordinary_eval["scores"]
    )

    nonmember_scores = (
        nonmember_eval["scores"]
    )

    # =========================================================================
    # AUC
    # =========================================================================

    strong_auc, strong_low, strong_high = (
        bootstrap_auc_ci(
            strong_scores,
            nonmember_scores,
            seed=10000 + repeat
        )
    )

    ordinary_auc, ordinary_low, ordinary_high = (
        bootstrap_auc_ci(
            ordinary_scores,
            nonmember_scores,
            seed=20000 + repeat
        )
    )

    strong_vs_ordinary_auc, svo_low, svo_high = (
        bootstrap_auc_ci(
            strong_scores,
            ordinary_scores,
            seed=30000 + repeat
        )
    )

    # =========================================================================
    # EFFECT SIZES
    # =========================================================================

    d_strong = cohens_d(
        strong_scores,
        nonmember_scores
    )

    d_ordinary = cohens_d(
        ordinary_scores,
        nonmember_scores
    )

    d_strong_ordinary = cohens_d(
        strong_scores,
        ordinary_scores
    )

    # =========================================================================
    # PREFIX DISTANCE AUC
    #
    # Euclidean distance is naturally:
    #
    #     larger = more different
    #
    # Therefore positive = strong and negative = non-member.
    #
    # This fixes the direction problem in the previous script.
    # =========================================================================

    distance_auc_strong = []

    distance_auc_ordinary = []

    for j, prefix in enumerate(
        PREFIX_LENGTHS
    ):

        strong_dist = (
            strong_eval["distances"][:, j]
        )

        ordinary_dist = (
            ordinary_eval["distances"][:, j]
        )

        nonmember_dist = (
            nonmember_eval["distances"][:, j]
        )

        auc_s = auc_score(
            strong_dist,
            nonmember_dist
        )

        auc_o = auc_score(
            ordinary_dist,
            nonmember_dist
        )

        distance_auc_strong.append(
            auc_s
        )

        distance_auc_ordinary.append(
            auc_o
        )

    # =========================================================================
    # PRINT PRIMARY RESULTS
    # =========================================================================

    print()
    print("=" * 110)
    print(
        "PRIMARY MEMBERSHIP RESULTS"
    )
    print("=" * 110)

    print()
    print(
        "Strongly memorized vs non-member:"
    )

    print(
        f"  AUC = "
        f"{strong_auc:.4f}"
    )

    print(
        f"  95% bootstrap CI = "
        f"[{strong_low:.4f}, "
        f"{strong_high:.4f}]"
    )

    print()
    print(
        "Ordinary member vs non-member:"
    )

    print(
        f"  AUC = "
        f"{ordinary_auc:.4f}"
    )

    print(
        f"  95% bootstrap CI = "
        f"[{ordinary_low:.4f}, "
        f"{ordinary_high:.4f}]"
    )

    print()
    print(
        "Strongly memorized vs ordinary member:"
    )

    print(
        f"  AUC = "
        f"{strong_vs_ordinary_auc:.4f}"
    )

    print(
        f"  95% bootstrap CI = "
        f"[{svo_low:.4f}, "
        f"{svo_high:.4f}]"
    )

    # =========================================================================
    # SCORE DISTRIBUTIONS
    # =========================================================================

    print()
    print(
        "MEMBERSHIP SCORE DISTRIBUTIONS"
    )

    print()
    print(
        "Strongly memorized:"
    )

    print(
        f"  mean = "
        f"{np.mean(strong_scores):.4f}"
    )

    print(
        f"  SD   = "
        f"{np.std(strong_scores, ddof=1):.4f}"
    )

    print()
    print(
        "Ordinary members:"
    )

    print(
        f"  mean = "
        f"{np.mean(ordinary_scores):.4f}"
    )

    print(
        f"  SD   = "
        f"{np.std(ordinary_scores, ddof=1):.4f}"
    )

    print()
    print(
        "Non-members:"
    )

    print(
        f"  mean = "
        f"{np.mean(nonmember_scores):.4f}"
    )

    print(
        f"  SD   = "
        f"{np.std(nonmember_scores, ddof=1):.4f}"
    )

    # =========================================================================
    # EFFECT SIZES
    # =========================================================================

    print()
    print(
        "EFFECT SIZES"
    )

    print(
        f"Cohen's d: strong vs non-member = "
        f"{d_strong:.4f}"
    )

    print(
        f"Cohen's d: ordinary vs non-member = "
        f"{d_ordinary:.4f}"
    )

    print(
        f"Cohen's d: strong vs ordinary = "
        f"{d_strong_ordinary:.4f}"
    )

    # =========================================================================
    # CLASSIFIER PROBABILITIES
    # =========================================================================

    print()
    print(
        "CLASSIFIER PROBABILITIES"
    )

    print()

    print(
        "Strongly memorized:"
    )

    print(
        f"  p(original | original trajectory) = "
        f"{np.mean(strong_eval['p_original_original']):.4f}"
    )

    print(
        f"  p(original | generated trajectory) = "
        f"{np.mean(strong_eval['p_original_generated']):.4f}"
    )

    print()

    print(
        "Ordinary members:"
    )

    print(
        f"  p(original | original trajectory) = "
        f"{np.mean(ordinary_eval['p_original_original']):.4f}"
    )

    print(
        f"  p(original | generated trajectory) = "
        f"{np.mean(ordinary_eval['p_original_generated']):.4f}"
    )

    print()

    print(
        "Non-members:"
    )

    print(
        f"  p(original | original trajectory) = "
        f"{np.mean(nonmember_eval['p_original_original']):.4f}"
    )

    print(
        f"  p(original | generated trajectory) = "
        f"{np.mean(nonmember_eval['p_original_generated']):.4f}"
    )

    # =========================================================================
    # PREFIX DIAGNOSTIC
    # =========================================================================

    print()
    print("=" * 110)
    print(
        "PREFIX-WISE EMBEDDING DISTANCE DIAGNOSTIC"
    )
    print("=" * 110)

    for j, prefix in enumerate(
        PREFIX_LENGTHS
    ):

        print(
            f"Prefix {prefix:02d}: "
            f"strong-vs-nonmember "
            f"AUC = "
            f"{distance_auc_strong[j]:.4f} | "
            f"ordinary-vs-nonmember "
            f"AUC = "
            f"{distance_auc_ordinary[j]:.4f}"
        )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    classifier.eval()

    with torch.no_grad():

        val_logits = classifier(
            X_val.to(DEVICE)
        )

        val_pred = (
            val_logits.argmax(
                dim=1
            )
        )

        val_accuracy = float(
            (
                val_pred
                == y_val.to(DEVICE)
            )
            .float()
            .mean()
        )

    print()
    print(
        "FINAL CLASSIFIER VALIDATION ACCURACY = "
        f"{val_accuracy:.4f}"
    )

    # =========================================================================
    # INTERPRETATION
    # =========================================================================

    print()
    print("=" * 110)
    print(
        "INTERPRETATION"
    )
    print("=" * 110)

    print(
        """
This experiment is a controlled positive-control memorization benchmark.

The language model is trained only on member texts.

Strongly memorized texts receive repeated exposure during LM training,
whereas ordinary members are observed only once.

The trajectory classifier is trained exclusively on separate classifier
texts and receives only the labels:

    original trajectory
            vs
    generated trajectory

It never receives membership labels.

Membership inference is performed only on the held-out evaluation groups.

The key hypothesis is:

    AUC(strongly memorized vs non-member)
        >
    AUC(ordinary member vs non-member)

A successful result therefore demonstrates that the proposed
whole-trajectory original-vs-generated representation is sensitive
to increasing memorization.

This experiment should be described as a controlled positive-control
experiment rather than as a realistic estimate of attack performance
under naturally occurring LM memorization.
"""
    )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print(
        f"Experiment runtime = "
        f"{elapsed:.1f} seconds"
    )

    # =========================================================================
    # RETURN
    # =========================================================================

    return {

        "repeat":
            repeat,

        "auc_strong":
            strong_auc,

        "auc_strong_low":
            strong_low,

        "auc_strong_high":
            strong_high,

        "auc_ordinary":
            ordinary_auc,

        "auc_ordinary_low":
            ordinary_low,

        "auc_ordinary_high":
            ordinary_high,

        "auc_strong_vs_ordinary":
            strong_vs_ordinary_auc,

        "auc_svo_low":
            svo_low,

        "auc_svo_high":
            svo_high,

        "d_strong":
            d_strong,

        "d_ordinary":
            d_ordinary,

        "d_strong_vs_ordinary":
            d_strong_ordinary,

        "strong_mean":
            np.mean(strong_scores),

        "strong_sd":
            np.std(
                strong_scores,
                ddof=1
            ),

        "ordinary_mean":
            np.mean(ordinary_scores),

        "ordinary_sd":
            np.std(
                ordinary_scores,
                ddof=1
            ),

        "nonmember_mean":
            np.mean(nonmember_scores),

        "nonmember_sd":
            np.std(
                nonmember_scores,
                ddof=1
            ),

        "val_accuracy":
            val_accuracy,

        "prefix_distance_auc_strong":
            distance_auc_strong,

        "prefix_distance_auc_ordinary":
            distance_auc_ordinary,
    }


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(
    results,
    filename
):

    fieldnames = [
        "repeat",

        "auc_strong",
        "auc_strong_low",
        "auc_strong_high",

        "auc_ordinary",
        "auc_ordinary_low",
        "auc_ordinary_high",

        "auc_strong_vs_ordinary",
        "auc_svo_low",
        "auc_svo_high",

        "d_strong",
        "d_ordinary",
        "d_strong_vs_ordinary",

        "strong_mean",
        "strong_sd",

        "ordinary_mean",
        "ordinary_sd",

        "nonmember_mean",
        "nonmember_sd",

        "val_accuracy",
    ]

    with open(
        filename,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            row = {
                key: result[key]
                for key in fieldnames
            }

            writer.writerow(row)

    print()
    print(
        f"Results saved to: "
        f"{os.path.abspath(filename)}"
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    all_results = []

    for repeat in range(
        1,
        N_REPEATS + 1
    ):

        result = run_experiment(
            repeat
        )

        all_results.append(
            result
        )

    # =========================================================================
    # AGGREGATION
    # =========================================================================

    print()
    print("#" * 110)
    print(
        "FINAL SUMMARY"
    )
    print("#" * 110)

    strong_auc = np.array(
        [
            x["auc_strong"]
            for x in all_results
        ]
    )

    ordinary_auc = np.array(
        [
            x["auc_ordinary"]
            for x in all_results
        ]
    )

    strong_vs_ordinary_auc = np.array(
        [
            x["auc_strong_vs_ordinary"]
            for x in all_results
        ]
    )

    strong_d = np.array(
        [
            x["d_strong"]
            for x in all_results
        ]
    )

    ordinary_d = np.array(
        [
            x["d_ordinary"]
            for x in all_results
        ]
    )

    validation_accuracy = np.array(
        [
            x["val_accuracy"]
            for x in all_results
        ]
    )

    # =========================================================================
    # AUC SUMMARY
    # =========================================================================

    print()

    print(
        "Strongly memorized vs non-member:"
    )

    print(
        f"  AUC = "
        f"{strong_auc.mean():.4f} "
        f"+/- "
        f"{strong_auc.std(ddof=1):.4f}"
    )

    print()

    print(
        "Ordinary member vs non-member:"
    )

    print(
        f"  AUC = "
        f"{ordinary_auc.mean():.4f} "
        f"+/- "
        f"{ordinary_auc.std(ddof=1):.4f}"
    )

    print()

    print(
        "Strongly memorized vs ordinary member:"
    )

    print(
        f"  AUC = "
        f"{strong_vs_ordinary_auc.mean():.4f} "
        f"+/- "
        f"{strong_vs_ordinary_auc.std(ddof=1):.4f}"
    )

    # =========================================================================
    # EFFECT SIZE SUMMARY
    # =========================================================================

    print()
    print(
        "EFFECT SIZE SUMMARY"
    )

    print(
        f"Strong vs non-member Cohen's d: "
        f"{strong_d.mean():.4f} "
        f"+/- "
        f"{strong_d.std(ddof=1):.4f}"
    )

    print(
        f"Ordinary vs non-member Cohen's d: "
        f"{ordinary_d.mean():.4f} "
        f"+/- "
        f"{ordinary_d.std(ddof=1):.4f}"
    )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    print()
    print(
        "CLASSIFIER VALIDATION ACCURACY:"
    )

    print(
        f"  mean = "
        f"{validation_accuracy.mean():.4f}"
    )

    print(
        f"  SD   = "
        f"{validation_accuracy.std(ddof=1):.4f}"
    )

    # =========================================================================
    # PREFIX SUMMARY
    # =========================================================================

    print()
    print("=" * 110)
    print(
        "PREFIX-WISE DISTANCE AUC SUMMARY"
    )
    print("=" * 110)

    for j, prefix in enumerate(
        PREFIX_LENGTHS
    ):

        strong_values = np.array(
            [
                x[
                    "prefix_distance_auc_strong"
                ][j]
                for x in all_results
            ]
        )

        ordinary_values = np.array(
            [
                x[
                    "prefix_distance_auc_ordinary"
                ][j]
                for x in all_results
            ]
        )

        print(
            f"Prefix {prefix:02d}: "
            f"strong = "
            f"{strong_values.mean():.4f} "
            f"+/- "
            f"{strong_values.std(ddof=1):.4f} | "
            f"ordinary = "
            f"{ordinary_values.mean():.4f} "
            f"+/- "
            f"{ordinary_values.std(ddof=1):.4f}"
        )

    # =========================================================================
    # SAVE
    # =========================================================================

    save_results(
        all_results,
        RESULTS_CSV
    )

    # =========================================================================
    # FINAL MESSAGE
    # =========================================================================

    print()
    print("=" * 110)
    print(
        "DONE"
    )
    print("=" * 110)

    print()
    print(
        "Publication-relevant primary endpoint:"
    )

    print(
        "  Strongly memorized vs non-member AUC"
    )

    print()
    print(
        "Secondary endpoint:"
    )

    print(
        "  Ordinary member vs non-member AUC"
    )

    print()
    print(
        "Memorization-gradient endpoint:"
    )

    print(
        "  Strongly memorized vs ordinary member AUC"
    )

    print()
    print(
        "The classifier never receives membership labels."
    )

    print(
        "Membership evaluation is performed only after "
        "classifier training."
    )

    print(
        "Results are saved to:"
    )

    print(
        f"  {os.path.abspath(RESULTS_CSV)}"
    )