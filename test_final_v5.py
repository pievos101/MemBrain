# =============================================================================
# WHOLE-TRAJECTORY SUFFIX-EMBEDDING MEMBERSHIP BENCHMARK
#
# PUBLICATION-ORIENTED POSITIVE-CONTROL BENCHMARK
#
# EXTENDED WITH POINT-WISE BASELINE
# =============================================================================
#
# CORE COMPARISON
# ---------------
#
# PROPOSED:
#
#     complete K x D trajectory
#              |
#              v
#       trajectory CNN
#
#
# BASELINE:
#
#     one D-dimensional representation
#              |
#              v
#       point-wise MLP
#
#
# The same classifier-training texts are used for both approaches.
#
# The point-wise baseline is evaluated separately at every prefix:
#
#     prefix 8
#     prefix 16
#     prefix 24
#     prefix 32
#
# The primary point-wise baseline is PREFIX 32, which provides the
# longest available context.
#
# This allows us to test whether the complete trajectory contains
# additional membership information beyond any single representation.
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
print("WITH POINT-WISE BASELINE")
print("=" * 110)
print(f"Device = {DEVICE}")


# =============================================================================
# DATA
# =============================================================================

N_ORDINARY_MEMBERS = 80
N_STRONG_MEMBERS = 10

N_CLASSIFIER_TRAIN = 30
N_CLASSIFIER_VAL = 10

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
# TRAJECTORY
# =============================================================================

PREFIX_LENGTHS = [
    8,
    16,
    24,
    32,
]

K = len(PREFIX_LENGTHS)

# Primary point-wise baseline.
#
# This is fixed a priori as the longest-context single-point baseline.
SINGLE_POINT_PREFIX = 32

SINGLE_POINT_INDEX = PREFIX_LENGTHS.index(
    SINGLE_POINT_PREFIX
)


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
# WHOLE-TRAJECTORY CLASSIFIER
# =============================================================================

CNN_CHANNELS = 24
MLP_HIDDEN = 48

CLS_MAX_EPOCHS = 80
CLS_LR = 2e-3

WEIGHT_DECAY = 1e-4
DROPOUT = 0.10

BATCH_SIZE = 16

PATIENCE = 10


# =============================================================================
# POINT-WISE CLASSIFIER
# =============================================================================

POINT_MLP_HIDDEN = 48

POINT_MAX_EPOCHS = 80
POINT_LR = 2e-3

POINT_DROPOUT = 0.10
POINT_WEIGHT_DECAY = 1e-4

POINT_PATIENCE = 10


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

RESULTS_CSV = (
    "whole_trajectory_vs_pointwise_results.csv"
)


# =============================================================================
# SEED
# =============================================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

    for seq in ordinary_sequences:

        training_sequences.append(seq)

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
# HIDDEN STATES
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
            "shape [T,D]."
        )

    if hidden.shape[1] != HIDDEN_DIM:

        raise RuntimeError(
            "Unexpected hidden dimension."
        )

    return hidden.cpu()


# =============================================================================
# SUFFIX EMBEDDING
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

    suffix_start = min(
        max(prefix_length, 0),
        T
    )

    suffix_states = hidden[
        suffix_start:
    ]

    if suffix_states.numel() == 0:

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

    if representation.numel() != HIDDEN_DIM:

        raise RuntimeError(
            "Wrong suffix embedding dimension."
        )

    return representation


# =============================================================================
# GENERATION
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
# WHOLE TRAJECTORY
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
            f"Wrong trajectory shape: "
            f"{tuple(trajectory.shape)}"
        )

    return trajectory


# =============================================================================
# GENERATED TRAJECTORY
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
            f"Wrong generated trajectory shape: "
            f"{tuple(trajectory.shape)}"
        )

    return trajectory


# =============================================================================
# TRAJECTORY PAIR
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

    return (
        original,
        generated
    )


# =============================================================================
# WHOLE-TRAJECTORY CNN
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
                "Expected [B,K,D]."
            )

        # B x K x D -> B x D x K
        x = x.transpose(
            1,
            2
        )

        x = F.relu(
            self.norm1(
                self.conv1(x)
            )
        )

        x = F.relu(
            self.norm2(
                self.conv2(x)
            )
        )

        x = self.pool(x)

        x = x.squeeze(-1)

        return self.classifier(x)


# =============================================================================
# POINT-WISE CLASSIFIER
# =============================================================================
#
# Input:
#
#     one D-dimensional suffix representation.
#
# It therefore has access to exactly one point in the trajectory.
#
# =============================================================================

class PointwiseClassifier(nn.Module):

    def __init__(
        self,
        embedding_dim
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embedding_dim,
                POINT_MLP_HIDDEN
            ),

            nn.ReLU(),

            nn.Dropout(
                POINT_DROPOUT
            ),

            nn.Linear(
                POINT_MLP_HIDDEN,
                2
            )
        )

    def forward(self, x):

        if x.ndim != 2:

            raise RuntimeError(
                "Point-wise classifier expects "
                "[B,D]."
            )

        return self.network(x)


# =============================================================================
# BUILD TRAJECTORY CLASSIFIER DATA
# =============================================================================

@torch.no_grad()
def build_trajectory_classifier_dataset(
    model,
    sequences
):

    X = []
    y = []

    print()
    print(
        "BUILD WHOLE-TRAJECTORY "
        "CLASSIFIER DATA"
    )

    for i, sequence in enumerate(
        sequences
    ):

        original, generated = (
            build_trajectory_pair(
                model,
                sequence
            )
        )

        X.append(original)
        y.append(0)

        X.append(generated)
        y.append(1)

        print(
            f"\rMatrices: "
            f"{i+1}/{len(sequences)}",
            end="",
            flush=True
        )

    print()

    return (
        torch.stack(X),
        torch.tensor(
            y,
            dtype=torch.long
        )
    )


# =============================================================================
# TRAIN WHOLE-TRAJECTORY CLASSIFIER
# =============================================================================

def train_classifier(
    X_train,
    y_train,
    X_val,
    y_val
):

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

    print()
    print(
        "TRAIN WHOLE-TRAJECTORY "
        "CLASSIFIER"
    )

    for epoch in range(
        1,
        CLS_MAX_EPOCHS + 1
    ):

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
                start:start + BATCH_SIZE
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

            correct += int(
                (
                    logits.argmax(dim=1)
                    == yb
                ).sum()
            )

        train_loss = (
            total_loss
            / len(X_train)
        )

        train_acc = (
            correct
            / len(X_train)
        )

        model.eval()

        with torch.no_grad():

            val_logits = model(
                X_val
            )

            val_loss = criterion(
                val_logits,
                y_val
            ).item()

            val_acc = float(
                (
                    val_logits.argmax(dim=1)
                    == y_val
                )
                .float()
                .mean()
            )

        improved = (
            val_loss
            < best_val_loss - 1e-6
        )

        if improved:

            best_val_loss = val_loss

            best_state = {
                k: v.detach().cpu().clone()
                for k, v
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
                f"Epoch {epoch:03d} | "
                f"train loss={train_loss:.4f} | "
                f"train acc={train_acc:.4f} | "
                f"val loss={val_loss:.4f} | "
                f"val acc={val_acc:.4f}"
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

    model.load_state_dict(
        best_state
    )

    model.to(DEVICE)
    model.eval()

    return model


# =============================================================================
# TRAIN POINT-WISE CLASSIFIER
# =============================================================================

def train_pointwise_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    prefix_index
):

    model = PointwiseClassifier(
        embedding_dim=HIDDEN_DIM
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=POINT_LR,
        weight_decay=POINT_WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Select ONE prefix.
    X_train_point = (
        X_train[:, prefix_index, :]
        .to(DEVICE)
    )

    X_val_point = (
        X_val[:, prefix_index, :]
        .to(DEVICE)
    )

    y_train = y_train.to(DEVICE)
    y_val = y_val.to(DEVICE)

    print()
    print(
        "=" * 110
    )
    print(
        "TRAIN POINT-WISE CLASSIFIER"
    )
    print(
        f"Prefix = "
        f"{PREFIX_LENGTHS[prefix_index]}"
    )
    print(
        "=" * 110
    )

    for epoch in range(
        1,
        POINT_MAX_EPOCHS + 1
    ):

        model.train()

        permutation = torch.randperm(
            len(X_train_point),
            device=DEVICE
        )

        total_loss = 0.0
        correct = 0

        for start in range(
            0,
            len(X_train_point),
            BATCH_SIZE
        ):

            idx = permutation[
                start:start + BATCH_SIZE
            ]

            xb = X_train_point[idx]
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

            correct += int(
                (
                    logits.argmax(dim=1)
                    == yb
                ).sum()
            )

        train_loss = (
            total_loss
            / len(X_train_point)
        )

        train_acc = (
            correct
            / len(X_train_point)
        )

        model.eval()

        with torch.no_grad():

            val_logits = model(
                X_val_point
            )

            val_loss = criterion(
                val_logits,
                y_val
            ).item()

            val_acc = float(
                (
                    val_logits.argmax(dim=1)
                    == y_val
                )
                .float()
                .mean()
            )

        improved = (
            val_loss
            < best_val_loss - 1e-6
        )

        if improved:

            best_val_loss = val_loss

            best_state = {
                k: v.detach().cpu().clone()
                for k, v
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
                f"Epoch {epoch:03d} | "
                f"train loss={train_loss:.4f} | "
                f"train acc={train_acc:.4f} | "
                f"val loss={val_loss:.4f} | "
                f"val acc={val_acc:.4f}"
            )

        if (
            epochs_without_improvement
            >= POINT_PATIENCE
        ):

            print(
                f"Early stopping at epoch "
                f"{epoch}."
            )

            break

    model.load_state_dict(
        best_state
    )

    model.to(DEVICE)
    model.eval()

    return model


# =============================================================================
# EVALUATE ONE TEXT
# =============================================================================

@torch.no_grad()
def evaluate_text(
    lm,
    trajectory_classifier,
    pointwise_classifiers,
    sequence
):

    original, generated = (
        build_trajectory_pair(
            lm,
            sequence
        )
    )

    # -------------------------------------------------------------------------
    # WHOLE TRAJECTORY
    # -------------------------------------------------------------------------

    original_logits = (
        trajectory_classifier(
            original.unsqueeze(0)
            .to(DEVICE)
        )
    )

    generated_logits = (
        trajectory_classifier(
            generated.unsqueeze(0)
            .to(DEVICE)
        )
    )

    original_probs = F.softmax(
        original_logits,
        dim=1
    )[0]

    generated_probs = F.softmax(
        generated_logits,
        dim=1
    )[0]

    trajectory_score = float(
        original_probs[0]
        - generated_probs[0]
    )

    # -------------------------------------------------------------------------
    # POINT-WISE
    # -------------------------------------------------------------------------

    pointwise_scores = {}

    for prefix_index, classifier in (
        pointwise_classifiers.items()
    ):

        original_point = (
            original[prefix_index]
            .unsqueeze(0)
            .to(DEVICE)
        )

        generated_point = (
            generated[prefix_index]
            .unsqueeze(0)
            .to(DEVICE)
        )

        p_orig_original = F.softmax(
            classifier(original_point),
            dim=1
        )[0, 0]

        p_orig_generated = F.softmax(
            classifier(generated_point),
            dim=1
        )[0, 0]

        pointwise_scores[
            prefix_index
        ] = float(
            p_orig_original
            - p_orig_generated
        )

    distances = torch.norm(
        original - generated,
        dim=1
    ).numpy()

    return {
        "trajectory_score":
            trajectory_score,

        "pointwise_scores":
            pointwise_scores,

        "distances":
            distances,

        "original":
            original,

        "generated":
            generated,
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

    sorted_scores = scores[order]
    sorted_labels = labels[order]

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
            (i + 1 + j) / 2.0
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

        boot.append(
            auc_score(
                pos,
                neg
            )
        )

    return (
        observed,
        float(
            np.percentile(
                boot,
                2.5
            )
        ),
        float(
            np.percentile(
                boot,
                97.5
            )
        )
    )


# =============================================================================
# BOOTSTRAP DIFFERENCE IN AUC
# =============================================================================
#
# Important:
#
# The same evaluation texts are used for both methods.
#
# Therefore the trajectory-vs-pointwise comparison is paired.
#
# We bootstrap evaluation texts jointly rather than bootstrapping the
# two AUCs independently.
#
# =============================================================================

def bootstrap_auc_difference(
    trajectory_positive,
    trajectory_negative,
    point_positive,
    point_negative,
    n_bootstrap=N_BOOTSTRAP,
    seed=54321
):

    trajectory_positive = np.asarray(
        trajectory_positive
    )

    trajectory_negative = np.asarray(
        trajectory_negative
    )

    point_positive = np.asarray(
        point_positive
    )

    point_negative = np.asarray(
        point_negative
    )

    rng = np.random.default_rng(
        seed
    )

    n_pos = len(
        trajectory_positive
    )

    n_neg = len(
        trajectory_negative
    )

    observed = (
        auc_score(
            trajectory_positive,
            trajectory_negative
        )
        -
        auc_score(
            point_positive,
            point_negative
        )
    )

    boot = []

    for _ in range(
        n_bootstrap
    ):

        pos_idx = rng.integers(
            0,
            n_pos,
            size=n_pos
        )

        neg_idx = rng.integers(
            0,
            n_neg,
            size=n_neg
        )

        trajectory_auc = auc_score(
            trajectory_positive[pos_idx],
            trajectory_negative[neg_idx]
        )

        point_auc = auc_score(
            point_positive[pos_idx],
            point_negative[neg_idx]
        )

        boot.append(
            trajectory_auc
            - point_auc
        )

    return (
        float(observed),
        float(
            np.percentile(
                boot,
                2.5
            )
        ),
        float(
            np.percentile(
                boot,
                97.5
            )
        )
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
    trajectory_classifier,
    pointwise_classifiers,
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
            trajectory_classifier,
            pointwise_classifiers,
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

    trajectory_scores = np.array(
        [
            x["trajectory_score"]
            for x in results
        ]
    )

    pointwise_scores = {}

    for prefix_index in range(K):

        pointwise_scores[
            prefix_index
        ] = np.array(
            [
                x["pointwise_scores"][
                    prefix_index
                ]
                for x in results
            ]
        )

    distances = np.stack(
        [
            x["distances"]
            for x in results
        ],
        axis=0
    )

    return {
        "results":
            results,

        "trajectory_scores":
            trajectory_scores,

        "pointwise_scores":
            pointwise_scores,

        "distances":
            distances,
    }


# =============================================================================
# TRAJECTORY SANITY CHECK
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
    # CORPUS
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

    eval_ordinary_texts = (
        ordinary_texts[
            :N_EVAL_ORDINARY
        ]
    )

    eval_strong_texts = strong_texts

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
        encode_text(x, vocab)
        for x in ordinary_texts
    ]

    strong_sequences = [
        encode_text(x, vocab)
        for x in strong_texts
    ]

    classifier_train_sequences = [
        encode_text(x, vocab)
        for x in classifier_train_texts
    ]

    classifier_val_sequences = [
        encode_text(x, vocab)
        for x in classifier_val_texts
    ]

    eval_ordinary_sequences = [
        encode_text(x, vocab)
        for x in eval_ordinary_texts
    ]

    eval_strong_sequences = [
        encode_text(x, vocab)
        for x in eval_strong_texts
    ]

    nonmember_sequences = [
        encode_text(x, vocab)
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
        f"ordinary members = "
        f"{len(ordinary_sequences)}"
    )

    print(
        f"strongly memorized = "
        f"{len(strong_sequences)}"
    )

    print(
        f"repetitions per strong text = "
        f"{STRONG_REPEAT}"
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
    # SANITY CHECK
    # =========================================================================

    trajectory_sanity_check(
        lm,
        classifier_train_sequences[0]
    )

    # =========================================================================
    # STEP 2: BUILD SHARED CLASSIFIER DATA
    # =========================================================================

    print()
    print("=" * 110)
    print(
        "STEP 2: BUILD SHARED "
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
        f"Training tensor = "
        f"{tuple(X_train.shape)}"
    )

    print(
        f"Validation tensor = "
        f"{tuple(X_val.shape)}"
    )

    # =========================================================================
    # STEP 3: TRAIN WHOLE-TRAJECTORY CLASSIFIER
    # =========================================================================

    trajectory_classifier = (
        train_classifier(
            X_train,
            y_train,
            X_val,
            y_val
        )
    )

    # =========================================================================
    # STEP 4: TRAIN POINT-WISE CLASSIFIERS
    #
    # One classifier per prefix.
    #
    # This means the point-wise baseline gets to choose the best
    # representation separately, rather than being disadvantaged by
    # a particular prefix.
    # =========================================================================

    pointwise_classifiers = {}

    for prefix_index, prefix in enumerate(
        PREFIX_LENGTHS
    ):

        pointwise_classifiers[
            prefix_index
        ] = train_pointwise_classifier(
            X_train,
            y_train,
            X_val,
            y_val,
            prefix_index
        )

    # =========================================================================
    # STEP 5: EVALUATION
    # =========================================================================

    strong_eval = evaluate_group(
        "STRONGLY MEMORIZED",
        lm,
        trajectory_classifier,
        pointwise_classifiers,
        eval_strong_sequences
    )

    ordinary_eval = evaluate_group(
        "ORDINARY MEMBERS",
        lm,
        trajectory_classifier,
        pointwise_classifiers,
        eval_ordinary_sequences
    )

    nonmember_eval = evaluate_group(
        "NON-MEMBERS",
        lm,
        trajectory_classifier,
        pointwise_classifiers,
        nonmember_sequences
    )

    # =========================================================================
    # TRAJECTORY AUCs
    # =========================================================================

    strong_scores = (
        strong_eval[
            "trajectory_scores"
        ]
    )

    ordinary_scores = (
        ordinary_eval[
            "trajectory_scores"
        ]
    )

    nonmember_scores = (
        nonmember_eval[
            "trajectory_scores"
        ]
    )

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
    # POINT-WISE AUCs
    # =========================================================================

    pointwise_auc = {}
    pointwise_ci = {}

    for prefix_index, prefix in enumerate(
        PREFIX_LENGTHS
    ):

        strong_point = (
            strong_eval[
                "pointwise_scores"
            ][prefix_index]
        )

        ordinary_point = (
            ordinary_eval[
                "pointwise_scores"
            ][prefix_index]
        )

        nonmember_point = (
            nonmember_eval[
                "pointwise_scores"
            ][prefix_index]
        )

        auc, low, high = (
            bootstrap_auc_ci(
                strong_point,
                nonmember_point,
                seed=40000
                + repeat
                + prefix
            )
        )

        pointwise_auc[
            prefix_index
        ] = auc

        pointwise_ci[
            prefix_index
        ] = (
            low,
            high
        )

    # =========================================================================
    # TRAJECTORY ADVANTAGE
    # =========================================================================
    #
    # Primary:
    #
    # trajectory AUC
    #       -
    # point-wise AUC at prefix 32
    #
    # Secondary:
    #
    # trajectory AUC
    #       -
    # best point-wise AUC
    #
    # =========================================================================

    primary_point_auc = (
        pointwise_auc[
            SINGLE_POINT_INDEX
        ]
    )

    best_point_index = max(
        pointwise_auc,
        key=pointwise_auc.get
    )

    best_point_auc = (
        pointwise_auc[
            best_point_index
        ]
    )

    primary_advantage = (
        strong_auc
        - primary_point_auc
    )

    best_point_advantage = (
        strong_auc
        - best_point_auc
    )

    # Paired bootstrap difference.
    primary_point_scores_strong = (
        strong_eval[
            "pointwise_scores"
        ][SINGLE_POINT_INDEX]
    )

    primary_point_scores_nonmember = (
        nonmember_eval[
            "pointwise_scores"
        ][SINGLE_POINT_INDEX]
    )

    (
        advantage,
        advantage_low,
        advantage_high
    ) = bootstrap_auc_difference(
        strong_scores,
        nonmember_scores,
        primary_point_scores_strong,
        primary_point_scores_nonmember,
        seed=50000 + repeat
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
    # PREFIX DISTANCE DIAGNOSTIC
    # =========================================================================

    distance_auc_strong = []
    distance_auc_ordinary = []

    for j, prefix in enumerate(
        PREFIX_LENGTHS
    ):

        strong_dist = (
            strong_eval[
                "distances"
            ][:, j]
        )

        ordinary_dist = (
            ordinary_eval[
                "distances"
            ][:, j]
        )

        nonmember_dist = (
            nonmember_eval[
                "distances"
            ][:, j]
        )

        distance_auc_strong.append(
            auc_score(
                strong_dist,
                nonmember_dist
            )
        )

        distance_auc_ordinary.append(
            auc_score(
                ordinary_dist,
                nonmember_dist
            )
        )

    # =========================================================================
    # VALIDATION ACCURACY
    # =========================================================================

    trajectory_classifier.eval()

    with torch.no_grad():

        val_logits = (
            trajectory_classifier(
                X_val.to(DEVICE)
            )
        )

        val_accuracy = float(
            (
                val_logits.argmax(dim=1)
                == y_val.to(DEVICE)
            )
            .float()
            .mean()
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
        "WHOLE TRAJECTORY"
    )

    print(
        f"Strongly memorized vs non-member: "
        f"AUC = {strong_auc:.4f}"
    )

    print(
        f"Ordinary member vs non-member: "
        f"AUC = {ordinary_auc:.4f}"
    )

    print(
        f"Strongly memorized vs ordinary: "
        f"AUC = {strong_vs_ordinary_auc:.4f}"
    )

    print()
    print(
        "POINT-WISE BASELINES"
    )

    for prefix_index, prefix in enumerate(
        PREFIX_LENGTHS
    ):

        low, high = (
            pointwise_ci[
                prefix_index
            ]
        )

        marker = (
            "  <-- PRIMARY"
            if prefix == SINGLE_POINT_PREFIX
            else ""
        )

        print(
            f"Prefix {prefix:02d}: "
            f"AUC = "
            f"{pointwise_auc[prefix_index]:.4f} "
            f"95% CI = "
            f"[{low:.4f}, {high:.4f}]"
            f"{marker}"
        )

    print()
    print(
        "TRAJECTORY ADVANTAGE"
    )

    print(
        f"Trajectory AUC = "
        f"{strong_auc:.4f}"
    )

    print(
        f"Point-wise AUC "
        f"(prefix {SINGLE_POINT_PREFIX}) = "
        f"{primary_point_auc:.4f}"
    )

    print(
        f"Trajectory - point-wise = "
        f"{primary_advantage:+.4f}"
    )

    print(
        f"Paired bootstrap 95% CI = "
        f"[{advantage_low:+.4f}, "
        f"{advantage_high:+.4f}]"
    )

    print()

    print(
        f"Best point-wise prefix = "
        f"{PREFIX_LENGTHS[best_point_index]}"
    )

    print(
        f"Best point-wise AUC = "
        f"{best_point_auc:.4f}"
    )

    print(
        f"Trajectory - best point-wise = "
        f"{best_point_advantage:+.4f}"
    )

    # =========================================================================
    # SCORE DISTRIBUTIONS
    # =========================================================================

    print()
    print(
        "TRAJECTORY SCORE DISTRIBUTIONS"
    )

    print(
        f"Strongly memorized: "
        f"mean={strong_scores.mean():.4f}, "
        f"SD={strong_scores.std(ddof=1):.4f}"
    )

    print(
        f"Ordinary members: "
        f"mean={ordinary_scores.mean():.4f}, "
        f"SD={ordinary_scores.std(ddof=1):.4f}"
    )

    print(
        f"Non-members: "
        f"mean={nonmember_scores.mean():.4f}, "
        f"SD={nonmember_scores.std(ddof=1):.4f}"
    )

    # =========================================================================
    # EFFECT SIZES
    # =========================================================================

    print()
    print(
        "EFFECT SIZES"
    )

    print(
        f"Strong vs non-member: "
        f"d={d_strong:.4f}"
    )

    print(
        f"Ordinary vs non-member: "
        f"d={d_ordinary:.4f}"
    )

    print(
        f"Strong vs ordinary: "
        f"d={d_strong_ordinary:.4f}"
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
            f"distance strong-vs-nonmember "
            f"AUC = "
            f"{distance_auc_strong[j]:.4f} | "
            f"ordinary-vs-nonmember "
            f"AUC = "
            f"{distance_auc_ordinary[j]:.4f}"
        )

    print()
    print(
        f"FINAL TRAJECTORY CLASSIFIER "
        f"VALIDATION ACCURACY = "
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
The primary comparison is between:

    WHOLE TRAJECTORY
        vs
    SINGLE POINT

Both classifiers are trained on exactly the same original-vs-generated
classifier dataset.

The only structural difference is the information available to the
classifier:

    trajectory:
        all four prefix representations jointly

    point-wise:
        one prefix representation only

The primary point-wise baseline is prefix 32.

A stronger result for the proposed method is therefore:

    trajectory AUC
        >
    point-wise AUC

especially if the trajectory AUC exceeds the BEST point-wise AUC across
all tested prefixes.

The paired bootstrap confidence interval for the AUC difference provides
an additional assessment of whether the observed advantage is robust.

This comparison directly tests whether the trajectory contains
membership information that is not available from a single
prefix-specific representation.
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

    result = {

        "repeat":
            repeat,

        "trajectory_auc_strong":
            strong_auc,

        "trajectory_auc_ordinary":
            ordinary_auc,

        "trajectory_auc_strong_vs_ordinary":
            strong_vs_ordinary_auc,

        "trajectory_auc_strong_low":
            strong_low,

        "trajectory_auc_strong_high":
            strong_high,

        "trajectory_advantage_prefix32":
            primary_advantage,

        "trajectory_advantage_prefix32_low":
            advantage_low,

        "trajectory_advantage_prefix32_high":
            advantage_high,

        "best_pointwise_prefix":
            PREFIX_LENGTHS[
                best_point_index
            ],

        "best_pointwise_auc":
            best_point_auc,

        "trajectory_advantage_best_pointwise":
            best_point_advantage,

        "d_strong":
            d_strong,

        "d_ordinary":
            d_ordinary,

        "d_strong_vs_ordinary":
            d_strong_ordinary,

        "val_accuracy":
            val_accuracy,

        "strong_mean":
            strong_scores.mean(),

        "strong_sd":
            strong_scores.std(ddof=1),

        "ordinary_mean":
            ordinary_scores.mean(),

        "ordinary_sd":
            ordinary_scores.std(ddof=1),

        "nonmember_mean":
            nonmember_scores.mean(),

        "nonmember_sd":
            nonmember_scores.std(ddof=1),
    }

    # Add point-wise AUCs
    for prefix_index, prefix in enumerate(
        PREFIX_LENGTHS
    ):

        result[
            f"pointwise_auc_{prefix}"
        ] = pointwise_auc[
            prefix_index
        ]

    return result


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(
    results,
    filename
):

    fieldnames = [
        "repeat",

        "trajectory_auc_strong",
        "trajectory_auc_ordinary",
        "trajectory_auc_strong_vs_ordinary",

        "pointwise_auc_8",
        "pointwise_auc_16",
        "pointwise_auc_24",
        "pointwise_auc_32",

        "trajectory_advantage_prefix32",
        "trajectory_advantage_prefix32_low",
        "trajectory_advantage_prefix32_high",

        "best_pointwise_prefix",
        "best_pointwise_auc",

        "trajectory_advantage_best_pointwise",

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

            writer.writerow(
                {
                    key: result[key]
                    for key in fieldnames
                }
            )

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
    # FINAL SUMMARY
    # =========================================================================

    print()
    print("#" * 110)
    print(
        "FINAL SUMMARY"
    )
    print("#" * 110)

    # -------------------------------------------------------------------------
    # TRAJECTORY
    # -------------------------------------------------------------------------

    trajectory_auc = np.array(
        [
            r["trajectory_auc_strong"]
            for r in all_results
        ]
    )

    print()
    print(
        "WHOLE-TRAJECTORY:"
    )

    print(
        f"  Strong vs non-member AUC = "
        f"{trajectory_auc.mean():.4f} "
        f"+/- "
        f"{trajectory_auc.std(ddof=1):.4f}"
    )

    # -------------------------------------------------------------------------
    # POINT-WISE
    # -------------------------------------------------------------------------

    print()
    print(
        "POINT-WISE BASELINES:"
    )

    pointwise_summary = {}

    for prefix in PREFIX_LENGTHS:

        values = np.array(
            [
                r[
                    f"pointwise_auc_{prefix}"
                ]
                for r in all_results
            ]
        )

        pointwise_summary[
            prefix
        ] = values

        marker = (
            " <-- PRIMARY"
            if prefix == SINGLE_POINT_PREFIX
            else ""
        )

        print(
            f"  Prefix {prefix:02d}: "
            f"{values.mean():.4f} "
            f"+/- "
            f"{values.std(ddof=1):.4f}"
            f"{marker}"
        )

    # -------------------------------------------------------------------------
    # TRAJECTORY ADVANTAGE
    # -------------------------------------------------------------------------

    primary_advantages = np.array(
        [
            r[
                "trajectory_advantage_prefix32"
            ]
            for r in all_results
        ]
    )

    best_advantages = np.array(
        [
            r[
                "trajectory_advantage_best_pointwise"
            ]
            for r in all_results
        ]
    )

    print()
    print(
        "TRAJECTORY ADVANTAGE:"
    )

    print(
        f"  vs prefix {SINGLE_POINT_PREFIX}: "
        f"{primary_advantages.mean():+.4f} "
        f"+/- "
        f"{primary_advantages.std(ddof=1):.4f}"
    )

    print(
        f"  vs best point-wise: "
        f"{best_advantages.mean():+.4f} "
        f"+/- "
        f"{best_advantages.std(ddof=1):.4f}"
    )

    # -------------------------------------------------------------------------
    # BEST POINT-WISE PREFIX
    # -------------------------------------------------------------------------

    best_prefixes = [
        r["best_pointwise_prefix"]
        for r in all_results
    ]

    print()
    print(
        "BEST POINT-WISE PREFIX PER REPEAT:"
    )

    print(
        "  "
        + ", ".join(
            str(x)
            for x in best_prefixes
        )
    )

    # -------------------------------------------------------------------------
    # EFFECT SIZES
    # -------------------------------------------------------------------------

    strong_d = np.array(
        [
            r["d_strong"]
            for r in all_results
        ]
    )

    ordinary_d = np.array(
        [
            r["d_ordinary"]
            for r in all_results
        ]
    )

    print()
    print(
        "EFFECT SIZES:"
    )

    print(
        f"  Strong vs non-member: "
        f"{strong_d.mean():.4f} "
        f"+/- "
        f"{strong_d.std(ddof=1):.4f}"
    )

    print(
        f"  Ordinary vs non-member: "
        f"{ordinary_d.mean():.4f} "
        f"+/- "
        f"{ordinary_d.std(ddof=1):.4f}"
    )

    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------

    validation = np.array(
        [
            r["val_accuracy"]
            for r in all_results
        ]
    )

    print()
    print(
        "TRAJECTORY CLASSIFIER VALIDATION:"
    )

    print(
        f"  accuracy = "
        f"{validation.mean():.4f} "
        f"+/- "
        f"{validation.std(ddof=1):.4f}"
    )

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    save_results(
        all_results,
        RESULTS_CSV
    )

    # -------------------------------------------------------------------------
    # FINAL INTERPRETATION
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "FINAL INTERPRETATION"
    )
    print("=" * 110)

    print(
        f"""
Primary proposed method:
    Whole-trajectory classifier

Primary baseline:
    Point-wise classifier at prefix {SINGLE_POINT_PREFIX}

Secondary baseline analysis:
    Point-wise classifiers at all prefixes:
        {PREFIX_LENGTHS}

The critical quantity is:

    trajectory AUC - point-wise AUC

A positive value indicates an advantage for the trajectory representation.

The strongest evidence for the proposed method would be a positive
trajectory advantage over BOTH:

    1. the pre-specified prefix-{SINGLE_POINT_PREFIX} baseline
    2. the best-performing point-wise prefix

This establishes that the benefit is not simply caused by choosing
a particularly informative single prefix.
"""
    )

    print()
    print(
        "DONE"
    )