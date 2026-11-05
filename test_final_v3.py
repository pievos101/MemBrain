# =============================================================================
# WHOLE-TRAJECTORY SUFFIX-EMBEDDING MEMBERSHIP BENCHMARK
#
# PAPER-READY VERSION
# =============================================================================
#
# CORE QUESTION
# -----------------------------------------------------------------------------
#
# Can a whole-trajectory suffix-embedding classifier distinguish:
#
#     1. strongly memorized members
#     2. ordinary members
#     3. non-members
#
# The language model is trained ONLY on:
#
#     - ordinary members
#     - strongly memorized members
#
# The classifier is trained on SEPARATE texts.
#
# The classifier learns ONLY:
#
#     class 0 = original suffix trajectory
#     class 1 = generated suffix trajectory
#
# It NEVER receives membership labels.
#
# For every text:
#
#     prefix_1 -> suffix embedding
#     prefix_2 -> suffix embedding
#     ...
#     prefix_K -> suffix embedding
#
# This produces:
#
#     trajectory = K x D
#
# The CNN receives the COMPLETE trajectory.
#
# Membership inference is performed ONLY after classifier training.
#
# =============================================================================


import random
import time
import csv

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
# DATA SIZE
# =============================================================================
#
# LM TRAINING:
#
# Ordinary members:
#     seen exactly once
#
# Strong members:
#     deliberately repeated STRONG_REPEAT times
#
# Classifier training texts:
#     separate texts, never used for LM training
#
# Classifier validation texts:
#     separate texts, never used for LM training
#
# Evaluation:
#     held-out ordinary members
#     strongly memorized members
#     non-members
#
# =============================================================================

N_ORDINARY_MEMBERS = 80

N_STRONG_MEMBERS = 10

N_CLASSIFIER_TRAIN = 30
N_CLASSIFIER_VAL = 10

N_EVAL_ORDINARY = 40

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
# WHOLE-TRAJECTORY PREFIXES
# =============================================================================

PREFIX_LENGTHS = [
    8,
    16,
    24,
    32,
]


# =============================================================================
# GENERATION
# =============================================================================

GEN_LEN = 8

TEMPERATURE = 1.15


# =============================================================================
# STRONG MEMORIZATION
# =============================================================================
#
# Deliberately artificial positive control.
#
# Each strong-member document occurs STRONG_REPEAT times during LM training.
#
# =============================================================================

STRONG_REPEAT = 15


# =============================================================================
# TRAJECTORY CLASSIFIER
# =============================================================================

CNN_CHANNELS = 24

MLP_HIDDEN = 48

MLP_EPOCHS = 50

MLP_LR = 2e-3

WEIGHT_DECAY = 1e-4

DROPOUT = 0.10

BATCH_SIZE = 32


# =============================================================================
# REPEATS
# =============================================================================
#
# For publication-quality estimates use >= 5.
#
# Increase to 10 if runtime permits.
#
# =============================================================================

N_REPEATS = 5


# =============================================================================
# OUTPUT
# =============================================================================

RESULTS_CSV = (
    "whole_trajectory_membership_results.csv"
)


# =============================================================================
# RANDOM SEED
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
# SYNTHETIC MEMBER TEXT
# =============================================================================

def make_document(
    i,
    length=SEQ_LEN,
):

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
# SYNTHETIC NON-MEMBER TEXT
#
# Same vocabulary/domain.
# Independently generated.
#
# =============================================================================

def make_nonmember_document(
    i,
    length=SEQ_LEN,
):

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
        key=lambda x: (-x[1], x[0]),
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
# ENCODE TEXT
# =============================================================================

def encode_text(
    text,
    vocab,
):

    ids = [
        vocab.get(
            token,
            vocab["<UNK>"],
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
        dtype=torch.long,
    )


# =============================================================================
# LANGUAGE MODEL
# =============================================================================

class TinyLM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
        )

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.output = nn.Linear(
            hidden_dim,
            vocab_size,
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
# TRAIN LANGUAGE MODEL
# =============================================================================

def train_lm(
    model,
    ordinary_sequences,
    strong_sequences,
    vocab_size,
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LM_LR,
    )

    model.train()

    training_sequences = []

    # ------------------------------------------------------------
    # Ordinary members appear once.
    # ------------------------------------------------------------

    for seq in ordinary_sequences:

        training_sequences.append(
            seq
        )

    # ------------------------------------------------------------
    # Strong members are deliberately repeated.
    # ------------------------------------------------------------

    for seq in strong_sequences:

        for _ in range(
            STRONG_REPEAT
        ):

            training_sequences.append(
                seq
            )

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

    for epoch in range(
        1,
        LM_EPOCHS + 1,
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
                    vocab_size,
                ),
                y.reshape(-1),
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            losses.append(
                loss.item()
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
# ROBUST SUFFIX EMBEDDING
# =============================================================================
#
# THIS IS THE IMPORTANT FIX FOR YOUR ERROR.
#
# The returned representation is ALWAYS:
#
#       [HIDDEN_DIM]
#
# We deliberately do NOT use squeeze().
#
# =============================================================================

@torch.no_grad()
def get_suffix_embedding(
    model,
    sequence,
    prefix_length,
):

    model.eval()

    # ------------------------------------------------------------
    # Force sequence to be exactly one-dimensional.
    # ------------------------------------------------------------

    sequence = (
        sequence
        .to(DEVICE)
        .long()
        .reshape(-1)
    )

    if sequence.numel() < 2:

        return torch.zeros(
            HIDDEN_DIM,
            dtype=torch.float32,
        )

    # ------------------------------------------------------------
    # Valid prefix.
    # ------------------------------------------------------------

    prefix_length = int(
        prefix_length
    )

    prefix_length = max(
        1,
        min(
            prefix_length,
            sequence.numel() - 1,
        ),
    )

    # ------------------------------------------------------------
    # LM input.
    #
    # [1, L-1]
    # ------------------------------------------------------------

    x = (
        sequence[:-1]
        .unsqueeze(0)
    )

    embeddings = model.embedding(
        x
    )

    # [1, L-1, HIDDEN_DIM]
    hidden, _ = model.gru(
        embeddings
    )

    # Remove ONLY batch dimension.
    #
    # [L-1, HIDDEN_DIM]
    #
    # Do NOT use squeeze().
    # ------------------------------------------------------------

    hidden = hidden[0]

    if hidden.ndim != 2:

        raise RuntimeError(
            "LM hidden representation must "
            "be 2-D after removing batch dimension. "
            f"Got {tuple(hidden.shape)}"
        )

    if hidden.shape[1] != HIDDEN_DIM:

        raise RuntimeError(
            "Unexpected hidden dimension. "
            f"Expected {HIDDEN_DIM}, "
            f"got {hidden.shape[1]}"
        )

    # ------------------------------------------------------------
    # Suffix begins immediately after prefix.
    #
    # Because hidden corresponds to sequence[:-1],
    # the corresponding hidden index is prefix_length - 1.
    # ------------------------------------------------------------

    suffix_start = (
        prefix_length - 1
    )

    suffix_states = hidden[
        suffix_start:
    ]

    if suffix_states.ndim != 2:

        raise RuntimeError(
            "Suffix states must be 2-D. "
            f"Got {tuple(suffix_states.shape)}"
        )

    if suffix_states.shape[0] == 0:

        return torch.zeros(
            HIDDEN_DIM,
            dtype=torch.float32,
        )

    # ------------------------------------------------------------
    # Pool over TIME ONLY.
    #
    # [suffix_length, HIDDEN_DIM]
    #
    #       mean(dim=0)
    #
    # becomes:
    #
    # [HIDDEN_DIM]
    # ------------------------------------------------------------

    representation = (
        suffix_states.mean(
            dim=0
        )
    )

    # ------------------------------------------------------------
    # HARD FINAL CHECK
    # ------------------------------------------------------------

    if representation.ndim != 1:

        raise RuntimeError(
            "Suffix embedding must be 1-D. "
            f"Got shape "
            f"{tuple(representation.shape)}"
        )

    if representation.shape[0] != HIDDEN_DIM:

        raise RuntimeError(
            "Wrong suffix embedding dimension. "
            f"Expected {HIDDEN_DIM}, "
            f"got {representation.shape[0]}"
        )

    return (
        representation
        .detach()
        .cpu()
        .float()
    )


# =============================================================================
# GENERATE SUFFIX
# =============================================================================

@torch.no_grad()
def generate_suffix(
    model,
    prefix,
    gen_len,
    temperature,
):

    model.eval()

    generated = (
        prefix
        .clone()
        .to(DEVICE)
        .long()
        .reshape(-1)
    )

    for _ in range(
        gen_len
    ):

        logits = model(
            generated.unsqueeze(0)
        )

        next_logits = (
            logits[0, -1]
            / temperature
        )

        probs = F.softmax(
            next_logits,
            dim=-1,
        )

        next_token = torch.multinomial(
            probs,
            num_samples=1,
        )

        generated = torch.cat(
            [
                generated,
                next_token,
            ],
            dim=0,
        )

    return generated


# =============================================================================
# BUILD ORIGINAL TRAJECTORY
# =============================================================================

@torch.no_grad()
def build_original_trajectory(
    model,
    sequence,
):

    rows = []

    for prefix_length in PREFIX_LENGTHS:

        prefix_length = min(
            int(prefix_length),
            len(sequence) - 1,
        )

        embedding = (
            get_suffix_embedding(
                model,
                sequence,
                prefix_length,
            )
        )

        # --------------------------------------------------------
        # HARD CHECK
        # --------------------------------------------------------

        if embedding.ndim != 1:

            raise RuntimeError(
                "Original trajectory row "
                "must be 1-D. "
                f"Got {tuple(embedding.shape)}"
            )

        if embedding.shape[0] != HIDDEN_DIM:

            raise RuntimeError(
                "Original trajectory row "
                "has wrong dimension. "
                f"Expected {HIDDEN_DIM}, "
                f"got {embedding.shape[0]}"
            )

        rows.append(
            embedding
        )

    trajectory = torch.stack(
        rows,
        dim=0,
    )

    expected_shape = (
        len(PREFIX_LENGTHS),
        HIDDEN_DIM,
    )

    if tuple(
        trajectory.shape
    ) != expected_shape:

        raise RuntimeError(
            "Original trajectory has "
            "wrong shape. "
            f"Expected {expected_shape}, "
            f"got {tuple(trajectory.shape)}"
        )

    return trajectory


# =============================================================================
# BUILD GENERATED TRAJECTORY
# =============================================================================

@torch.no_grad()
def build_generated_trajectory(
    model,
    sequence,
):

    rows = []

    for prefix_length in PREFIX_LENGTHS:

        prefix_length = min(
            int(prefix_length),
            len(sequence) - 1,
        )

        prefix = sequence[
            :prefix_length
        ]

        generated_sequence = (
            generate_suffix(
                model,
                prefix,
                GEN_LEN,
                TEMPERATURE,
            )
        )

        embedding = (
            get_suffix_embedding(
                model,
                generated_sequence,
                prefix_length,
            )
        )

        # --------------------------------------------------------
        # HARD CHECK
        # --------------------------------------------------------

        if embedding.ndim != 1:

            raise RuntimeError(
                "Generated trajectory row "
                "must be 1-D. "
                f"Got {tuple(embedding.shape)}"
            )

        if embedding.shape[0] != HIDDEN_DIM:

            raise RuntimeError(
                "Generated trajectory row "
                "has wrong dimension. "
                f"Expected {HIDDEN_DIM}, "
                f"got {embedding.shape[0]}"
            )

        rows.append(
            embedding
        )

    trajectory = torch.stack(
        rows,
        dim=0,
    )

    expected_shape = (
        len(PREFIX_LENGTHS),
        HIDDEN_DIM,
    )

    if tuple(
        trajectory.shape
    ) != expected_shape:

        raise RuntimeError(
            "Generated trajectory has "
            "wrong shape. "
            f"Expected {expected_shape}, "
            f"got {tuple(trajectory.shape)}"
        )

    return trajectory


# =============================================================================
# BUILD TRAJECTORY PAIR
# =============================================================================

@torch.no_grad()
def build_trajectory_pair(
    model,
    sequence,
):

    original = (
        build_original_trajectory(
            model,
            sequence,
        )
    )

    generated = (
        build_generated_trajectory(
            model,
            sequence,
        )
    )

    expected_shape = (
        len(PREFIX_LENGTHS),
        HIDDEN_DIM,
    )

    if tuple(
        original.shape
    ) != expected_shape:

        raise RuntimeError(
            "Original trajectory shape "
            f"{tuple(original.shape)} "
            f"!= {expected_shape}"
        )

    if tuple(
        generated.shape
    ) != expected_shape:

        raise RuntimeError(
            "Generated trajectory shape "
            f"{tuple(generated.shape)} "
            f"!= {expected_shape}"
        )

    return original, generated


# =============================================================================
# TRAJECTORY CNN
# =============================================================================

class TrajectoryCNN(nn.Module):

    def __init__(
        self,
        embedding_dim,
    ):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=CNN_CHANNELS,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv1d(
            in_channels=CNN_CHANNELS,
            out_channels=CNN_CHANNELS,
            kernel_size=3,
            padding=1,
        )

        self.pool = (
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                CNN_CHANNELS,
                MLP_HIDDEN,
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                MLP_HIDDEN,
                2,
            ),
        )

    def forward(
        self,
        x,
    ):

        # ========================================================
        # x:
        #
        #       [batch, K, D]
        #
        # Conv1D expects:
        #
        #       [batch, D, K]
        # ========================================================

        if x.ndim != 3:

            raise RuntimeError(
                "Trajectory CNN expects "
                "[batch, K, D]. "
                f"Got {tuple(x.shape)}"
            )

        x = x.transpose(
            1,
            2,
        )

        x = F.relu(
            self.conv1(x)
        )

        x = F.relu(
            self.conv2(x)
        )

        x = self.pool(x)

        # Remove ONLY final singleton spatial dimension.
        x = x[:, :, 0]

        return self.classifier(
            x
        )


# =============================================================================
# BUILD CLASSIFIER DATASET
# =============================================================================
#
# IMPORTANT:
#
# For every classifier-training text:
#
#       original trajectory -> label 0
#       generated trajectory -> label 1
#
# NO membership information is provided.
#
# =============================================================================

@torch.no_grad()
def build_trajectory_classifier_dataset(
    model,
    sequences,
):

    X = []

    y = []

    print()

    print("=" * 110)
    print(
        "BUILD WHOLE-TRAJECTORY "
        "CLASSIFIER DATA"
    )
    print("=" * 110)

    expected_shape = (
        len(PREFIX_LENGTHS),
        HIDDEN_DIM,
    )

    for i, sequence in enumerate(
        sequences
    ):

        original, generated = (
            build_trajectory_pair(
                model,
                sequence,
            )
        )

        if tuple(
            original.shape
        ) != expected_shape:

            raise RuntimeError(
                f"Original trajectory "
                f"shape error: "
                f"{tuple(original.shape)}"
            )

        if tuple(
            generated.shape
        ) != expected_shape:

            raise RuntimeError(
                f"Generated trajectory "
                f"shape error: "
                f"{tuple(generated.shape)}"
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
            f"{i + 1}/{len(sequences)}",
            end="",
            flush=True,
        )

    print()

    X = torch.stack(
        X,
        dim=0,
    )

    y = torch.tensor(
        y,
        dtype=torch.long,
    )

    expected_dataset_shape = (
        2 * len(sequences),
        len(PREFIX_LENGTHS),
        HIDDEN_DIM,
    )

    if tuple(
        X.shape
    ) != expected_dataset_shape:

        raise RuntimeError(
            "Classifier dataset has "
            "wrong shape. "
            f"Expected "
            f"{expected_dataset_shape}, "
            f"got {tuple(X.shape)}"
        )

    return X, y


# =============================================================================
# TRAIN CLASSIFIER
# =============================================================================

def train_classifier(
    X,
    y,
    X_val,
    y_val,
):

    print()

    print("=" * 110)
    print(
        "TRAIN WHOLE-TRAJECTORY "
        "ORIGINAL-vs-GENERATED CLASSIFIER"
    )
    print("=" * 110)

    print(
        f"Training matrices = {len(X)}"
    )

    print(
        f"Validation matrices = {len(X_val)}"
    )

    print(
        f"Trajectory shape = "
        f"{tuple(X.shape[1:])}"
    )

    model = TrajectoryCNN(
        embedding_dim=HIDDEN_DIM,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=MLP_LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss()

    X = X.to(DEVICE)
    y = y.to(DEVICE)

    X_val = X_val.to(DEVICE)
    y_val = y_val.to(DEVICE)

    best_state = None

    best_val_loss = float(
        "inf"
    )

    model.train()

    for epoch in range(
        1,
        MLP_EPOCHS + 1,
    ):

        permutation = torch.randperm(
            len(X),
            device=DEVICE,
        )

        total_loss = 0.0

        correct = 0

        for start in range(
            0,
            len(X),
            BATCH_SIZE,
        ):

            idx = permutation[
                start:
                start + BATCH_SIZE
            ]

            xb = X[idx]

            yb = y[idx]

            logits = model(xb)

            loss = criterion(
                logits,
                yb,
            )

            optimizer.zero_grad()

            loss.backward()

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

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_logits = model(
                X_val
            )

            val_loss = criterion(
                val_logits,
                y_val,
            )

            val_pred = (
                val_logits.argmax(
                    dim=1
                )
            )

            val_accuracy = float(
                (
                    val_pred == y_val
                ).float().mean()
            )

        model.train()

        train_loss = (
            total_loss / len(X)
        )

        train_accuracy = (
            correct / len(X)
        )

        if (
            val_loss.item()
            < best_val_loss
        ):

            best_val_loss = (
                val_loss.item()
            )

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == MLP_EPOCHS
        ):

            print(
                f"Epoch "
                f"{epoch:03d}/{MLP_EPOCHS} | "
                f"train loss = "
                f"{train_loss:.4f} | "
                f"train acc = "
                f"{train_accuracy:.4f} | "
                f"val loss = "
                f"{val_loss.item():.4f} | "
                f"val acc = "
                f"{val_accuracy:.4f}"
            )

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return model


# =============================================================================
# EVALUATE ONE TEXT
# =============================================================================

@torch.no_grad()
def evaluate_text(
    lm,
    classifier,
    sequence,
):

    original, generated = (
        build_trajectory_pair(
            lm,
            sequence,
        )
    )

    original_logits = classifier(
        original.unsqueeze(0)
        .to(DEVICE)
    )

    generated_logits = classifier(
        generated.unsqueeze(0)
        .to(DEVICE)
    )

    original_probs = F.softmax(
        original_logits,
        dim=1,
    )[0]

    generated_probs = F.softmax(
        generated_logits,
        dim=1,
    )[0]

    p0_original = float(
        original_probs[0]
    )

    p0_generated = float(
        generated_probs[0]
    )

    # ============================================================
    # MEMBERSHIP SCORE
    #
    # Higher:
    #
    #       original looks more ORIGINAL
    #       generated looks more GENERATED
    #
    # ============================================================

    separation = (
        p0_original
        - p0_generated
    )

    return {
        "p0_original": p0_original,
        "p0_generated": p0_generated,
        "separation": separation,
        "original_matrix": original,
        "generated_matrix": generated,
    }


# =============================================================================
# AUC
# =============================================================================

def auc_score(
    positive_scores,
    negative_scores,
):

    positive_scores = np.asarray(
        positive_scores,
        dtype=float,
    )

    negative_scores = np.asarray(
        negative_scores,
        dtype=float,
    )

    if (
        len(positive_scores) == 0
        or len(negative_scores) == 0
    ):

        return np.nan

    scores = np.concatenate(
        [
            positive_scores,
            negative_scores,
        ]
    )

    labels = np.concatenate(
        [
            np.ones(
                len(positive_scores)
            ),
            np.zeros(
                len(negative_scores)
            ),
        ]
    )

    order = np.argsort(
        scores,
        kind="mergesort",
    )

    sorted_scores = scores[
        order
    ]

    sorted_labels = labels[
        order
    ]

    ranks = np.empty(
        len(sorted_scores),
        dtype=float,
    )

    i = 0

    while i < len(
        sorted_scores
    ):

        j = i + 1

        while (
            j < len(sorted_scores)
            and sorted_scores[j]
            == sorted_scores[i]
        ):

            j += 1

        ranks[i:j] = (
            (
                i + 1
                + j
            )
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
            - n_pos
            * (n_pos + 1)
            / 2
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
    n_boot=2000,
    seed=1234,
):

    rng = np.random.default_rng(
        seed
    )

    positive_scores = np.asarray(
        positive_scores,
        dtype=float,
    )

    negative_scores = np.asarray(
        negative_scores,
        dtype=float,
    )

    observed = auc_score(
        positive_scores,
        negative_scores,
    )

    boot = []

    for _ in range(
        n_boot
    ):

        p = rng.choice(
            positive_scores,
            size=len(
                positive_scores
            ),
            replace=True,
        )

        n = rng.choice(
            negative_scores,
            size=len(
                negative_scores
            ),
            replace=True,
        )

        boot.append(
            auc_score(
                p,
                n,
            )
        )

    lower = float(
        np.percentile(
            boot,
            2.5,
        )
    )

    upper = float(
        np.percentile(
            boot,
            97.5,
        )
    )

    return (
        observed,
        lower,
        upper,
    )


# =============================================================================
# EVALUATE GROUP
# =============================================================================

def evaluate_group(
    name,
    lm,
    classifier,
    sequences,
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
            sequence,
        )

        results.append(
            result
        )

        print(
            f"\r{name}: "
            f"{i + 1}/{len(sequences)}",
            end="",
            flush=True,
        )

    print()

    scores = np.asarray(
        [
            r["separation"]
            for r in results
        ],
        dtype=float,
    )

    p_original = np.asarray(
        [
            r["p0_original"]
            for r in results
        ],
        dtype=float,
    )

    p_generated = np.asarray(
        [
            r["p0_generated"]
            for r in results
        ],
        dtype=float,
    )

    return {
        "results": results,
        "scores": scores,
        "p_original": p_original,
        "p_generated": p_generated,
    }


# =============================================================================
# SIMPLE STANDARDIZED EFFECT SIZE
# =============================================================================

def cohens_d(
    x,
    y,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    nx = len(x)

    ny = len(y)

    vx = np.var(
        x,
        ddof=1,
    )

    vy = np.var(
        y,
        ddof=1,
    )

    pooled = np.sqrt(
        (
            (nx - 1) * vx
            + (ny - 1) * vy
        )
        / (
            nx + ny - 2
        )
    )

    if pooled == 0:

        return np.nan

    return float(
        (
            x.mean()
            - y.mean()
        )
        / pooled
    )


# =============================================================================
# ONE EXPERIMENT
# =============================================================================

def run_experiment(
    repeat,
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
    # CREATE TEXTS
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
    # EVALUATION ORDINARY MEMBERS
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
    #
    # The vocabulary is constructed from training-side texts only.
    #
    # Evaluation texts are NOT needed to define the vocabulary.
    #
    # =========================================================================

    print(
        "Building vocabulary..."
    )

    vocabulary_texts = (
        ordinary_texts
        + strong_texts
        + classifier_train_texts
        + classifier_val_texts
    )

    vocab = build_vocabulary(
        vocabulary_texts
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
            text,
            vocab,
        )
        for text in ordinary_texts
    ]

    strong_sequences = [
        encode_text(
            text,
            vocab,
        )
        for text in strong_texts
    ]

    classifier_train_sequences = [
        encode_text(
            text,
            vocab,
        )
        for text in classifier_train_texts
    ]

    classifier_val_sequences = [
        encode_text(
            text,
            vocab,
        )
        for text in classifier_val_texts
    ]

    eval_ordinary_sequences = [
        encode_text(
            text,
            vocab,
        )
        for text in eval_ordinary_texts
    ]

    eval_strong_sequences = [
        encode_text(
            text,
            vocab,
        )
        for text in eval_strong_texts
    ]

    nonmember_sequences = [
        encode_text(
            text,
            vocab,
        )
        for text in nonmember_texts
    ]

    # =========================================================================
    # STEP 1 -- TRAIN LM
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
        hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)

    train_lm(
        lm,
        ordinary_sequences,
        strong_sequences,
        len(vocab),
    )

    # =========================================================================
    # TRAJECTORY SANITY CHECK
    # =========================================================================

    print()

    print("=" * 110)

    print(
        "TRAJECTORY SHAPE SANITY CHECK"
    )

    print("=" * 110)

    test_sequence = (
        classifier_train_sequences[0]
    )

    test_original, test_generated = (
        build_trajectory_pair(
            lm,
            test_sequence,
        )
    )

    expected_shape = (
        len(PREFIX_LENGTHS),
        HIDDEN_DIM,
    )

    print(
        f"Original trajectory shape = "
        f"{tuple(test_original.shape)}"
    )

    print(
        f"Generated trajectory shape = "
        f"{tuple(test_generated.shape)}"
    )

    print(
        f"Expected trajectory shape = "
        f"{expected_shape}"
    )

    assert tuple(
        test_original.shape
    ) == expected_shape

    assert tuple(
        test_generated.shape
    ) == expected_shape

    print(
        "✓ Trajectory shape check passed."
    )

    # =========================================================================
    # STEP 2 -- BUILD CLASSIFIER TRAINING DATA
    # =========================================================================

    print()

    print("=" * 110)

    print(
        "STEP 2: BUILD WHOLE-TRAJECTORY "
        "CLASSIFIER DATA"
    )

    print("=" * 110)

    X_traj_train, y_traj_train = (
        build_trajectory_classifier_dataset(
            lm,
            classifier_train_sequences,
        )
    )

    X_traj_val, y_traj_val = (
        build_trajectory_classifier_dataset(
            lm,
            classifier_val_sequences,
        )
    )

    # =========================================================================
    # FINAL DATA SHAPE CHECK
    # =========================================================================

    expected_train_shape = (
        2 * N_CLASSIFIER_TRAIN,
        len(PREFIX_LENGTHS),
        HIDDEN_DIM,
    )

    expected_val_shape = (
        2 * N_CLASSIFIER_VAL,
        len(PREFIX_LENGTHS),
        HIDDEN_DIM,
    )

    print()

    print(
        "Training trajectory tensor = "
        f"{tuple(X_traj_train.shape)}"
    )

    print(
        "Validation trajectory tensor = "
        f"{tuple(X_traj_val.shape)}"
    )

    assert tuple(
        X_traj_train.shape
    ) == expected_train_shape

    assert tuple(
        X_traj_val.shape
    ) == expected_val_shape

    # =========================================================================
    # STEP 3 -- TRAIN CLASSIFIER
    # =========================================================================

    print()

    print("=" * 110)

    print(
        "STEP 3: TRAIN WHOLE-TRAJECTORY "
        "ORIGINAL-vs-GENERATED CLASSIFIER"
    )

    print("=" * 110)

    classifier = train_classifier(
        X_traj_train,
        y_traj_train,
        X_traj_val,
        y_traj_val,
    )

    # =========================================================================
    # STEP 4 -- EVALUATION
    # =========================================================================

    strong_eval = evaluate_group(
        "STRONGLY MEMORIZED",
        lm,
        classifier,
        eval_strong_sequences,
    )

    ordinary_eval = evaluate_group(
        "ORDINARY MEMBERS",
        lm,
        classifier,
        eval_ordinary_sequences,
    )

    nonmember_eval = evaluate_group(
        "NON-MEMBERS",
        lm,
        classifier,
        nonmember_sequences,
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
    # AUCs
    # =========================================================================

    auc_strong = auc_score(
        strong_scores,
        nonmember_scores,
    )

    auc_ordinary = auc_score(
        ordinary_scores,
        nonmember_scores,
    )

    auc_strong_vs_ordinary = auc_score(
        strong_scores,
        ordinary_scores,
    )

    # =========================================================================
    # BOOTSTRAP CIs
    # =========================================================================

    strong_auc_obs, strong_ci_low, strong_ci_high = (
        bootstrap_auc_ci(
            strong_scores,
            nonmember_scores,
            seed=SEED + repeat,
        )
    )

    ordinary_auc_obs, ordinary_ci_low, ordinary_ci_high = (
        bootstrap_auc_ci(
            ordinary_scores,
            nonmember_scores,
            seed=SEED + 1000 + repeat,
        )
    )

    strong_vs_ordinary_obs, strong_vs_ordinary_ci_low, strong_vs_ordinary_ci_high = (
        bootstrap_auc_ci(
            strong_scores,
            ordinary_scores,
            seed=SEED + 2000 + repeat,
        )
    )

    # =========================================================================
    # EFFECT SIZES
    # =========================================================================

    d_strong = cohens_d(
        strong_scores,
        nonmember_scores,
    )

    d_ordinary = cohens_d(
        ordinary_scores,
        nonmember_scores,
    )

    # =========================================================================
    # PRINT RESULTS
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
        f"{auc_strong:.4f}"
    )

    print(
        f"  95% bootstrap CI = "
        f"[{strong_ci_low:.4f}, "
        f"{strong_ci_high:.4f}]"
    )

    print()

    print(
        "Ordinary member vs non-member:"
    )

    print(
        f"  AUC = "
        f"{auc_ordinary:.4f}"
    )

    print(
        f"  95% bootstrap CI = "
        f"[{ordinary_ci_low:.4f}, "
        f"{ordinary_ci_high:.4f}]"
    )

    print()

    print(
        "Strongly memorized vs ordinary member:"
    )

    print(
        f"  AUC = "
        f"{auc_strong_vs_ordinary:.4f}"
    )

    print(
        f"  95% bootstrap CI = "
        f"[{strong_vs_ordinary_ci_low:.4f}, "
        f"{strong_vs_ordinary_ci_high:.4f}]"
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
        f"Strongly memorized:"
    )

    print(
        f"  mean = "
        f"{strong_scores.mean():.4f}"
    )

    print(
        f"  SD   = "
        f"{strong_scores.std(ddof=1):.4f}"
    )

    print()

    print(
        f"Ordinary members:"
    )

    print(
        f"  mean = "
        f"{ordinary_scores.mean():.4f}"
    )

    print(
        f"  SD   = "
        f"{ordinary_scores.std(ddof=1):.4f}"
    )

    print()

    print(
        f"Non-members:"
    )

    print(
        f"  mean = "
        f"{nonmember_scores.mean():.4f}"
    )

    print(
        f"  SD   = "
        f"{nonmember_scores.std(ddof=1):.4f}"
    )

    # =========================================================================
    # EFFECT SIZES
    # =========================================================================

    print()

    print(
        "EFFECT SIZES"
    )

    print()

    print(
        f"Cohen's d: strong vs non-member = "
        f"{d_strong:.4f}"
    )

    print(
        f"Cohen's d: ordinary vs non-member = "
        f"{d_ordinary:.4f}"
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
        f"{strong_eval['p_original'].mean():.4f}"
    )

    print(
        f"  p(original | generated trajectory) = "
        f"{strong_eval['p_generated'].mean():.4f}"
    )

    print()

    print(
        "Ordinary members:"
    )

    print(
        f"  p(original | original trajectory) = "
        f"{ordinary_eval['p_original'].mean():.4f}"
    )

    print(
        f"  p(original | generated trajectory) = "
        f"{ordinary_eval['p_generated'].mean():.4f}"
    )

    print()

    print(
        "Non-members:"
    )

    print(
        f"  p(original | original trajectory) = "
        f"{nonmember_eval['p_original'].mean():.4f}"
    )

    print(
        f"  p(original | generated trajectory) = "
        f"{nonmember_eval['p_generated'].mean():.4f}"
    )

    # =========================================================================
    # PREFIX-WISE DIAGNOSTIC
    # =========================================================================

    print()

    print("=" * 110)

    print(
        "PREFIX-WISE EMBEDDING DISTANCE DIAGNOSTIC"
    )

    print("=" * 110)

    prefix_auc_results = []

    for j, prefix in enumerate(
        PREFIX_LENGTHS
    ):

        strong_distances = []

        nonmember_distances = []

        for result in (
            strong_eval["results"]
        ):

            original = result[
                "original_matrix"
            ][j]

            generated = result[
                "generated_matrix"
            ][j]

            distance = float(
                torch.norm(
                    original
                    - generated
                )
            )

            strong_distances.append(
                distance
            )

        for result in (
            nonmember_eval["results"]
        ):

            original = result[
                "original_matrix"
            ][j]

            generated = result[
                "generated_matrix"
            ][j]

            distance = float(
                torch.norm(
                    original
                    - generated
                )
            )

            nonmember_distances.append(
                distance
            )

        prefix_auc = auc_score(
            strong_distances,
            nonmember_distances,
        )

        prefix_auc_results.append(
            prefix_auc
        )

        print(
            f"Prefix {prefix:02d}: "
            f"strong-vs-nonmember "
            f"distance AUC = "
            f"{prefix_auc:.4f}"
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

    print()

    print(
        "The primary hypothesis is:"
    )

    print()

    print(
        "    AUC(strongly memorized, non-member)"
    )

    print(
        "        >"
    )

    print(
        "    AUC(ordinary member, non-member)"
    )

    print()

    print(
        "The strongly memorized group is a deliberate "
        "positive-control condition."
    )

    print()

    print(
        "The classifier itself never receives "
        "membership labels."
    )

    print()

    print(
        "It learns only to distinguish:"
    )

    print(
        "    original trajectory"
    )

    print(
        "            vs"
    )

    print(
        "    generated trajectory"
    )

    print()

    print(
        "Membership inference is therefore performed "
        "after classifier training."
    )

    # =========================================================================
    # RUNTIME
    # =========================================================================

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

        "repeat": repeat,

        "auc_strong":
            auc_strong,

        "auc_strong_ci_low":
            strong_ci_low,

        "auc_strong_ci_high":
            strong_ci_high,

        "auc_ordinary":
            auc_ordinary,

        "auc_ordinary_ci_low":
            ordinary_ci_low,

        "auc_ordinary_ci_high":
            ordinary_ci_high,

        "auc_strong_vs_ordinary":
            auc_strong_vs_ordinary,

        "auc_strong_vs_ordinary_ci_low":
            strong_vs_ordinary_ci_low,

        "auc_strong_vs_ordinary_ci_high":
            strong_vs_ordinary_ci_high,

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

        "cohens_d_strong":
            d_strong,

        "cohens_d_ordinary":
            d_ordinary,

        "classifier_original_strong":
            strong_eval[
                "p_original"
            ].mean(),

        "classifier_generated_strong":
            strong_eval[
                "p_generated"
            ].mean(),

        "classifier_original_ordinary":
            ordinary_eval[
                "p_original"
            ].mean(),

        "classifier_generated_ordinary":
            ordinary_eval[
                "p_generated"
            ].mean(),

        "classifier_original_nonmember":
            nonmember_eval[
                "p_original"
            ].mean(),

        "classifier_generated_nonmember":
            nonmember_eval[
                "p_generated"
            ].mean(),

        "runtime":
            elapsed,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    all_results = []

    for repeat in range(
        1,
        N_REPEATS + 1,
    ):

        result = run_experiment(
            repeat
        )

        all_results.append(
            result
        )

    # =========================================================================
    # FINAL AGGREGATION
    # =========================================================================

    print()

    print("#" * 110)

    print(
        "FINAL SUMMARY"
    )

    print("#" * 110)

    strong_auc = np.asarray(
        [
            r["auc_strong"]
            for r in all_results
        ]
    )

    ordinary_auc = np.asarray(
        [
            r["auc_ordinary"]
            for r in all_results
        ]
    )

    strong_vs_ordinary_auc = np.asarray(
        [
            r[
                "auc_strong_vs_ordinary"
            ]
            for r in all_results
        ]
    )

    # =========================================================================
    # PRIMARY RESULTS
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
    # AUC DIFFERENCE
    # =========================================================================

    auc_difference = (
        strong_auc
        - ordinary_auc
    )

    print()

    print(
        "Trajectory sensitivity to memorization:"
    )

    print(
        "  AUC difference "
        "(strong - ordinary) = "
        f"{auc_difference.mean():+.4f} "
        f"+/- "
        f"{auc_difference.std(ddof=1):.4f}"
    )

    # =========================================================================
    # SCORE SUMMARY
    # =========================================================================

    strong_mean = np.asarray(
        [
            r["strong_mean"]
            for r in all_results
        ]
    )

    ordinary_mean = np.asarray(
        [
            r["ordinary_mean"]
            for r in all_results
        ]
    )

    nonmember_mean = np.asarray(
        [
            r["nonmember_mean"]
            for r in all_results
        ]
    )

    print()

    print(
        "Mean membership scores "
        "across repeats:"
    )

    print(
        f"  Strongly memorized = "
        f"{strong_mean.mean():.4f} "
        f"+/- "
        f"{strong_mean.std(ddof=1):.4f}"
    )

    print(
        f"  Ordinary members   = "
        f"{ordinary_mean.mean():.4f} "
        f"+/- "
        f"{ordinary_mean.std(ddof=1):.4f}"
    )

    print(
        f"  Non-members        = "
        f"{nonmember_mean.mean():.4f} "
        f"+/- "
        f"{nonmember_mean.std(ddof=1):.4f}"
    )

    # =========================================================================
    # SAVE CSV
    # =========================================================================

    fieldnames = list(
        all_results[0].keys()
    )

    with open(
        RESULTS_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            all_results
        )

    print()

    print(
        f"Per-repeat results saved to:"
    )

    print(
        f"  {RESULTS_CSV}"
    )

    # =========================================================================
    # FINAL INTERPRETATION
    # =========================================================================

    print()

    print("=" * 110)

    print(
        "FINAL INTERPRETATION"
    )

    print("=" * 110)

    print()

    if (
        strong_auc.mean()
        > ordinary_auc.mean()
    ):

        print(
            "✓ The strongly memorized condition "
            "shows higher mean membership AUC "
            "than the ordinary-member condition."
        )

    else:

        print(
            "⚠ The strongly memorized condition "
            "does NOT show higher mean membership "
            "AUC than the ordinary-member condition."
        )

    print()

    print(
        "IMPORTANT FOR THE PAPER:"
    )

    print()

    print(
        "This experiment should be described as "
        "a controlled synthetic positive-control "
        "experiment rather than evidence of "
        "real-world privacy leakage."
    )

    print()

    print(
        "The deliberate repetition of strong-member "
        "texts creates a controlled memorization "
        "gradient."
    )

    print()

    print(
        "The scientifically interesting result is "
        "whether the trajectory-based score tracks "
        "this memorization manipulation."
    )

    print()

    print("=" * 110)

    print(
        "DONE"
    )

    print("=" * 110)