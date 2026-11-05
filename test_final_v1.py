# ================================================================
# WHOLE-TRAJECTORY SUFFIX-EMBEDDING MEMBERSHIP BENCHMARK
#
# CORE IDEA
# ================================================================
#
# The LM is trained ONLY on MEMBER TEXTS.
#
# For every text and every prefix length:
#
#       prefix | suffix
#
# We obtain an embedding vector for the SUFFIX ONLY.
#
# For K different prefixes:
#
#       suffix embedding 1
#       suffix embedding 2
#       ...
#       suffix embedding K
#
# This gives one matrix per text:
#
#       K x D
#
# where:
#
#       K = number of prefixes
#       D = suffix embedding dimension
#
#
# TWO TYPES OF MATRICES
# ---------------------
#
# ORIGINAL MATRIX:
#
#   [ embedding(original suffix | prefix 1) ]
#   [ embedding(original suffix | prefix 2) ]
#   [ embedding(original suffix | prefix 3) ]
#   ...
#
#
# GENERATED MATRIX:
#
#   [ embedding(generated suffix | prefix 1) ]
#   [ embedding(generated suffix | prefix 2) ]
#   [ embedding(generated suffix | prefix 3) ]
#   ...
#
#
# The classifier is trained ONLY to distinguish:
#
#       class 0 = ORIGINAL suffix matrix
#       class 1 = GENERATED suffix matrix
#
# It NEVER receives membership labels.
#
#
# FINAL MEMBERSHIP TEST
# ---------------------
#
# We then take:
#
#       SEEN MEMBER TEXTS
#       NEVER-SEEN NON-MEMBER TEXTS
#
# and construct their original and generated matrices.
#
# The classifier's behavior is used as the membership score.
#
# ================================================================


import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# CONFIGURATION
# ================================================================

SEED = 1234

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ------------------------------------------------
# Corpus
# ------------------------------------------------

N_LM_MEMBERS = 120

# Separate texts used ONLY to train the
# original-vs-generated classifier.
N_CLASSIFIER_TEXTS = 60

# These are members of the LM training set
# and therefore TRUE MEMBERS for final evaluation.
N_EVAL_MEMBERS = 60

N_NONMEMBERS = 120


# ------------------------------------------------
# Text
# ------------------------------------------------

SEQ_LEN = 48


# ------------------------------------------------
# Language model
# ------------------------------------------------

EMBED_DIM = 32
HIDDEN_DIM = 48

LM_EPOCHS = 30
LM_LR = 2e-3


# ------------------------------------------------
# Prefix trajectory
# ------------------------------------------------

PREFIX_LENGTHS = [
    8,
    12,
    16,
    20,
    24,
    28,
    32,
]


# ------------------------------------------------
# Generation
# ------------------------------------------------

GEN_LEN = 8
TEMPERATURE = 1.15


# ------------------------------------------------
# Matrix classifier
# ------------------------------------------------

CNN_CHANNELS = 32
CNN_HIDDEN = 64

MLP_HIDDEN = 128

MLP_EPOCHS = 100
MLP_LR = 2e-3
WEIGHT_DECAY = 1e-4

DROPOUT = 0.10
BATCH_SIZE = 32


# ------------------------------------------------
# Repeats
# ------------------------------------------------

N_REPEATS = 5


# ================================================================
# SEED
# ================================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ================================================================
# SYNTHETIC MEMBER DOCUMENTS
# ================================================================

def make_document(i, length=SEQ_LEN):

    vocab = [
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

    tokens = []

    offset = i % len(vocab)

    for j in range(length):

        idx = (
            offset
            + j * 7
            + i * 13
            + j * j
        ) % len(vocab)

        tokens.append(vocab[idx])

    return " ".join(tokens)


# ================================================================
# NON-MEMBER DOCUMENTS
# ================================================================

def make_nonmember_document(i, length=SEQ_LEN):

    vocab = [
        "astronomy",
        "geology",
        "economics",
        "architecture",
        "engineering",
        "history",
        "language",
        "ecology",
        "computer",
        "network",
        "algorithm",
        "structure",
        "material",
        "energy",
        "planet",
        "market",
        "building",
        "system",
        "design",
        "process",
        "environment",
        "resource",
        "development",
        "technology",
        "measurement",
        "simulation",
        "experiment",
        "observation",
        "model",
        "analysis",
        "estimate",
        "parameter",
        "variation",
        "distribution",
        "sample",
        "result",
        "method",
        "comparison",
        "evaluation",
        "prediction",
        "classification",
        "validation",
        "performance",
        "feature",
        "signal",
        "pattern",
        "sequence",
        "temporal",
        "change",
        "effect",
    ]

    rng = random.Random(500000 + i)

    tokens = []

    for j in range(length):

        idx = (
            i * 17
            + j * 11
            + j * j * 3
            + rng.randint(0, 5)
        ) % len(vocab)

        tokens.append(vocab[idx])

    return " ".join(tokens)


# ================================================================
# TOKENIZATION
# ================================================================

def tokenize(text):
    return text.lower().split()


# ================================================================
# VOCABULARY
# ================================================================

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

        if len(vocab) >= 5000:
            break

        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab


# ================================================================
# ENCODE
# ================================================================

def encode_text(text, vocab):

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


# ================================================================
# LANGUAGE MODEL
# ================================================================

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

        hidden, _ = self.gru(emb)

        logits = self.output(hidden)

        return logits


# ================================================================
# TRAIN LM
# ================================================================

def train_lm(
    model,
    sequences,
    vocab_size
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LM_LR
    )

    model.train()

    for epoch in range(
        1,
        LM_EPOCHS + 1
    ):

        order = list(
            range(len(sequences))
        )

        random.shuffle(order)

        losses = []

        for idx in order:

            seq = sequences[idx]

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
                loss.item()
            )

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == LM_EPOCHS
        ):

            print(
                f"LM epoch "
                f"{epoch:03d}/{LM_EPOCHS} "
                f"loss = "
                f"{np.mean(losses):.4f}"
            )


# ================================================================
# SUFFIX EMBEDDING
#
# ================================================================
#
# THIS IS THE IMPORTANT PART.
#
# Given:
#
#       prefix + suffix
#
# we extract ONLY the hidden states belonging to the suffix.
#
# The prefix is used by the GRU as context, but the resulting
# embedding is calculated from the SUFFIX states.
#
# We then mean-pool the suffix hidden states.
#
# Therefore:
#
#       one prefix
#           -> one suffix embedding vector
#
# Several prefixes:
#
#       -> matrix K x HIDDEN_DIM
#
# ================================================================

@torch.no_grad()
def get_suffix_embedding(
    model,
    sequence,
    prefix_length
):

    model.eval()

    sequence = sequence.to(DEVICE)

    # ------------------------------------------------------------
    # We feed the entire sequence through the LM.
    # ------------------------------------------------------------

    x = sequence[:-1].unsqueeze(0)

    embeddings = model.embedding(x)

    hidden, _ = model.gru(
        embeddings
    )

    hidden = hidden[0]

    # hidden[t] represents the contextual state after token t.
    #
    # Token sequence:
    #
    #   0 ... prefix ... suffix
    #
    # We want hidden states associated with the suffix.
    #
    suffix_start = max(
        prefix_length - 1,
        0
    )

    suffix_states = hidden[
        suffix_start:
    ]

    if suffix_states.shape[0] == 0:

        return torch.zeros(
            HIDDEN_DIM
        )

    # ------------------------------------------------------------
    # IMPORTANT:
    #
    # ONLY SUFFIX STATES ARE USED.
    #
    # No explicit prefix embedding is concatenated.
    # ------------------------------------------------------------

    suffix_embedding = (
        suffix_states.mean(
            dim=0
        )
    )

    return suffix_embedding.cpu()


# ================================================================
# GENERATE SUFFIX
# ================================================================

@torch.no_grad()
def generate_suffix(
    model,
    prefix,
    gen_len,
    temperature
):

    model.eval()

    generated = (
        prefix.clone()
        .to(DEVICE)
    )

    for _ in range(gen_len):

        logits = model(
            generated.unsqueeze(0)
        )

        next_logits = (
            logits[0, -1]
            / temperature
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

    return generated


# ================================================================
# BUILD ONE TRAJECTORY MATRIX
# ================================================================
#
# For ONE TEXT:
#
#       prefix 8  -> suffix embedding
#       prefix 12 -> suffix embedding
#       prefix 16 -> suffix embedding
#       ...
#
# gives:
#
#       K x D
#
# ================================================================

@torch.no_grad()
def build_suffix_matrix(
    model,
    sequence,
    generated=False
):

    rows = []

    for prefix_length in PREFIX_LENGTHS:

        prefix_length = min(
            prefix_length,
            len(sequence) - 1
        )

        if generated:

            prefix = sequence[
                :prefix_length
            ]

            generated_sequence = (
                generate_suffix(
                    model,
                    prefix,
                    GEN_LEN,
                    TEMPERATURE
                )
            )

            row = get_suffix_embedding(
                model,
                generated_sequence,
                prefix_length
            )

        else:

            row = get_suffix_embedding(
                model,
                sequence,
                prefix_length
            )

        rows.append(row)

    matrix = torch.stack(
        rows,
        dim=0
    )

    return matrix


# ================================================================
# BUILD PAIRED MATRICES
# ================================================================

@torch.no_grad()
def build_pair(
    model,
    sequence
):

    original_matrix = (
        build_suffix_matrix(
            model,
            sequence,
            generated=False
        )
    )

    generated_matrix = (
        build_suffix_matrix(
            model,
            sequence,
            generated=True
        )
    )

    return (
        original_matrix,
        generated_matrix
    )


# ================================================================
# MATRIX CLASSIFIER
# ================================================================
#
# Input:
#
#       K x D
#
# The classifier receives the COMPLETE matrix.
#
# It is NOT trained row-by-row.
#
# A small 1D CNN operates along the PREFIX trajectory.
#
# Thus it can learn:
#
#       - individual suffix embeddings
#       - changes across prefixes
#       - trajectory patterns
#       - interactions between dimensions
#
# ================================================================

class TrajectoryCNN(nn.Module):

    def __init__(
        self,
        n_prefixes,
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

        # Input:
        #
        # batch x K x D
        #
        # Conv1d expects:
        #
        # batch x D x K

        x = x.transpose(
            1,
            2
        )

        x = F.relu(
            self.conv1(x)
        )

        x = F.relu(
            self.conv2(x)
        )

        x = self.pool(x)

        x = x.squeeze(
            -1
        )

        return self.classifier(x)


# ================================================================
# BUILD CLASSIFIER TRAINING DATA
#
# IMPORTANT:
#
# Each observation is an ENTIRE K x D MATRIX.
#
# class 0:
#       original suffix matrix
#
# class 1:
#       generated suffix matrix
#
# ================================================================

@torch.no_grad()
def build_classifier_dataset(
    model,
    sequences
):

    X = []
    y = []

    print()
    print("=" * 100)
    print(
        "BUILDING WHOLE-MATRIX "
        "ORIGINAL-vs-GENERATED DATA"
    )
    print("=" * 100)

    for i, sequence in enumerate(
        sequences
    ):

        (
            original_matrix,
            generated_matrix
        ) = build_pair(
            model,
            sequence
        )

        # --------------------------------------------------------
        # CLASS 0
        # --------------------------------------------------------

        X.append(
            original_matrix
        )

        y.append(0)

        # --------------------------------------------------------
        # CLASS 1
        # --------------------------------------------------------

        X.append(
            generated_matrix
        )

        y.append(1)

        if (
            (i + 1) % 10 == 0
            or i == len(sequences) - 1
        ):

            print(
                f"Processed "
                f"{i+1} / "
                f"{len(sequences)}"
            )

    X = torch.stack(
        X,
        dim=0
    )

    y = torch.tensor(
        y,
        dtype=torch.long
    )

    return X, y


# ================================================================
# TRAIN MATRIX CLASSIFIER
# ================================================================

def train_classifier(
    X,
    y
):

    model = TrajectoryCNN(
        n_prefixes=len(
            PREFIX_LENGTHS
        ),
        embedding_dim=HIDDEN_DIM
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=MLP_LR,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    X = X.to(DEVICE)
    y = y.to(DEVICE)

    print()
    print("=" * 100)
    print(
        "TRAINING WHOLE-TRAJECTORY "
        "ORIGINAL-vs-GENERATED CLASSIFIER"
    )
    print("=" * 100)

    print(
        f"Training matrices = "
        f"{len(X)}"
    )

    print(
        f"Matrix shape = "
        f"{tuple(X.shape[1:])}"
    )

    model.train()

    for epoch in range(
        1,
        MLP_EPOCHS + 1
    ):

        permutation = torch.randperm(
            len(X),
            device=DEVICE
        )

        total_loss = 0.0
        correct = 0

        for start in range(
            0,
            len(X),
            BATCH_SIZE
        ):

            idx = permutation[
                start:start + BATCH_SIZE
            ]

            xb = X[idx]
            yb = y[idx]

            logits = model(xb)

            loss = criterion(
                logits,
                yb
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

        mean_loss = (
            total_loss / len(X)
        )

        accuracy = (
            correct / len(X)
        )

        if (
            epoch == 1
            or epoch % 20 == 0
            or epoch == MLP_EPOCHS
        ):

            print(
                f"Epoch "
                f"{epoch:03d}/"
                f"{MLP_EPOCHS} "
                f"loss = "
                f"{mean_loss:.4f} "
                f"accuracy = "
                f"{accuracy:.4f}"
            )

    return model


# ================================================================
# EVALUATE ONE TEXT
# ================================================================
#
# For one text:
#
#       original matrix
#       generated matrix
#
# are fed independently through the classifier.
#
# We obtain:
#
#       p0(original matrix)
#       p0(generated matrix)
#
# Proposed membership score:
#
#       p0(original) - p0(generated)
#
# ================================================================

@torch.no_grad()
def evaluate_text(
    lm,
    classifier,
    sequence
):

    original_matrix, generated_matrix = (
        build_pair(
            lm,
            sequence
        )
    )

    original_input = (
        original_matrix
        .unsqueeze(0)
        .to(DEVICE)
    )

    generated_input = (
        generated_matrix
        .unsqueeze(0)
        .to(DEVICE)
    )

    original_logits = classifier(
        original_input
    )

    generated_logits = classifier(
        generated_input
    )

    original_probs = F.softmax(
        original_logits,
        dim=1
    )[0]

    generated_probs = F.softmax(
        generated_logits,
        dim=1
    )[0]

    p0_original = float(
        original_probs[0]
    )

    p0_generated = float(
        generated_probs[0]
    )

    p1_original = float(
        original_probs[1]
    )

    p1_generated = float(
        generated_probs[1]
    )

    separation = (
        p0_original
        - p0_generated
    )

    return {
        "p0_original": p0_original,
        "p0_generated": p0_generated,
        "p1_original": p1_original,
        "p1_generated": p1_generated,
        "separation": separation,
        "original_matrix": original_matrix,
        "generated_matrix": generated_matrix,
    }


# ================================================================
# AUC WITHOUT SKLEARN
# ================================================================
#
# This avoids the NumPy / sklearn compatibility problem.
#
# We calculate ROC-AUC from ranks.
#
# ================================================================

def auc_score(
    member_scores,
    nonmember_scores
):

    member_scores = np.asarray(
        member_scores,
        dtype=float
    )

    nonmember_scores = np.asarray(
        nonmember_scores,
        dtype=float
    )

    scores = np.concatenate(
        [
            member_scores,
            nonmember_scores
        ]
    )

    labels = np.concatenate(
        [
            np.ones(
                len(member_scores)
            ),
            np.zeros(
                len(nonmember_scores)
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

    ranks = np.empty_like(
        sorted_scores,
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

        avg_rank = (
            (i + 1)
            + j
        ) / 2.0

        ranks[i:j] = avg_rank

        i = j

    positive_ranks = ranks[
        sorted_labels == 1
    ]

    n_pos = len(
        member_scores
    )

    n_neg = len(
        nonmember_scores
    )

    if n_pos == 0 or n_neg == 0:
        return np.nan

    auc = (
        positive_ranks.sum()
        - n_pos * (n_pos + 1) / 2
    ) / (
        n_pos * n_neg
    )

    return float(auc)


# ================================================================
# ONE EXPERIMENT
# ================================================================

def run_experiment(
    repeat
):

    print()
    print("#" * 100)
    print(
        f"REPEAT {repeat:02d}"
    )
    print("#" * 100)

    set_seed(
        SEED + repeat
    )

    # ============================================================
    # CREATE DATA
    # ============================================================

    lm_member_texts = [
        make_document(i)
        for i in range(
            N_LM_MEMBERS
        )
    ]

    classifier_texts = [
        make_document(
            10000 + i
        )
        for i in range(
            N_CLASSIFIER_TEXTS
        )
    ]

    # IMPORTANT:
    #
    # These are ACTUAL members.
    #
    # They are selected from the same LM training corpus.
    #
    eval_member_texts = (
        lm_member_texts[
            :N_EVAL_MEMBERS
        ]
    )

    nonmember_texts = [
        make_nonmember_document(i)
        for i in range(
            N_NONMEMBERS
        )
    ]

    # ============================================================
    # VOCABULARY
    # ============================================================

    all_texts = (
        lm_member_texts
        + classifier_texts
        + nonmember_texts
    )

    vocab = build_vocabulary(
        all_texts
    )

    print()
    print("=" * 100)
    print(
        "WHOLE-TRAJECTORY "
        "SUFFIX-EMBEDDING BENCHMARK"
    )
    print("=" * 100)

    print(
        f"Device = {DEVICE}"
    )

    print(
        f"LM members = "
        f"{N_LM_MEMBERS}"
    )

    print(
        f"Evaluation members = "
        f"{N_EVAL_MEMBERS}"
    )

    print(
        f"Non-members = "
        f"{N_NONMEMBERS}"
    )

    print(
        f"Classifier texts = "
        f"{N_CLASSIFIER_TEXTS}"
    )

    print(
        f"Prefixes = "
        f"{PREFIX_LENGTHS}"
    )

    print(
        f"Number of trajectory points = "
        f"{len(PREFIX_LENGTHS)}"
    )

    print(
        f"Suffix embedding dimension = "
        f"{HIDDEN_DIM}"
    )

    print(
        f"Matrix shape = "
        f"({len(PREFIX_LENGTHS)}, "
        f"{HIDDEN_DIM})"
    )

    # ============================================================
    # ENCODE
    # ============================================================

    lm_sequences = [
        encode_text(
            text,
            vocab
        )
        for text in lm_member_texts
    ]

    classifier_sequences = [
        encode_text(
            text,
            vocab
        )
        for text in classifier_texts
    ]

    eval_member_sequences = [
        encode_text(
            text,
            vocab
        )
        for text in eval_member_texts
    ]

    nonmember_sequences = [
        encode_text(
            text,
            vocab
        )
        for text in nonmember_texts
    ]

    # ============================================================
    # STEP 1
    #
    # TRAIN LM ONLY ON MEMBERS
    # ============================================================

    print()
    print("=" * 100)
    print(
        "STEP 1: TRAIN LM ON MEMBERS ONLY"
    )
    print("=" * 100)

    lm = TinyLM(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    train_lm(
        lm,
        lm_sequences,
        len(vocab)
    )

    # ============================================================
    # STEP 2
    #
    # TRAIN ORIGINAL-vs-GENERATED CLASSIFIER
    #
    # IMPORTANT:
    #
    # This classifier never sees membership labels.
    # ============================================================

    X_train, y_train = (
        build_classifier_dataset(
            lm,
            classifier_sequences
        )
    )

    classifier = train_classifier(
        X_train,
        y_train
    )

    # ============================================================
    # STEP 3
    #
    # EVALUATE ACTUAL MEMBERS
    # ============================================================

    print()
    print("=" * 100)
    print(
        "EVALUATING TRUE MEMBERS"
    )
    print("=" * 100)

    member_results = []

    classifier.eval()

    for i, seq in enumerate(
        eval_member_sequences
    ):

        result = evaluate_text(
            lm,
            classifier,
            seq
        )

        member_results.append(
            result
        )

        if (
            (i + 1) % 10 == 0
            or i == len(
                eval_member_sequences
            ) - 1
        ):

            print(
                f"Members: "
                f"{i+1} / "
                f"{len(eval_member_sequences)}"
            )

    # ============================================================
    # STEP 4
    #
    # EVALUATE TRUE NON-MEMBERS
    # ============================================================

    print()
    print("=" * 100)
    print(
        "EVALUATING TRUE NON-MEMBERS"
    )
    print("=" * 100)

    nonmember_results = []

    for i, seq in enumerate(
        nonmember_sequences
    ):

        result = evaluate_text(
            lm,
            classifier,
            seq
        )

        nonmember_results.append(
            result
        )

        if (
            (i + 1) % 20 == 0
            or i == len(
                nonmember_sequences
            ) - 1
        ):

            print(
                f"Non-members: "
                f"{i+1} / "
                f"{len(nonmember_sequences)}"
            )

    # ============================================================
    # PREFIX-WISE MEMBERSHIP SIGNAL
    #
    # We can additionally inspect each row independently.
    #
    # BUT this is diagnostic only.
    #
    # The PRIMARY classifier operates on the WHOLE MATRIX.
    # ============================================================

    print()
    print("=" * 100)
    print(
        "WHOLE-MATRIX MEMBERSHIP RESULTS"
    )
    print("=" * 100)

    member_scores = np.array(
        [
            x["separation"]
            for x in member_results
        ]
    )

    nonmember_scores = np.array(
        [
            x["separation"]
            for x in nonmember_results
        ]
    )

    membership_auc = auc_score(
        member_scores,
        nonmember_scores
    )

    print()
    print(
        "PRIMARY MEMBERSHIP TEST"
    )

    print(
        f"Whole-matrix "
        f"membership AUC = "
        f"{membership_auc:.4f}"
    )

    print()
    print(
        "MEAN MEMBERSHIP SCORES"
    )

    print(
        f"Member mean score = "
        f"{member_scores.mean():.4f}"
    )

    print(
        f"Non-member mean score = "
        f"{nonmember_scores.mean():.4f}"
    )

    # ============================================================
    # PROBABILITIES
    # ============================================================

    member_p0_original = np.mean(
        [
            x["p0_original"]
            for x in member_results
        ]
    )

    member_p0_generated = np.mean(
        [
            x["p0_generated"]
            for x in member_results
        ]
    )

    nonmember_p0_original = np.mean(
        [
            x["p0_original"]
            for x in nonmember_results
        ]
    )

    nonmember_p0_generated = np.mean(
        [
            x["p0_generated"]
            for x in nonmember_results
        ]
    )

    print()
    print(
        "WHOLE-MATRIX CLASSIFIER "
        "PROBABILITIES"
    )

    print(
        f"Member p(class 0 | original matrix) = "
        f"{member_p0_original:.4f}"
    )

    print(
        f"Member p(class 0 | generated matrix) = "
        f"{member_p0_generated:.4f}"
    )

    print(
        f"Non-member p(class 0 | original matrix) = "
        f"{nonmember_p0_original:.4f}"
    )

    print(
        f"Non-member p(class 0 | generated matrix) = "
        f"{nonmember_p0_generated:.4f}"
    )

    # ============================================================
    # PREFIX-WISE DIAGNOSTIC
    #
    # Here we use the individual rows of the matrices only to
    # determine whether the signal is concentrated at certain
    # prefixes.
    #
    # This is NOT the primary classifier.
    # ============================================================

    print()
    print("=" * 100)
    print(
        "DIAGNOSTIC: PREFIX-WISE MATRIX ROW SIGNAL"
    )
    print("=" * 100)

    prefix_scores = []

    for j, prefix in enumerate(
        PREFIX_LENGTHS
    ):

        member_row_scores = []

        nonmember_row_scores = []

        for result in member_results:

            # Difference in the raw suffix embeddings.
            #
            # This is diagnostic, not the classifier score.
            original = (
                result[
                    "original_matrix"
                ][j]
            )

            generated = (
                result[
                    "generated_matrix"
                ][j]
            )

            score = float(
                torch.norm(
                    original - generated
                ).item()
            )

            member_row_scores.append(
                score
            )

        for result in nonmember_results:

            original = (
                result[
                    "original_matrix"
                ][j]
            )

            generated = (
                result[
                    "generated_matrix"
                ][j]
            )

            score = float(
                torch.norm(
                    original - generated
                ).item()
            )

            nonmember_row_scores.append(
                score
            )

        auc = auc_score(
            member_row_scores,
            nonmember_row_scores
        )

        prefix_scores.append(
            auc
        )

        print(
            f"Prefix "
            f"{prefix:02d}: "
            f"embedding-distance AUC = "
            f"{auc:.4f}"
        )

    print()

    return {
        "membership_auc":
            membership_auc,

        "member_mean_score":
            member_scores.mean(),

        "nonmember_mean_score":
            nonmember_scores.mean(),

        "prefix_scores":
            prefix_scores,
    }


# ================================================================
# MAIN
# ================================================================

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

    # ============================================================
    # SUMMARY
    # ============================================================

    aucs = np.array(
        [
            x["membership_auc"]
            for x in all_results
        ]
    )

    member_scores = np.array(
        [
            x["member_mean_score"]
            for x in all_results
        ]
    )

    nonmember_scores = np.array(
        [
            x["nonmember_mean_score"]
            for x in all_results
        ]
    )

    prefix_matrix = np.array(
        [
            x["prefix_scores"]
            for x in all_results
        ]
    )

    print()
    print("=" * 100)
    print(
        "FINAL SUMMARY"
    )
    print("=" * 100)

    print()

    print(
        f"WHOLE-MATRIX MEMBERSHIP AUC = "
        f"{aucs.mean():.4f} "
        f"+/- "
        f"{aucs.std():.4f}"
    )

    print()

    print(
        f"Member mean classifier score = "
        f"{member_scores.mean():.4f}"
    )

    print(
        f"Non-member mean classifier score = "
        f"{nonmember_scores.mean():.4f}"
    )

    print()

    print(
        "PREFIX-WISE DIAGNOSTIC AUC"
    )

    for j, prefix in enumerate(
        PREFIX_LENGTHS
    ):

        print(
            f"Prefix "
            f"{prefix:02d}: "
            f"{prefix_matrix[:, j].mean():.4f} "
            f"+/- "
            f"{prefix_matrix[:, j].std():.4f}"
        )

    print()
    print("=" * 100)
    print(
        "WHAT THE BENCHMARK TESTS"
    )
    print("=" * 100)

    print(
        """
The experiment now has the following structure:

                         TRAINING
                            |
                            v
                  +-------------------+
                  |   Language Model  |
                  |                   |
                  | trained ONLY on   |
                  | MEMBER documents  |
                  +-------------------+
                            |
                            |
             +--------------+--------------+
             |                             |
             v                             v
      ORIGINAL suffixes             GENERATED suffixes
             |                             |
             v                             v
       suffix embedding              suffix embedding
             |                             |
             +--------------+--------------+
                            |
                            v
                  K x D trajectory matrix
                            |
                            v
                 Original-vs-Generated
                       classifier
                            |
                    learns only:
                            |
                class 0 = ORIGINAL
                class 1 = GENERATED


                         EVALUATION
                            |
          +-----------------+-----------------+
          |                                   |
          v                                   v
    SEEN MEMBERS                         NON-MEMBERS
    (LM saw them)                       (LM never saw)
          |                                   |
          v                                   v
    original matrix                     original matrix
    generated matrix                    generated matrix
          |                                   |
          +-----------------+-----------------+
                            |
                            v
                    classifier scores
                            |
                            v
             membership discrimination
                            |
                            v
                          AUC
"""
    )