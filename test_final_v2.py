# =============================================================================
# FAST WHOLE-TRAJECTORY SUFFIX-EMBEDDING MEMBERSHIP BENCHMARK
#
# THREE-GROUP DESIGN
# =============================================================================
#
# LM TRAINING:
#
#   STRONG MEMBERS
#       -> deliberately repeated many times
#       -> intended to create a strong memorization condition
#
#   ORDINARY MEMBERS
#       -> seen once
#
#   CLASSIFIER TEXTS
#       -> separate texts
#       -> used only to train original-vs-generated classifier
#
#
# EVALUATION:
#
#   GROUP 1: STRONGLY MEMORIZED
#   GROUP 2: ORDINARY MEMBERS
#   GROUP 3: NON-MEMBERS
#
#
# For every text and every prefix:
#
#       prefix + suffix
#
# We extract an embedding of the SUFFIX ONLY.
#
# K prefixes -> K x D matrix.
#
# The CNN sees the WHOLE matrix.
#
# It learns:
#
#       class 0 = original suffix matrix
#       class 1 = generated suffix matrix
#
# It NEVER receives membership labels.
#
#
# FINAL QUESTION:
#
# Does the classifier's original-vs-generated behavior differ between:
#
#       strongly memorized texts
#       ordinary members
#       non-members?
#
# =============================================================================


import random
import time
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

print("=" * 100)
print("FAST WHOLE-TRAJECTORY SUFFIX-EMBEDDING MEMBERSHIP BENCHMARK")
print("=" * 100)
print(f"Device = {DEVICE}")


# =============================================================================
# DATA SIZE
# =============================================================================

# Ordinary texts used to train the LM.
N_ORDINARY_MEMBERS = 80

# Strongly memorized texts.
N_STRONG_MEMBERS = 10

# Separate texts used to train the original-vs-generated classifier.
N_CLASSIFIER_TEXTS = 40

# Final evaluation.
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
# PREFIX TRAJECTORY
# =============================================================================
#
# Fewer prefixes makes the benchmark substantially faster.
#
# You can later restore:
#
# [8, 12, 16, 20, 24, 28, 32]
#
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
# Each strongly memorized text is repeated this many times during LM training.
#
# This is deliberately artificial.
#
# The purpose is NOT to represent realistic language-model training.
#
# It creates a positive control where the LM has had many opportunities
# to memorize a particular text.
#
# =============================================================================

STRONG_REPEAT = 15


# =============================================================================
# MATRIX CLASSIFIER
# =============================================================================

CNN_CHANNELS = 24
MLP_HIDDEN = 48

MLP_EPOCHS = 50
MLP_LR = 2e-3
WEIGHT_DECAY = 1e-4

DROPOUT = 0.10
BATCH_SIZE = 32


# =============================================================================
# NUMBER OF REPEATS
# =============================================================================
#
# Start with ONE repeat for speed.
#
# Once the experiment behaves sensibly, increase to 3 or 5.
#
# =============================================================================

N_REPEATS = 1


# =============================================================================
# SEED
# =============================================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# SYNTHETIC TEXT
# =============================================================================

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


# =============================================================================
# NON-MEMBER TEXTS
#
# IMPORTANT:
#
# They use the SAME general text-generating vocabulary/domain.
#
# They are generated independently and are never included in LM training.
#
# =============================================================================

def make_nonmember_document(i, length=SEQ_LEN):

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
        ) % len(vocab)

        tokens.append(vocab[idx])

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

        hidden, _ = self.gru(emb)

        logits = self.output(hidden)

        return logits


# =============================================================================
# TRAIN LANGUAGE MODEL
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

    model.train()

    # ------------------------------------------------------------
    # Construct training list.
    #
    # Ordinary members appear once.
    #
    # Strong members appear STRONG_REPEAT times.
    # ------------------------------------------------------------

    training_sequences = []

    for seq in ordinary_sequences:

        training_sequences.append(seq)

    for seq in strong_sequences:

        for _ in range(STRONG_REPEAT):

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
# SUFFIX EMBEDDING
# =============================================================================
#
# IMPORTANT:
#
# The PREFIX provides context to the GRU.
#
# But only hidden states corresponding to the SUFFIX are pooled.
#
# Therefore:
#
#       one prefix -> one suffix embedding
#
# =============================================================================

@torch.no_grad()
def get_suffix_embedding(
    model,
    sequence,
    prefix_length
):

    model.eval()

    sequence = sequence.to(DEVICE)

    # We feed the entire sequence through the LM.
    x = sequence[:-1].unsqueeze(0)

    embeddings = model.embedding(x)

    hidden, _ = model.gru(
        embeddings
    )

    hidden = hidden[0]

    # ------------------------------------------------------------
    # Suffix starts immediately after prefix.
    #
    # hidden[t] corresponds to contextual processing around token t.
    # ------------------------------------------------------------

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

    return suffix_states.mean(
        dim=0
    ).cpu()


# =============================================================================
# GENERATE SUFFIX
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


# =============================================================================
# BUILD ORIGINAL MATRIX
#
# This is cached because it does not involve generation.
# =============================================================================

@torch.no_grad()
def build_original_matrix(
    model,
    sequence
):

    rows = []

    for prefix_length in PREFIX_LENGTHS:

        prefix_length = min(
            prefix_length,
            len(sequence) - 1
        )

        row = get_suffix_embedding(
            model,
            sequence,
            prefix_length
        )

        rows.append(row)

    return torch.stack(
        rows,
        dim=0
    )


# =============================================================================
# BUILD GENERATED MATRIX
# =============================================================================

@torch.no_grad()
def build_generated_matrix(
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

        rows.append(row)

    return torch.stack(
        rows,
        dim=0
    )


# =============================================================================
# BUILD PAIR
# =============================================================================

@torch.no_grad()
def build_pair(
    model,
    sequence
):

    original = build_original_matrix(
        model,
        sequence
    )

    generated = build_generated_matrix(
        model,
        sequence
    )

    return original, generated


# =============================================================================
# MATRIX CLASSIFIER
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

        # x:
        #
        # batch x K x D
        #
        # Conv1D:
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


# =============================================================================
# BUILD CLASSIFIER DATA
# =============================================================================

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
        "BUILDING ORIGINAL-vs-GENERATED "
        "TRAINING MATRICES"
    )
    print("=" * 100)

    total = len(sequences)

    for i, sequence in enumerate(
        sequences
    ):

        original = build_original_matrix(
            model,
            sequence
        )

        generated = build_generated_matrix(
            model,
            sequence
        )

        X.append(original)
        y.append(0)

        X.append(generated)
        y.append(1)

        print(
            f"\rClassifier matrices: "
            f"{i+1}/{total}",
            end="",
            flush=True
        )

    print()

    X = torch.stack(X)

    y = torch.tensor(
        y,
        dtype=torch.long
    )

    return X, y


# =============================================================================
# TRAIN CLASSIFIER
# =============================================================================

def train_classifier(
    X,
    y
):

    print()
    print("=" * 100)
    print(
        "TRAINING WHOLE-TRAJECTORY "
        "ORIGINAL-vs-GENERATED CLASSIFIER"
    )
    print("=" * 100)

    print(
        f"Training matrices = {len(X)}"
    )

    print(
        f"Matrix shape = "
        f"{tuple(X.shape[1:])}"
    )

    model = TrajectoryCNN(
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

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == MLP_EPOCHS
        ):

            mean_loss = (
                total_loss / len(X)
            )

            accuracy = (
                correct / len(X)
            )

            print(
                f"Epoch "
                f"{epoch:03d}/{MLP_EPOCHS} "
                f"loss = "
                f"{mean_loss:.4f} "
                f"accuracy = "
                f"{accuracy:.4f}"
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

    original = build_original_matrix(
        lm,
        sequence
    )

    generated = build_generated_matrix(
        lm,
        sequence
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

    p0_original = float(
        original_probs[0]
    )

    p0_generated = float(
        generated_probs[0]
    )

    # ------------------------------------------------------------
    # Membership score
    #
    # Large positive value means:
    #
    #   original looks ORIGINAL
    #   generated looks GENERATED
    #
    # This is the score we use for membership inference.
    # ------------------------------------------------------------

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

    order = np.argsort(scores)

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

    if n_pos == 0 or n_neg == 0:
        return np.nan

    return float(
        (
            positive_ranks.sum()
            - n_pos * (n_pos + 1) / 2
        )
        / (n_pos * n_neg)
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
    print("=" * 100)
    print(
        f"EVALUATING {name}"
    )
    print("=" * 100)

    results = []

    total = len(sequences)

    for i, sequence in enumerate(
        sequences
    ):

        result = evaluate_text(
            lm,
            classifier,
            sequence
        )

        results.append(result)

        print(
            f"\r{name}: "
            f"{i+1}/{total}",
            end="",
            flush=True
        )

    print()

    scores = np.array(
        [
            r["separation"]
            for r in results
        ]
    )

    p_original = np.array(
        [
            r["p0_original"]
            for r in results
        ]
    )

    p_generated = np.array(
        [
            r["p0_generated"]
            for r in results
        ]
    )

    return {
        "results": results,
        "scores": scores,
        "p_original": p_original,
        "p_generated": p_generated,
    }


# =============================================================================
# ONE EXPERIMENT
# =============================================================================

def run_experiment(repeat):

    start_time = time.time()

    print()
    print("#" * 100)
    print(
        f"REPEAT {repeat}"
    )
    print("#" * 100)

    set_seed(
        SEED + repeat
    )

    # =========================================================================
    # CREATE TEXT
    # =========================================================================

    print()
    print("Creating texts...")

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

    classifier_texts = [
        make_document(
            10000 + i
        )
        for i in range(
            N_CLASSIFIER_TEXTS
        )
    ]

    nonmember_texts = [
        make_nonmember_document(i)
        for i in range(
            N_NONMEMBERS
        )
    ]

    # =========================================================================
    # EVALUATION MEMBERS
    # =========================================================================

    eval_ordinary_texts = (
        ordinary_texts[
            :N_EVAL_ORDINARY
        ]
    )

    eval_strong_texts = strong_texts

    # =========================================================================
    # VOCABULARY
    # =========================================================================

    print("Building vocabulary...")

    all_texts = (
        ordinary_texts
        + strong_texts
        + classifier_texts
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

    print("Encoding texts...")

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

    classifier_sequences = [
        encode_text(
            x,
            vocab
        )
        for x in classifier_texts
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
    # TRAIN LM
    # =========================================================================

    print()
    print("=" * 100)
    print(
        "STEP 1: TRAIN LANGUAGE MODEL"
    )
    print("=" * 100)

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
    # TRAIN ORIGINAL-vs-GENERATED CLASSIFIER
    # =========================================================================

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

    # =========================================================================
    # EVALUATION
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
    # RESULTS
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

    # -------------------------------------------------------------------------
    # AUC 1:
    #
    # Strongly memorized vs non-member
    # -------------------------------------------------------------------------

    auc_strong = auc_score(
        strong_scores,
        nonmember_scores
    )

    # -------------------------------------------------------------------------
    # AUC 2:
    #
    # Ordinary member vs non-member
    # -------------------------------------------------------------------------

    auc_ordinary = auc_score(
        ordinary_scores,
        nonmember_scores
    )

    # -------------------------------------------------------------------------
    # AUC 3:
    #
    # Strongly memorized vs ordinary member
    #
    # This is particularly interesting.
    # -------------------------------------------------------------------------

    auc_strong_vs_ordinary = auc_score(
        strong_scores,
        ordinary_scores
    )

    # =========================================================================
    # PRINT
    # =========================================================================

    print()
    print("=" * 100)
    print(
        "FINAL MEMBERSHIP RESULTS"
    )
    print("=" * 100)

    print()

    print(
        "MEMBERSHIP AUC: STRONGLY MEMORIZED "
        "vs NON-MEMBER"
    )

    print(
        f"AUC = "
        f"{auc_strong:.4f}"
    )

    print()

    print(
        "MEMBERSHIP AUC: ORDINARY MEMBER "
        "vs NON-MEMBER"
    )

    print(
        f"AUC = "
        f"{auc_ordinary:.4f}"
    )

    print()

    print(
        "AUC: STRONGLY MEMORIZED "
        "vs ORDINARY MEMBER"
    )

    print(
        f"AUC = "
        f"{auc_strong_vs_ordinary:.4f}"
    )

    # =========================================================================
    # MEAN SCORES
    # =========================================================================

    print()
    print(
        "MEAN MEMBERSHIP SCORES"
    )

    print(
        f"Strongly memorized = "
        f"{strong_scores.mean():.4f}"
    )

    print(
        f"Ordinary members   = "
        f"{ordinary_scores.mean():.4f}"
    )

    print(
        f"Non-members        = "
        f"{nonmember_scores.mean():.4f}"
    )

    # =========================================================================
    # PROBABILITIES
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
        f"  p(original | original matrix)  = "
        f"{strong_eval['p_original'].mean():.4f}"
    )

    print(
        f"  p(original | generated matrix) = "
        f"{strong_eval['p_generated'].mean():.4f}"
    )

    print()

    print(
        "Ordinary members:"
    )

    print(
        f"  p(original | original matrix)  = "
        f"{ordinary_eval['p_original'].mean():.4f}"
    )

    print(
        f"  p(original | generated matrix) = "
        f"{ordinary_eval['p_generated'].mean():.4f}"
    )

    print()

    print(
        "Non-members:"
    )

    print(
        f"  p(original | original matrix)  = "
        f"{nonmember_eval['p_original'].mean():.4f}"
    )

    print(
        f"  p(original | generated matrix) = "
        f"{nonmember_eval['p_generated'].mean():.4f}"
    )

    # =========================================================================
    # PREFIX DIAGNOSTIC
    # =========================================================================

    print()
    print("=" * 100)
    print(
        "PREFIX-WISE DIAGNOSTIC"
    )
    print("=" * 100)

    for j, prefix in enumerate(
        PREFIX_LENGTHS
    ):

        strong_dist = []
        nonmember_dist = []

        for result in (
            strong_eval["results"]
        ):

            original = result[
                "original_matrix"
            ][j]

            generated = result[
                "generated_matrix"
            ][j]

            strong_dist.append(
                float(
                    torch.norm(
                        original - generated
                    )
                )
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

            nonmember_dist.append(
                float(
                    torch.norm(
                        original - generated
                    )
                )
            )

        auc = auc_score(
            strong_dist,
            nonmember_dist
        )

        print(
            f"Prefix {prefix:02d}: "
            f"strong-vs-nonmember "
            f"distance AUC = "
            f"{auc:.4f}"
        )

    # =========================================================================
    # INTERPRETATION
    # =========================================================================

    print()
    print("=" * 100)
    print(
        "INTERPRETATION"
    )
    print("=" * 100)

    print(
        """
The important comparison is:

    STRONGLY MEMORIZED vs NON-MEMBER
        versus
    ORDINARY MEMBER vs NON-MEMBER

If the benchmark is sensitive to memorization, we would hope to see:

    AUC(strongly memorized, non-member)
        >
    AUC(ordinary member, non-member)

and ideally the strongly memorized texts should have
a clearly different membership-score distribution.

The third AUC,

    strongly memorized vs ordinary member,

tests whether the method can distinguish the deliberately
memorized examples from ordinary members.

The classifier itself never receives membership labels.
It only learns:

    original suffix matrix
              vs
    generated suffix matrix

Membership inference happens only afterward.
"""
    )

    elapsed = time.time() - start_time

    print()
    print(
        f"Experiment runtime = "
        f"{elapsed:.1f} seconds"
    )

    return {
        "auc_strong": auc_strong,
        "auc_ordinary": auc_ordinary,
        "auc_strong_vs_ordinary":
            auc_strong_vs_ordinary,

        "strong_mean":
            strong_scores.mean(),

        "ordinary_mean":
            ordinary_scores.mean(),

        "nonmember_mean":
            nonmember_scores.mean(),
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    results = []

    for repeat in range(
        1,
        N_REPEATS + 1
    ):

        result = run_experiment(
            repeat
        )

        results.append(
            result
        )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print()
    print("#" * 100)
    print(
        "FINAL SUMMARY"
    )
    print("#" * 100)

    strong_auc = np.array(
        [
            x["auc_strong"]
            for x in results
        ]
    )

    ordinary_auc = np.array(
        [
            x["auc_ordinary"]
            for x in results
        ]
    )

    strong_vs_ordinary = np.array(
        [
            x["auc_strong_vs_ordinary"]
            for x in results
        ]
    )

    print()

    print(
        "Strongly memorized vs non-member:"
    )

    print(
        f"  AUC = "
        f"{strong_auc.mean():.4f} "
        f"+/- "
        f"{strong_auc.std():.4f}"
    )

    print()

    print(
        "Ordinary member vs non-member:"
    )

    print(
        f"  AUC = "
        f"{ordinary_auc.mean():.4f} "
        f"+/- "
        f"{ordinary_auc.std():.4f}"
    )

    print()

    print(
        "Strongly memorized vs ordinary member:"
    )

    print(
        f"  AUC = "
        f"{strong_vs_ordinary.mean():.4f} "
        f"+/- "
        f"{strong_vs_ordinary.std():.4f}"
    )

    print()
    print("=" * 100)
    print(
        "DONE"
    )
    print("=" * 100)