"""Fine-tune MPNet as a bi-encoder for contrastive text matching.

This is deliberately plain PyTorch + Hugging Face Transformers. It uses:

* one shared MPNet encoder for queries and documents;
* attention-mask-aware mean pooling;
* L2-normalized sentence embeddings;
* symmetric InfoNCE / in-batch-negative loss;
* small synthetic query-document pairs so the file runs without a dataset SDK.

Full run (downloads microsoft/mpnet-base the first time):

    python mpnet_contrastive_finetune.py --epochs 3

Offline loss smoke test (does not download a model):

    python mpnet_contrastive_finetune.py --check-loss-only

For real work, replace PSEUDO_PAIRS with clean, one-to-one positive pairs. Within
each minibatch, avoid duplicate/near-duplicate positives because the loss would
incorrectly treat them as negatives.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


# -----------------------------------------------------------------------------
# Pseudo data
# -----------------------------------------------------------------------------
# Each pair says: "these two texts should have nearby embeddings."
# Other documents in the same minibatch automatically act as negatives.
PSEUDO_PAIRS: list[tuple[str, str]] = [
    (
        "How do I reset a password that I forgot?",
        "Open Account Settings, choose Security, and select Reset password.",
    ),
    (
        "Can I get my money back after buying an item?",
        "Eligible purchases can be refunded within 30 days of delivery.",
    ),
    (
        "Where can I download my monthly invoice?",
        "Invoices are available under Billing, then Documents and invoices.",
    ),
    (
        "How can I change the email address on my account?",
        "Edit your primary email from Profile Settings and confirm the new address.",
    ),
    (
        "Does the service support two-factor authentication?",
        "Two-factor authentication can be enabled from the Security page.",
    ),
    (
        "Why was my credit card payment declined?",
        "A payment may fail because of bank restrictions, an expired card, or incorrect details.",
    ),
    (
        "How do I cancel my paid subscription?",
        "Go to Billing, open Manage subscription, and choose Cancel plan.",
    ),
    (
        "Can I export all of my project data?",
        "Project owners can create a full export from Settings under Data export.",
    ),
    (
        "How many people can join the team plan?",
        "The team plan supports up to 50 active members per workspace.",
    ),
    (
        "Where do I create an API key?",
        "Create and revoke API keys from Developer Settings on the API keys tab.",
    ),
    (
        "The mobile app is not sending notifications.",
        "Check notification permissions in the phone settings and the app notification preferences.",
    ),
    (
        "How do I make another user a workspace administrator?",
        "Workspace owners can change a member's role to Admin from the Members page.",
    ),
    (
        "Can deleted files be restored?",
        "Deleted files remain in Trash for 30 days and can be restored during that period.",
    ),
    (
        "How do I connect the product to Slack?",
        "Install the Slack integration from the Integrations gallery and authorize your workspace.",
    ),
    (
        "What is the maximum file upload size?",
        "A single uploaded file may be up to 2 GB on paid plans.",
    ),
    (
        "How can I turn off marketing emails?",
        "Unsubscribe from promotional email in Notification Settings; service messages remain enabled.",
    ),
    (
        "Can I use the application without an internet connection?",
        "Offline mode allows reading cached documents, but syncing requires an internet connection.",
    ),
    (
        "How do I permanently delete my account?",
        "Request account deletion from Privacy Settings and confirm it with your password.",
    ),
    (
        "Where can I see who edited a document?",
        "Open Version history to view editors, timestamps, and previous document versions.",
    ),
    (
        "Does the product provide a student discount?",
        "Verified students receive 50 percent off the individual annual plan.",
    ),
]


@dataclass(frozen=True)
class TextPair:
    query: str
    positive: str


class PairDataset(Dataset[TextPair]):
    def __init__(self, pairs: Sequence[tuple[str, str]]) -> None:
        self.pairs = [TextPair(query=q, positive=p) for q, p in pairs]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> TextPair:
        return self.pairs[index]


class PairCollator:
    """Tokenize both sides independently; do not concatenate query and document."""

    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: Sequence[TextPair]) -> dict[str, object]:
        queries = [example.query for example in examples]
        positives = [example.positive for example in examples]
        tokenize = lambda texts: self.tokenizer(  # noqa: E731 - compact local adapter
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "query_inputs": tokenize(queries),
            "positive_inputs": tokenize(positives),
            "queries": queries,
            "positives": positives,
        }


# -----------------------------------------------------------------------------
# Model and loss
# -----------------------------------------------------------------------------
class MPNetBiEncoder(nn.Module):
    """A shared MPNet encoder that converts texts into unit-length vectors."""

    def __init__(self, model_name: str, projection_dim: int | None = None) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.projection = (
            nn.Linear(hidden_size, projection_dim, bias=False)
            if projection_dim is not None
            else nn.Identity()
        )
        self.output_dim = projection_dim or hidden_size

    @staticmethod
    def mean_pool(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        """Average only real tokens; exclude padding from the denominator."""
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        token_sum = (last_hidden_state * mask).sum(dim=1)
        token_count = mask.sum(dim=1).clamp_min(1e-9)
        return token_sum / token_count

    def forward(self, input_ids: Tensor, attention_mask: Tensor, **kwargs) -> Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)
        projected = self.projection(pooled)
        return F.normalize(projected, p=2, dim=-1)


def symmetric_infonce_loss(
    query_embeddings: Tensor,
    document_embeddings: Tensor,
    temperature: float = 0.05,
) -> tuple[Tensor, Tensor]:
    """Symmetric multiple-negatives ranking loss.

    For a batch of N aligned pairs, similarity[i, i] is positive. Every
    similarity[i, j] where i != j is an in-batch negative.

    Returns:
        loss: average of query->document and document->query cross-entropy.
        similarity: cosine-similarity matrix before temperature scaling.
    """
    if query_embeddings.shape != document_embeddings.shape:
        raise ValueError(
            "Aligned in-batch loss requires equal query/document embedding shapes; "
            f"got {query_embeddings.shape} and {document_embeddings.shape}."
        )
    if query_embeddings.ndim != 2:
        raise ValueError("Expected embeddings shaped [batch, embedding_dim].")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    similarity = query_embeddings @ document_embeddings.T
    logits = similarity / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    query_to_document = F.cross_entropy(logits, labels)
    document_to_query = F.cross_entropy(logits.T, labels)
    loss = 0.5 * (query_to_document + document_to_query)
    return loss, similarity


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------
def move_token_batch(batch, device: torch.device) -> dict[str, Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


@torch.inference_mode()
def encode_texts(
    model: MPNetBiEncoder,
    tokenizer,
    texts: Sequence[str],
    device: torch.device,
    max_length: int,
    batch_size: int = 32,
) -> Tensor:
    model.eval()
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        tokenized = tokenizer(
            list(texts[start : start + batch_size]),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        embeddings = model(**move_token_batch(tokenized, device))
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)


@torch.inference_mode()
def retrieval_metrics(
    model: MPNetBiEncoder,
    tokenizer,
    pairs: Sequence[tuple[str, str]],
    device: torch.device,
    max_length: int,
) -> dict[str, float]:
    queries = [query for query, _ in pairs]
    documents = [document for _, document in pairs]
    query_embeddings = encode_texts(model, tokenizer, queries, device, max_length)
    document_embeddings = encode_texts(model, tokenizer, documents, device, max_length)
    scores = query_embeddings @ document_embeddings.T
    labels = torch.arange(len(pairs))
    ranking = scores.argsort(dim=1, descending=True)
    top1 = (ranking[:, 0] == labels).float().mean().item()
    reciprocal_ranks = []
    for row, label in zip(ranking, labels):
        rank = (row == label).nonzero(as_tuple=False).item() + 1
        reciprocal_ranks.append(1.0 / rank)
    return {
        "top1_accuracy": top1,
        "mean_reciprocal_rank": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }


def train(
    model: MPNetBiEncoder,
    train_loader: DataLoader,
    validation_pairs: Sequence[tuple[str, str]],
    tokenizer,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    warmup_ratio: float,
    max_length: int,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            query_inputs = move_token_batch(batch["query_inputs"], device)
            positive_inputs = move_token_batch(batch["positive_inputs"], device)

            query_embeddings = model(**query_inputs)
            positive_embeddings = model(**positive_inputs)
            loss, similarity = symmetric_infonce_loss(
                query_embeddings,
                positive_embeddings,
                temperature=temperature,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            diagonal = similarity.diag().mean().item()
            if similarity.shape[0] > 1:
                off_diagonal = similarity[~torch.eye(
                    similarity.shape[0], dtype=torch.bool, device=device
                )].mean().item()
            else:
                off_diagonal = math.nan

            print(
                f"epoch={epoch} step={step}/{len(train_loader)} "
                f"loss={loss.item():.4f} positive_cos={diagonal:.3f} "
                f"negative_cos={off_diagonal:.3f}"
            )

        metrics = retrieval_metrics(
            model,
            tokenizer,
            validation_pairs,
            device,
            max_length,
        )
        print(
            f"epoch={epoch} mean_loss={running_loss / len(train_loader):.4f} "
            f"validation_top1={metrics['top1_accuracy']:.3f} "
            f"validation_mrr={metrics['mean_reciprocal_rank']:.3f}"
        )


def show_retrieval_examples(
    model: MPNetBiEncoder,
    tokenizer,
    pairs: Sequence[tuple[str, str]],
    device: torch.device,
    max_length: int,
) -> None:
    queries = [query for query, _ in pairs]
    documents = [document for _, document in pairs]
    query_embeddings = encode_texts(model, tokenizer, queries, device, max_length)
    document_embeddings = encode_texts(model, tokenizer, documents, device, max_length)
    scores = query_embeddings @ document_embeddings.T

    print("\nRetrieval examples:")
    for row, query in enumerate(queries):
        best = scores[row].argmax().item()
        expected = row
        mark = "OK" if best == expected else "MISS"
        print(f"[{mark}] query:    {query}")
        print(f"       retrieved: {documents[best]}")
        print(f"       cosine:    {scores[row, best].item():.3f}\n")


def save_model(
    model: MPNetBiEncoder,
    tokenizer,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(output_dir / "encoder")
    tokenizer.save_pretrained(output_dir / "encoder")
    torch.save(
        {
            "projection_state_dict": model.projection.state_dict(),
            "output_dim": model.output_dim,
        },
        output_dir / "projection.pt",
    )
    print(f"Saved encoder and projection to {output_dir}")


# -----------------------------------------------------------------------------
# A download-free numerical check of the contrastive objective
# -----------------------------------------------------------------------------
def check_loss_only() -> None:
    """Show, with actual numbers, that aligned pairs produce a lower loss."""
    torch.manual_seed(7)
    query = F.normalize(torch.randn(4, 8), dim=-1)
    aligned_documents = F.normalize(query + 0.05 * torch.randn(4, 8), dim=-1)
    shuffled_documents = aligned_documents[torch.tensor([2, 0, 3, 1])]

    aligned_loss, aligned_similarity = symmetric_infonce_loss(query, aligned_documents)
    shuffled_loss, shuffled_similarity = symmetric_infonce_loss(query, shuffled_documents)

    print("Aligned cosine-similarity matrix (diagonal should be largest):")
    print(aligned_similarity.round(decimals=3))
    print(f"aligned loss:  {aligned_loss.item():.4f}")
    print(f"shuffled loss: {shuffled_loss.item():.4f}")
    assert aligned_loss < shuffled_loss
    assert aligned_similarity.diag().mean() > shuffled_similarity.diag().mean()
    print("Loss smoke test passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="microsoft/mpnet-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--projection-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("mpnet-contrastive-output"))
    parser.add_argument(
        "--check-loss-only",
        action="store_true",
        help="Run a download-free numerical test of InfoNCE and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_loss_only:
        check_loss_only()
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device={device} model={args.model_name}")

    # Keep validation topics unseen during training for a real retrieval check.
    pairs = PSEUDO_PAIRS.copy()
    random.Random(args.seed).shuffle(pairs)
    validation_pairs = pairs[-4:]
    training_pairs = pairs[:-4]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = MPNetBiEncoder(args.model_name, args.projection_dim).to(device)
    train_loader = DataLoader(
        PairDataset(training_pairs),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,  # stable number of in-batch negatives per update
        collate_fn=PairCollator(tokenizer, args.max_length),
    )
    if len(train_loader) == 0:
        raise ValueError("batch-size is larger than the training set.")

    before = retrieval_metrics(
        model, tokenizer, validation_pairs, device, args.max_length
    )
    print("validation before training:", before)

    train(
        model=model,
        train_loader=train_loader,
        validation_pairs=validation_pairs,
        tokenizer=tokenizer,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
    )

    show_retrieval_examples(
        model, tokenizer, validation_pairs, device, args.max_length
    )
    save_model(model, tokenizer, args.output_dir)


if __name__ == "__main__":
    main()
