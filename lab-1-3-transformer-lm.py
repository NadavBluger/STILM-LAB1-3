"""Lab 1-3: Minimal Transformer language models.

This lab extends the attention language model from Lab 1-2 with the core
components of a Transformer language model. The code is organized in three
steps:

1. TransformerLM: one explicit single-layer, single-head Transformer.
2. TransformerBlock + MultiLayerTransformerLM: the same block repeated.
3. MultiHeadAttention + MultiHeadTransformerLM: multiple attention heads.

The goal is clarity first. Each step adds one idea without hiding the previous
one inside a highly configurable abstraction.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Optional, Protocol

import torch
import torch.nn.functional as F
from datasets import load_dataset
from sympy.codegen import Print
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class Config:
    seed: int = 1234
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name: str = "roneneldan/TinyStories"
    tokenizer_vocab_size: int = 5_000

    max_train_examples: int = 5_000
    max_valid_examples: int = 1_000
    block_size: int = 64
    batch_size: int = 32

    embedding_size: int = 512
    hidden_size: int = 512
    max_len: int = 1024
    num_layers: int = 2
    num_heads: int = 4
    model_type: str = "single"
    num_epochs: int = 3
    learning_rate: float = 3e-4

    sample_prompt: str = "Once upon a time "
    sample_new_tokens: int = 80
    temperature: float = 0.8
    top_k: Optional[int] = 50


LAB_1_2_REFERENCE = {
    "uniform_attention_valid_ppl": 261.2,
    "attention_lm_valid_ppl": 56.7,
}


class TinyStoriesBPETokenizer:
    """Small BPE tokenizer trained on the lab corpus."""

    def __init__(self, texts: list[str], vocab_size: int):
        self.tokenizer = HFTokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = ByteLevelDecoder()

        special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        self.tokenizer.train_from_iterator(texts, trainer=trainer)

        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.bos_id = self.tokenizer.token_to_id("[BOS]")
        self.eos_id = self.tokenizer.token_to_id("[EOS]")
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_tinystories_texts(config: Config) -> tuple[list[str], list[str]]:
    train = load_dataset(
        config.dataset_name, split=f"train[:{config.max_train_examples}]"
    )
    valid = load_dataset(
        config.dataset_name, split=f"validation[:{config.max_valid_examples}]"
    )
    return list(train["text"]), list(valid["text"])


def load_tokenizer(config: Config, train_texts: list[str]) -> TinyStoriesBPETokenizer:
    return TinyStoriesBPETokenizer(train_texts, vocab_size=config.tokenizer_vocab_size)


def tokenize_texts(texts: list[str], tokenizer: TinyStoriesBPETokenizer) -> list[int]:
    token_ids: list[int] = []
    for text in texts:
        if tokenizer.bos_id is not None:
            token_ids.append(tokenizer.bos_id)
        token_ids.extend(tokenizer.encode(text))
        if tokenizer.eos_id is not None:
            token_ids.append(tokenizer.eos_id)
    return token_ids


class TokenBlockDataset(Dataset):
    """Fixed-length next-token prediction examples."""

    def __init__(self, token_ids: list[int], block_size: int):
        if len(token_ids) <= block_size:
            raise ValueError("Need more tokens than block_size.")
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size
        self.num_blocks = (len(token_ids) - 1) // block_size

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.block_size
        x = self.token_ids[start : start + self.block_size]
        y = self.token_ids[start + 1 : start + self.block_size + 1]
        return x, y


def make_dataloader(
    token_ids: list[int],
    block_size: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TokenBlockDataset(token_ids, block_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def get_positional_embeddings(seq_len: int, embedding_size: int) -> torch.Tensor:
    """Build the sinusoidal positional embedding table.

    Attention alone ignores the order of the tokens. To fix this, transformers add
    an embedding of each token's absolute position to the token's embedding.
    This function builds the fixed sinusoidal positional embeddings from "Attention Is All You Need".

    The embedding of position t is a vector of D = embedding_size elements.
    Element i in this vector is a sine (even i) or a cosine (odd i) of t,
    at a frequency set by i:

        pe(t, i) = sin(t / 10000^(i / D))          if i is even
        pe(t, i) = cos(t / 10000^((i - 1) / D))    if i is odd

    Each sine at index i and the cosine at index i + 1 share one frequency.
    The base 10000 and the division by D are empirical choices.

    Args:
        seq_len: the number of positions to build an embedding for.
        embedding_size: the size D of a positional embedding, always even.

    Returns:
        A tensor of shape (seq_len, embedding_size) whose row t is the
        embedding of position t, for t = 0, ..., seq_len - 1.
    """
    all_pes = []
    for t in range(seq_len):
        pe = []
        for i in range(embedding_size):
            if i % 2 == 0:
                pe.append(math.sin(t / 10000 ** (i / embedding_size)))
            else:
                pe.append(math.cos(t / 10000 ** ((i - 1) / embedding_size)))
        all_pes.append(pe)
    pe_tensor = torch.tensor(all_pes, dtype=torch.float)
    return pe_tensor


class PositionalEmbedding(nn.Module):
    """Embed each token by combining its token ID and its position."""

    def __init__(self, vocab_size: int, embedding_size: int, max_len: int = 1024):
        super().__init__()
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(vocab_size, embedding_size)
        positional_embeddings = get_positional_embeddings(max_len, embedding_size)
        self.register_buffer("pe", positional_embeddings.unsqueeze(0))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.embed(input_ids) * math.sqrt(self.embedding_size)
        return token_embeddings + self.pe[:, : input_ids.size(1)]


def causal_mask(seq_len: int, device: torch.device | str) -> torch.Tensor:
    """Return a mask where position i can attend to positions j <= i."""
    mask = torch.full((seq_len, seq_len), False).to(device)
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:
                mask[i, j] = True
    return mask


def batched_causal_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """Return a causal mask expanded to the batch size of input_ids."""
    batch_size = input_ids.size(0)
    seq_len = input_ids.size(1)
    mask = causal_mask(seq_len, input_ids.device)
    return mask.unsqueeze(0).expand(batch_size, -1, -1)


def attention(
    batched_Q: torch.Tensor,
    batched_K: torch.Tensor,
    batched_V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention."""
    hidden_size = batched_Q.size(-1)
    batch_size = batched_Q.size(0)
    query_len = batched_Q.size(1)
    key_len = batched_K.size(1)

    assert batched_K.size(-1) == hidden_size
    assert batched_V.size(-1) == hidden_size
    assert batched_K.size(0) == batch_size
    assert batched_V.size(0) == batch_size
    assert batched_V.size(1) == key_len
    if mask is not None:
        assert mask.shape == torch.Size([batch_size, query_len, key_len])


    attn = batched_Q @ batched_K.transpose(-1, -2) / math.sqrt(hidden_size)
    if mask is not None:
        attn = attn.masked_fill(~mask, float('-inf'))

    batched_A = torch.softmax(attn, dim=-1)
    batched_C = batched_A @ batched_V

    return batched_A, batched_C



class TransformerLM(nn.Module):
    """A minimal single-layer, single-head Transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        max_len: int = 1024,
    ):
        """Create the layers of the model.

        Args:
            vocab_size: the number of tokens the model scores.
            embedding_size: the size of a token embedding.
            hidden_size: the size of the queries, keys and values, and of every
                hidden vector inside the model.
            max_len: the longest sequence the positional embeddings cover.

        The layers each name stands for are:
            embed: the embedding of a token id together with its position.
            q_proj, k_proj, v_proj: three separate linear layers, turning a
                hidden vector into a query, key and value vectors.
            mlp: a linear layer, a ReLU and a linear layer, all of width
                hidden_size.
            norm1, norm2: two separate layer normalizations over hidden_size.
            hidden2output: the linear layer scoring the vocabulary from a
                hidden vector.
        """
        super().__init__()
        assert (
            embedding_size == hidden_size
        ), "Residual connections require matching sizes."

        self.embed = PositionalEmbedding(vocab_size, embedding_size, max_len)
        self.q_proj = nn.Linear(embedding_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(embedding_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(embedding_size, hidden_size, bias=True)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size, bias=True),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size, bias=True),
        )
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.hidden2output = nn.Linear(embedding_size, vocab_size, bias=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Score the next token at every position.

        Args:
            input_ids: token ids of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size). Each position
            attends causally over the positions up to and including itself. The
            attention output is added to the embedding and normalized by norm1,
            the mlp output is added to that and normalized by norm2, and
            hidden2output scores the vocabulary from the result.

        Example:
            for input_ids of shape (2, 5) and a vocabulary of 100 tokens,
            returns a tensor of shape (2, 5, 100).
        """
        X = self.embed(input_ids)
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)
        _, A = attention(Q, K, V, batched_causal_mask(input_ids))
        H = self.norm1(X+A)
        MLPH = self.mlp(H)
        Y = self.norm2(H + MLPH)
        return self.hidden2output(Y)


class TransformerBlock(nn.Module):
    """A reusable single-head Transformer block.

    This is the same computation as the middle of TransformerLM, factored out so
    we can stack it without introducing multi-head attention yet.
    """

    def __init__(self, hidden_size: int):
        """Create the layers of the block.

        Args:
            hidden_size: the size of the queries, keys and values, and of every
                hidden vector inside the block.

        The layers each name stands for are:
            q_proj, k_proj, v_proj: three separate linear layers, turning a
                hidden vector into a query, a key and a value.
            mlp: a linear layer, a ReLU and a linear layer, all of width
                hidden_size.
            norm1, norm2: two separate layer normalizations over hidden_size.
        """
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run one Transformer block over a sequence of hidden vectors.

        Args:
            x: hidden vectors of shape (batch_size, seq_len, hidden_size).
            mask: boolean tensor of shape (batch_size, seq_len, seq_len), where
                True means attention is allowed.

        Returns:
            Hidden vectors of the same shape as x. Each position attends over
            the positions the mask allows. The attention output is added to x
            and normalized by norm1, the mlp output is added to that and
            normalized by norm2.

        Example:
            for x of shape (2, 5, 16) and a mask of shape (2, 5, 5), returns a
            tensor of shape (2, 5, 16).
        """
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        _, A = attention(Q, K, V, mask)
        H = self.norm1(x + A)
        MLPH = self.mlp(H)
        Y = self.norm2(H + MLPH)
        return Y


class MultiLayerTransformerLM(nn.Module):
    """A Transformer LM with several single-head Transformer blocks."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        max_len: int = 1024,
    ):
        super().__init__()
        assert (
            embedding_size == hidden_size
        ), "Residual connections require matching sizes."

        self.embed = PositionalEmbedding(vocab_size, embedding_size, max_len=max_len)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size) for _ in range(num_layers)]
        )
        self.hidden2output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Score the next token at every position, through a stack of blocks.

        Args:
            input_ids: token ids of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size). Each position
            attends causally over the positions up to and including itself, in
            every block of the stack. The blocks run one after the other, each
            reading what the previous one produced, and hidden2output scores
            the vocabulary from the output of the last block.

        Example:
            for input_ids of shape (2, 5) and a vocabulary of 100 tokens,
            returns a tensor of shape (2, 5, 100).
        """
        embbedings = self.embed(input_ids)
        mask = batched_causal_mask(input_ids)
        proccesed  = embbedings
        for block in self.blocks:
            proccesed = block(proccesed, mask)
        return self.hidden2output(proccesed)


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention.

    The implementation shows the four steps explicitly:
    project to Q/K/V, split into heads, attend in each head, concatenate heads.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        """Create the layers of the attention module.

        Args:
            hidden_size: the size of a hidden vector, divisible by num_heads.
            num_heads: the number of attention heads.

        The layers each name stands for are:
            q_proj, k_proj, v_proj: three separate linear layers, turning a
                hidden vector into the queries, keys and values of all heads at
                once.
            output_proj: the linear layer applied to the concatenated heads.

        Returns nothing. Besides the layers it also fills in `hidden_size`,
        `num_heads` and `head_size`, the size of one head.
        """
        super().__init__()
        assert hidden_size % num_heads == 0
        # TODO COMPLETE (5)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Give every head its own slice of each hidden vector.

        Args:
            x: hidden vectors of shape (batch_size, seq_len, hidden_size).

        Returns:
            A tensor of shape (batch_size, num_heads, seq_len, head_size). Head
            h of position p holds entries h * head_size to (h + 1) * head_size
            of the hidden vector at position p.

        Example:
            with num_heads=2, split_heads on [[[1., 2., 3., 4.],
                                               [5., 6., 7., 8.]]] ->
                [[[[1., 2.], [5., 6.]], [[3., 4.], [7., 8.]]]]
        """

        splitted = x.unflatten(2, (self.num_heads, self.head_size))
        splitted = splitted.transpose(1, 2)
        return splitted

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate the heads of each position back into one hidden vector.

        Args:
            x: per-head vectors of shape
                (batch_size, num_heads, seq_len, head_size).

        Returns:
            A tensor of shape (batch_size, seq_len, num_heads * head_size), the
            inverse of split_heads.

        Example:
            combine_heads(split_heads(x)) equals x.
        """
        combined = x.transpose(2, 1)
        combined = combined.flatten(2, 3)
        return combined

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))

        attn = q @ k.transpose(-1, -2) / math.sqrt(self.head_size)
        if mask is not None:
            attn = attn.masked_fill(~mask, float('-inf'))

        scores = attn
        attention_weights = torch.softmax(scores, dim=-1)
        context = attention_weights @ v

        context = self.combine_heads(context)
        return self.output_proj(context)


class MultiHeadTransformerBlock(nn.Module):
    """A Transformer block that uses multi-head attention."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run one multi-head Transformer block over a sequence.

        Args:
            x: hidden vectors of shape (batch_size, seq_len, hidden_size).
            mask: boolean tensor of shape (batch_size, seq_len, seq_len), where
                True means attention is allowed.

        Returns:
            Hidden vectors of the same shape as x. The attention output is
            added to x and normalized by norm1, the mlp output is added to that
            and normalized by norm2.

        Example:
            for x of shape (2, 5, 16) and a mask of shape (2, 5, 5), returns a
            tensor of shape (2, 5, 16).
        """
        A = self.attention(x, mask)
        H = self.norm1(x + A)
        MLPH = self.mlp(H)
        Y = self.norm2(H + MLPH)
        return Y


class MultiHeadTransformerLM(nn.Module):
    """A Transformer LM with multiple layers and multiple attention heads."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_len: int = 1024,
    ):
        super().__init__()
        assert (
            embedding_size == hidden_size
        ), "Residual connections require matching sizes."

        self.embed = PositionalEmbedding(vocab_size, embedding_size, max_len=max_len)
        self.blocks = nn.ModuleList(
            [
                MultiHeadTransformerBlock(hidden_size, num_heads)
                for _ in range(num_layers)
            ]
        )
        self.hidden2output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Score the next token at every position, with multi-head blocks.

        Args:
            input_ids: token ids of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size). Each position
            attends causally over the positions up to and including itself, in
            every block of the stack. The blocks run one after the other, each
            reading what the previous one produced, and hidden2output scores
            the vocabulary from the output of the last block.

        Example:
            for input_ids of shape (2, 5) and a vocabulary of 100 tokens,
            returns a tensor of shape (2, 5, 100).
        """
        # TODO COMPLETE (7)
        raise NotImplementedError("TODO 7")


def build_model(config: Config, vocab_size: int) -> nn.Module:
    if config.model_type == "single":
        return TransformerLM(
            vocab_size, config.embedding_size, config.hidden_size, config.max_len
        )
    if config.model_type == "multilayer":
        return MultiLayerTransformerLM(
            vocab_size,
            config.embedding_size,
            config.hidden_size,
            config.num_layers,
            config.max_len,
        )
    if config.model_type == "multihead":
        return MultiHeadTransformerLM(
            vocab_size,
            config.embedding_size,
            config.hidden_size,
            config.num_layers,
            config.num_heads,
            config.max_len,
        )
    raise ValueError(f"Unknown model type: {config.model_type}")


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.size(-1)
    return F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for input_ids, target_ids in tqdm(dataloader, desc="train", leave=False):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = loss_fn(logits, target_ids)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / total_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        logits = model(input_ids)
        total_loss += loss_fn(logits, target_ids).item()
        total_batches += 1

    avg_loss = total_loss / total_batches
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


@torch.no_grad()
def evaluate_with_shuffled_targets(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        shuffled_targets = target_ids.reshape(-1)
        shuffled_targets = shuffled_targets[
            torch.randperm(shuffled_targets.numel(), device=device)
        ]
        shuffled_targets = shuffled_targets.reshape_as(target_ids)

        logits = model(input_ids)
        total_loss += loss_fn(logits, shuffled_targets).item()
        total_batches += 1

    avg_loss = total_loss / total_batches
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def print_evaluation_sanity_checks(
    model: nn.Module,
    valid_loader: DataLoader,
    device: torch.device | str,
    label: str,
) -> None:
    valid_loss, valid_ppl = evaluate(model, valid_loader, device)
    shuffled_loss, shuffled_ppl = evaluate_with_shuffled_targets(
        model, valid_loader, device
    )
    print(
        f"{label}: valid loss {valid_loss:.3f}, valid ppl {valid_ppl:.1f}; "
        f"shuffled-target loss {shuffled_loss:.3f}, shuffled-target ppl {shuffled_ppl:.1f}"
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def print_lab_1_2_reference() -> None:
    print("\nLab 1-2 reference numbers on the same default setup:")
    print(
        f"uniform attention valid ppl: {LAB_1_2_REFERENCE['uniform_attention_valid_ppl']:.1f}"
    )
    print(f"attention LM valid ppl: {LAB_1_2_REFERENCE['attention_lm_valid_ppl']:.1f}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config: Config,
    label: str,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    print(f"\nTraining {label} model")

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, config.device)
        valid_loss, valid_ppl = evaluate(model, valid_loader, config.device)
        print(
            f"epoch {epoch}: "
            f"train loss {train_loss:.3f}, "
            f"valid loss {valid_loss:.3f}, "
            f"valid ppl {valid_ppl:.1f}"
        )


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer: TinyStoriesBPETokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device | str,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> str:
    model.eval()
    token_ids = tokenizer.encode(prompt)
    if tokenizer.bos_id is not None:
        token_ids = [tokenizer.bos_id] + token_ids

    max_len = model.embed.pe.size(1)
    for _ in range(max_new_tokens):
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        input_ids = input_ids[:, -max_len:]
        logits = model(input_ids)[:, -1, :]
        logits = logits / temperature

        if top_k is not None:
            top_values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
            cutoff = top_values[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < cutoff, float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        token_ids.append(next_id)

        if tokenizer.eos_id is not None and next_id == tokenizer.eos_id:
            break

    return tokenizer.decode(token_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train minimal Transformer language models."
    )
    parser.add_argument(
        "--model",
        choices=["single", "multilayer", "multihead"],
        default=Config.model_type,
    )
    parser.add_argument(
        "--max-train-examples", type=int, default=Config.max_train_examples
    )
    parser.add_argument(
        "--max-valid-examples", type=int, default=Config.max_valid_examples
    )
    parser.add_argument("--num-epochs", type=int, default=Config.num_epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--block-size", type=int, default=Config.block_size)
    parser.add_argument("--num-layers", type=int, default=Config.num_layers)
    parser.add_argument("--num-heads", type=int, default=Config.num_heads)
    parser.add_argument(
        "--tokenizer-vocab-size", type=int, default=Config.tokenizer_vocab_size
    )
    parser.add_argument(
        "--sample-new-tokens", type=int, default=Config.sample_new_tokens
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        model_type=args.model,
        max_train_examples=args.max_train_examples,
        max_valid_examples=args.max_valid_examples,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        block_size=args.block_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        tokenizer_vocab_size=args.tokenizer_vocab_size,
        sample_new_tokens=args.sample_new_tokens,
    )
    set_seed(config.seed)

    print(f"Using device: {config.device}")

    train_texts, valid_texts = load_tinystories_texts(config)
    tokenizer = load_tokenizer(config, train_texts)

    train_ids = tokenize_texts(train_texts, tokenizer)
    valid_ids = tokenize_texts(valid_texts, tokenizer)
    print(f"Train tokens: {len(train_ids):,}")
    print(f"Valid tokens: {len(valid_ids):,}")
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")
    print_lab_1_2_reference()

    train_loader = make_dataloader(
        train_ids, config.block_size, config.batch_size, shuffle=True
    )
    valid_loader = make_dataloader(
        valid_ids, config.block_size, config.batch_size, shuffle=False
    )

    model = build_model(config, tokenizer.vocab_size).to(config.device)
    label = config.model_type
    if config.model_type in {"multilayer", "multihead"}:
        label += f" ({config.num_layers} layers"
        if config.model_type == "multihead":
            label += f", {config.num_heads} heads"
        label += ")"

    print(f"\n{label} parameters: {count_parameters(model):,}")
    print_evaluation_sanity_checks(
        model, valid_loader, config.device, f"{label} before training"
    )
    train_model(model, train_loader, valid_loader, config, label)
    print_evaluation_sanity_checks(
        model, valid_loader, config.device, f"{label} after training"
    )

    print(f"\n{label} sample:")
    print(
        generate(
            model,
            tokenizer,
            config.sample_prompt,
            config.sample_new_tokens,
            config.device,
            temperature=config.temperature,
            top_k=config.top_k,
        )
    )


if __name__ == "__main__":
    main()
