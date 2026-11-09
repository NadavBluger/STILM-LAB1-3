#!/usr/bin/env python3
"""Tests for lab 1-3. Usage: python test_1_3.py [todo_number ...]"""

import importlib.util
import math
import sys
from pathlib import Path

import torch

LAB_PATH = Path(__file__).resolve().parent / "lab-1-3-transformer-lm.py"


def load_lab():
    spec = importlib.util.spec_from_file_location("lab", LAB_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lab"] = module
    spec.loader.exec_module(module)
    return module


def check(name, got_fn, expected, compare=None):
    """Run got_fn(), compare against expected, print a PASS/FAIL line."""
    try:
        got = got_fn()
    except NotImplementedError:
        print(f"  [FAIL] {name}\n         not implemented yet")
        return False
    except Exception as error:
        print(f"  [FAIL] {name}\n         raised {type(error).__name__}: {error}")
        return False
    ok = compare(got, expected) if compare else got == expected
    if ok:
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name}\n         expected {expected!r}\n         got      {got!r}")
    return ok


def check_raises(name, fn, exc_type, message_substring):
    try:
        fn()
    except exc_type as error:
        if message_substring in str(error):
            print(f"  [PASS] {name}")
            return True
        print(f"  [FAIL] {name}\n         expected message containing {message_substring!r}\n         got      {str(error)!r}")
        return False
    except Exception as error:
        print(f"  [FAIL] {name}\n         expected {exc_type.__name__}, got {type(error).__name__}: {error}")
        return False
    print(f"  [FAIL] {name}\n         expected {exc_type.__name__}, nothing raised")
    return False


def nested_close(got, expected, tolerance=1e-5):
    """Compare nested lists of floats elementwise."""
    if isinstance(expected, list):
        if not isinstance(got, list) or len(got) != len(expected):
            return False
        return all(
            nested_close(g, e, tolerance) for g, e in zip(got, expected)
        )
    if not isinstance(got, (int, float)):
        return False
    return math.isclose(got, expected, rel_tol=tolerance, abs_tol=tolerance)


def set_fixed_weights(module):
    """Overwrite every parameter with a value fixed by its name and shape.

    The module is then fully deterministic, so its output can be compared
    against hardcoded numbers.
    """
    with torch.no_grad():
        for index, (name, parameter) in enumerate(sorted(module.named_parameters())):
            values = torch.sin(
                torch.arange(parameter.numel(), dtype=torch.float) * 0.37 + index
            )
            parameter.copy_((values * 0.3).view_as(parameter))
    return module


X = torch.tensor(
    [[[0.5, -0.5, 1.0, 0.0], [1.0, 0.25, -0.75, 0.5], [-1.0, 0.0, 0.5, 0.25]]]
)
MASK = torch.tensor(
    [[[True, False, False], [True, True, False], [True, True, True]]]
)
IDS = torch.tensor([[1, 2, 3]])
IDS_LAST_CHANGED = torch.tensor([[1, 2, 4]])
BATCH_IDS = torch.tensor([[1, 2, 3], [4, 0, 2]])


def parameter_names(module):
    return sorted(name for name, _ in module.named_parameters())


def todo_1(lab):
    """TODO 1: get_positional_embeddings"""
    return [
        check("returns one row per position",
              lambda: list(lab.get_positional_embeddings(5, 8).shape),
              [5, 8]),
        check("the first position is sin(0) and cos(0) interleaved",
              lambda: lab.get_positional_embeddings(1, 4).tolist(),
              [[0.0, 1.0, 0.0, 1.0]],
              compare=nested_close),
        check("sinusoidal values for three positions",
              lambda: lab.get_positional_embeddings(3, 4).tolist(),
              [[0.0, 1.0, 0.0, 1.0],
               [0.8414709568, 0.5403023362, 0.0099998331, 0.9999499917],
               [0.9092974067, -0.4161468446, 0.0199986659, 0.9998000264]],
              compare=nested_close),
        check("later dimensions use slower frequencies",
              lambda: lab.get_positional_embeddings(6, 4).sum(dim=0).tolist(),
              [0.1761615276, -0.2358185053, 0.1499624997, 5.9972500801],
              compare=nested_close),
    ]


def todo_2(lab):
    """TODO 2: TransformerLM"""
    model = set_fixed_weights(
        lab.TransformerLM(vocab_size=5, embedding_size=4, hidden_size=4, max_len=8)
    )
    return [
        check("the constructor builds the expected parameters",
              lambda: parameter_names(model),
              ["embed.embed.weight",
               "hidden2output.bias", "hidden2output.weight",
               "k_proj.bias", "k_proj.weight",
               "mlp.0.bias", "mlp.0.weight",
               "mlp.2.bias", "mlp.2.weight",
               "norm1.bias", "norm1.weight",
               "norm2.bias", "norm2.weight",
               "q_proj.bias", "q_proj.weight",
               "v_proj.bias", "v_proj.weight"]),
        check("returns one logit vector per position",
              lambda: list(model(BATCH_IDS).shape),
              [2, 3, 5]),
        check("logits of the fixed-weight model",
              lambda: model(IDS).tolist(),
              [[[0.02305475, 0.57225239, 0.57556653, 0.02990445, -0.13679959],
                [0.07289468, 0.55454022, 0.52251458, 0.03799598, -0.08228035],
                [0.06020135, 0.56268317, 0.53668457, 0.03242269, -0.097461]]],
              compare=nested_close),
        check("changing the last token leaves earlier logits untouched",
              lambda: torch.allclose(
                  model(IDS)[:, :2], model(IDS_LAST_CHANGED)[:, :2], atol=1e-6
              ),
              True),
        check("the last position does depend on the last token",
              lambda: torch.allclose(
                  model(IDS)[:, 2], model(IDS_LAST_CHANGED)[:, 2], atol=1e-6
              ),
              False),
    ]


def todo_3(lab):
    """TODO 3: TransformerBlock"""
    block = set_fixed_weights(lab.TransformerBlock(hidden_size=4))
    return [
        check("the constructor builds the expected parameters",
              lambda: parameter_names(block),
              ["k_proj.bias", "k_proj.weight",
               "mlp.0.bias", "mlp.0.weight",
               "mlp.2.bias", "mlp.2.weight",
               "norm1.bias", "norm1.weight",
               "norm2.bias", "norm2.weight",
               "q_proj.bias", "q_proj.weight",
               "v_proj.bias", "v_proj.weight"]),
        check("the block keeps the shape of its input",
              lambda: list(block(X, MASK).shape),
              [1, 3, 4]),
        check("output of the fixed-weight block",
              lambda: block(X, MASK).tolist(),
              [[[0.25270897, 0.24514697, 0.0336156, 0.16136092],
                [0.29130983, 0.27114633, 0.33863431, -0.10146715],
                [0.09231803, 0.26387405, 0.09618799, 0.00380011]]],
              compare=nested_close),
    ]


def todo_4(lab):
    """TODO 4: MultiLayerTransformerLM.forward"""
    model = set_fixed_weights(
        lab.MultiLayerTransformerLM(
            vocab_size=5, embedding_size=4, hidden_size=4, num_layers=2, max_len=8
        )
    )
    return [
        check("returns one logit vector per position",
              lambda: list(model(BATCH_IDS).shape),
              [2, 3, 5]),
        check("logits of the fixed-weight two-layer model",
              lambda: model(IDS).tolist(),
              [[[-0.12013067, -0.40621316, -0.40259039, -0.16899915, -0.11543907],
                [-0.1236958, -0.36186627, -0.39098331, -0.21124136, -0.13470653],
                [-0.10229252, -0.33314747, -0.40717858, -0.24289717, -0.12425187]]],
              compare=nested_close),
        check("changing the last token leaves earlier logits untouched",
              lambda: torch.allclose(
                  model(IDS)[:, :2], model(IDS_LAST_CHANGED)[:, :2], atol=1e-6
              ),
              True),
    ]


def todo_5(lab):
    """TODO 5: MultiHeadAttention"""
    layer = set_fixed_weights(lab.MultiHeadAttention(hidden_size=4, num_heads=2))
    heads = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    return [
        check("the constructor builds the expected parameters",
              lambda: parameter_names(layer),
              ["k_proj.bias", "k_proj.weight",
               "output_proj.bias", "output_proj.weight",
               "q_proj.bias", "q_proj.weight",
               "v_proj.bias", "v_proj.weight"]),
        check("split_heads puts the head axis before the position axis",
              lambda: list(layer.split_heads(heads).shape),
              [1, 2, 2, 2]),
        check("split_heads gives each head its own slice of the hidden vector",
              lambda: layer.split_heads(heads).tolist(),
              [[[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [7.0, 8.0]]]],
              compare=nested_close),
        check("combine_heads undoes split_heads",
              lambda: torch.equal(layer.combine_heads(layer.split_heads(heads)), heads),
              True),
        check("combine_heads concatenates the heads",
              lambda: list(layer.combine_heads(torch.zeros(2, 2, 3, 5)).shape),
              [2, 3, 10]),
        check("output of the fixed-weight attention layer",
              lambda: layer(X, MASK).tolist(),
              [[[0.27232033, 0.13414475, 0.10412662, 0.08213215],
                [0.27018532, 0.12496103, 0.10459621, 0.09140103],
                [0.24041715, 0.15111467, 0.13910717, 0.07150573]]],
              compare=nested_close),
    ]


def todo_6(lab):
    """TODO 6: MultiHeadTransformerBlock.forward"""
    block = set_fixed_weights(
        lab.MultiHeadTransformerBlock(hidden_size=4, num_heads=2)
    )
    return [
        check("the block keeps the shape of its input",
              lambda: list(block(X, MASK).shape),
              [1, 3, 4]),
        check("output of the fixed-weight block",
              lambda: block(X, MASK).tolist(),
              [[[0.17448883, 0.21542898, 0.23070838, 0.20529252],
                [0.13936071, 0.32262093, 0.25672421, -0.00715558],
                [-0.00353004, 0.29592347, 0.23512495, 0.13921556]]],
              compare=nested_close),
    ]


def todo_7(lab):
    """TODO 7: MultiHeadTransformerLM.forward"""
    model = set_fixed_weights(
        lab.MultiHeadTransformerLM(
            vocab_size=5,
            embedding_size=4,
            hidden_size=4,
            num_layers=2,
            num_heads=2,
            max_len=8,
        )
    )
    return [
        check("returns one logit vector per position",
              lambda: list(model(BATCH_IDS).shape),
              [2, 3, 5]),
        check("logits of the fixed-weight multi-head model",
              lambda: model(IDS).tolist(),
              [[[0.26419601, 0.62058026, 0.31669492, -0.19478169, -0.13349172],
                [0.2632345, 0.61815429, 0.31721649, -0.19226113, -0.13355622],
                [0.27034351, 0.61933053, 0.31032076, -0.19468787, -0.12710057]]],
              compare=nested_close),
        check("changing the last token leaves earlier logits untouched",
              lambda: torch.allclose(
                  model(IDS)[:, :2], model(IDS_LAST_CHANGED)[:, :2], atol=1e-6
              ),
              True),
    ]


TODOS = {
    1: ("get_positional_embeddings", todo_1),
    2: ("TransformerLM", todo_2),
    3: ("TransformerBlock", todo_3),
    4: ("MultiLayerTransformerLM.forward", todo_4),
    5: ("MultiHeadAttention", todo_5),
    6: ("MultiHeadTransformerBlock.forward", todo_6),
    7: ("MultiHeadTransformerLM.forward", todo_7),
}


def main():
    requested = sys.argv[1:]
    numbers = [int(n) for n in requested] if requested else sorted(TODOS)
    lab = load_lab()
    passed = total = 0
    for number in numbers:
        title, runner = TODOS[number]
        print(f"TODO {number}: {title}")
        try:
            results = runner(lab)
        except NotImplementedError:
            print("  [FAIL] not implemented yet")
            results = [False]
        except Exception as error:
            print(f"  [FAIL] raised {type(error).__name__}: {error}")
            results = [False]
        passed += sum(results)
        total += len(results)
        print()
    print(f"{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
