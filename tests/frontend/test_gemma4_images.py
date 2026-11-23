# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 image preparation settings match the checkpoint processor."""

import json
import os
from pathlib import Path

import pytest

from ignition.preparation.frontend import image_settings, prepare_frontend
from hyperflare_recipes.recipes.gemma4.recipe import RECIPE


ROOT = Path(__file__).resolve().parents[2]


def snapshot():
    checkpoint = next(c for c in RECIPE.checkpoints if c.alias == "gemma-4-31b")
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    path = (hub / ("models--" + checkpoint.repository.replace("/", "--"))
            / "snapshots" / checkpoint.revision)
    for name in ("config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (path / name).is_file():
            pytest.skip(f"{name} is not cached")
    return path


def test_prepared_image_settings_follow_the_checkpoint_processor():
    path = snapshot()
    settings = prepare_frontend(
        path, eos_token_ids=[1], image_profile="gemma4",
        preprocessor=path / "processor_config.json")["image"]
    assert settings == {"gemma4": {
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "max_soft_tokens": 280,
        "image_token": "<|image|>",
        "start_token": "<|image>",
        "end_token": "<image|>",
    }}


def test_image_tokens_are_atomic_and_distinct():
    from tokenizers import Tokenizer

    path = snapshot()
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    assert tokenizer.token_to_id("<|image>") == 255999
    assert tokenizer.token_to_id("<|image|>") == 258880
    assert tokenizer.token_to_id("<image|>") == 258882
    for token in ("<|image>", "<|image|>", "<image|>"):
        assert tokenizer.encode(token, add_special_tokens=False).ids == [
            tokenizer.token_to_id(token)]


def test_image_settings_reject_a_changed_processor():
    path = snapshot()
    config = json.loads((path / "processor_config.json").read_text())
    model = json.loads((path / "config.json").read_text())
    with pytest.raises(ValueError):
        image_settings({**config, "image_processor": {
            **config["image_processor"], "patch_size": 14}}, model, "gemma4")
    with pytest.raises(ValueError):
        image_settings({**config, "image_processor": {
            **config["image_processor"], "max_soft_tokens": 300}}, model, "gemma4")
