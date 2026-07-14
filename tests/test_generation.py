"""Adversarial tests for immutable generation publication and activation."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from typer.testing import CliRunner

from emb.generation import (
    DuplicateEntryIdError,
    EmbeddingValidationError,
    EntryValidationError,
    ManifestDigestMismatchError,
    ManifestValidationError,
    PayloadDigestMismatchError,
    PointerMalformedError,
    RecipeMismatchError,
    UnsafeGenerationIdError,
    activate_generation,
    read_active_manifest,
    read_generation,
    validate_generation,
    write_generation,
)
from emb.schema import Entry
from emb.search import SearchEngine
from emb.cli import app


runner = CliRunner()


def _recipe(model: str = "test-model", dimension: int = 4, recipe_id: str = "text-v1"):
    return {
        "recipe_id": recipe_id,
        "model": model,
        "dimension": dimension,
        "backend": "test",
        "task_type": "retrieval_document",
        "normalization": "l2",
        "input_modality": "text",
        "caption_inclusion": False,
        "version": 1,
    }


def _entries(prefix: str = "a"):
    return [
        Entry(id=f"{prefix}-1", text=f"{prefix} first", source="notes"),
        Entry(id=f"{prefix}-2", text=f"{prefix} second", source="docs"),
    ]


def _vectors(rows: int = 2, dimension: int = 4):
    values = np.arange(1, rows * dimension + 1, dtype=np.float32).reshape(rows, dimension)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _write(root, generation_id="g1", prefix="a", activate=True):
    return write_generation(
        root,
        _entries(prefix),
        _vectors(),
        generation_id=generation_id,
        lane="text",
        embedding_model="test-model",
        recipes=[_recipe()],
        source_manifest={"fingerprints": {"notes": "n1", "docs": "d1"}},
        generator={
            "app_git_commit": "app-sha",
            "emb_git_commit": "emb-sha",
            "generator_version": "test-v1",
        },
        activate=activate,
    )


def _rewrite_manifest(generation_path, mutate):
    path = generation_path / "generation.json"
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")


def _refresh_file_digest(generation_path, filename):
    def mutate(manifest):
        manifest["files"][filename]["sha256"] = hashlib.sha256(
            (generation_path / filename).read_bytes()
        ).hexdigest()

    _rewrite_manifest(generation_path, mutate)


def test_write_read_and_manifest_is_sole_authority(tmp_path):
    generation = _write(tmp_path)
    assert not (generation.path / "metadata.json").exists()
    entries, embeddings, manifest = read_generation(tmp_path)
    assert [entry.id for entry in entries] == ["a-1", "a-2"]
    assert embeddings.shape == (2, 4)
    assert manifest["recipe_id"] == "text-v1"
    assert manifest["sources"] == {
        "ids": ["docs", "notes"],
        "counts": {"docs": 1, "notes": 1},
        "fingerprints": {"notes": "n1", "docs": "d1"},
    }
    pointer = json.loads((tmp_path / "current.json").read_text())
    assert set(pointer) == {
        "schema_version",
        "generation_id",
        "generation_manifest_sha256",
    }


def test_active_manifest_status_read_does_not_load_payloads(tmp_path):
    generation = _write(tmp_path)
    (generation.path / "embeddings.npy").unlink()

    manifest = read_active_manifest(tmp_path)

    assert manifest["generation_id"] == "g1"
    with pytest.raises(PayloadDigestMismatchError):
        read_generation(tmp_path)


def test_interrupted_build_never_changes_active_pointer(tmp_path, monkeypatch):
    _write(tmp_path, "g1", "old")
    old_pointer = (tmp_path / "current.json").read_bytes()
    real_replace = os.replace

    def interrupt_build(source, target):
        if str(source).endswith("g2.building"):
            raise OSError("simulated interruption before publication")
        return real_replace(source, target)

    monkeypatch.setattr("emb.generation.os.replace", interrupt_build)
    with pytest.raises(OSError, match="simulated interruption"):
        _write(tmp_path, "g2", "new")
    assert (tmp_path / "current.json").read_bytes() == old_pointer
    assert (tmp_path / "generations" / "g2.building").is_dir()
    assert [entry.id for entry in read_generation(tmp_path)[0]] == ["old-1", "old-2"]


def test_corrupt_manifest_has_typed_failure(tmp_path):
    generation = _write(tmp_path, activate=False)
    (generation.path / "generation.json").write_text("{not json")
    with pytest.raises(ManifestValidationError) as failure:
        validate_generation(generation.path)
    assert failure.value.reason_code == "MANIFEST_INVALID"


def test_pointer_manifest_digest_mismatch_is_typed(tmp_path):
    _write(tmp_path)
    pointer_path = tmp_path / "current.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["generation_manifest_sha256"] = "0" * 64
    pointer_path.write_text(json.dumps(pointer))
    with pytest.raises(ManifestDigestMismatchError):
        read_generation(tmp_path)


def test_recipe_model_mismatch_is_typed(tmp_path):
    generation = _write(tmp_path, activate=False)
    _rewrite_manifest(
        generation.path,
        lambda manifest: manifest["recipes"][0].update(model="other-model"),
    )
    with pytest.raises(RecipeMismatchError):
        validate_generation(generation.path)


def test_mixed_recipe_requires_per_entry_recipe_ids(tmp_path):
    entries = _entries()
    recipes = [_recipe(recipe_id="text-v1"), _recipe(recipe_id="text-v2")]
    with pytest.raises(RecipeMismatchError, match="must name"):
        write_generation(
            tmp_path,
            entries,
            _vectors(),
            generation_id="mixed",
            lane="media",
            embedding_model="test-model",
            recipes=recipes,
            generator={
                "app_git_commit": "app",
                "emb_git_commit": "emb",
                "generator_version": "test-v1",
            },
        )


def test_duplicate_ids_are_rejected(tmp_path):
    entries = [Entry(id="same", text="first"), Entry(id="same", text="second")]
    with pytest.raises(DuplicateEntryIdError):
        write_generation(
            tmp_path,
            entries,
            _vectors(),
            generation_id="duplicate",
            lane="text",
            embedding_model="test-model",
            recipes=[_recipe()],
            generator={
                "app_git_commit": "app",
                "emb_git_commit": "emb",
                "generator_version": "test-v1",
            },
        )


def test_shape_mismatch_is_rejected(tmp_path):
    generation = _write(tmp_path, activate=False)
    _rewrite_manifest(
        generation.path,
        lambda manifest: manifest["embedding"].update(shape=[3, 4]),
    )
    with pytest.raises(EmbeddingValidationError, match="shape mismatch"):
        validate_generation(generation.path)


def test_count_mismatch_is_rejected(tmp_path):
    generation = _write(tmp_path, activate=False)
    _rewrite_manifest(
        generation.path,
        lambda manifest: manifest.update(entry_count=3),
    )
    with pytest.raises(EntryValidationError, match="entry count mismatch"):
        validate_generation(generation.path)


def test_dtype_mismatch_is_rejected(tmp_path):
    generation = _write(tmp_path, activate=False)
    _rewrite_manifest(
        generation.path,
        lambda manifest: manifest["embedding"].update(dtype="float64"),
    )
    with pytest.raises(EmbeddingValidationError, match="dtype mismatch"):
        validate_generation(generation.path)


def test_nonfinite_vectors_are_rejected_before_publication(tmp_path):
    vectors = _vectors()
    vectors[0, 0] = np.nan
    with pytest.raises(EmbeddingValidationError, match="NaN"):
        write_generation(
            tmp_path,
            _entries(),
            vectors,
            generation_id="nonfinite",
            lane="text",
            embedding_model="test-model",
            recipes=[_recipe()],
            generator={
                "app_git_commit": "app",
                "emb_git_commit": "emb",
                "generator_version": "test-v1",
            },
        )
    assert not (tmp_path / "current.json").exists()


def test_payload_digest_is_checked_before_use(tmp_path):
    generation = _write(tmp_path, activate=False)
    with (generation.path / "embeddings.npy").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(PayloadDigestMismatchError):
        validate_generation(generation.path)


def test_blank_jsonl_row_is_rejected(tmp_path):
    generation = _write(tmp_path, activate=False)
    entries_path = generation.path / "entries.jsonl"
    entries_path.write_bytes(entries_path.read_bytes().replace(b"\n", b"\n\n", 1))
    _refresh_file_digest(generation.path, "entries.jsonl")
    with pytest.raises(EntryValidationError, match="blank JSONL row"):
        validate_generation(generation.path)


def test_pointer_rollback_selects_previous_generation(tmp_path):
    _write(tmp_path, "g1", "old")
    _write(tmp_path, "g2", "new")
    assert [entry.id for entry in read_generation(tmp_path)[0]] == ["new-1", "new-2"]
    activate_generation(tmp_path, "g1")
    assert [entry.id for entry in read_generation(tmp_path)[0]] == ["old-1", "old-2"]


@pytest.mark.parametrize("generation_id", ["../escape", "a/b", "g1.building", ".."])
def test_pointer_rejects_traversal_and_building_targets(tmp_path, generation_id):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "current.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation_id": generation_id,
                "generation_manifest_sha256": "0" * 64,
            }
        )
    )
    with pytest.raises(UnsafeGenerationIdError):
        read_generation(tmp_path)


def test_pointer_rejects_stored_paths_and_extra_fields(tmp_path):
    (tmp_path / "current.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation_id": "g1",
                "generation_manifest_sha256": "0" * 64,
                "path": "/tmp/arbitrary",
            }
        )
    )
    with pytest.raises(PointerMalformedError):
        read_generation(tmp_path)


def test_concurrent_readers_observe_only_complete_generations(tmp_path):
    _write(tmp_path, "g1", "old")
    _write(tmp_path, "g2", "new", activate=False)

    def read_many(_):
        snapshots = []
        for _ in range(40):
            ids = tuple(entry.id for entry in read_generation(tmp_path)[0])
            snapshots.append(ids)
        return snapshots

    with ThreadPoolExecutor(max_workers=8) as pool:
        readers = [pool.submit(read_many, index) for index in range(8)]
        for _ in range(30):
            activate_generation(tmp_path, "g2")
            activate_generation(tmp_path, "g1")
        snapshots = [snapshot for reader in readers for snapshot in reader.result()]

    assert snapshots
    assert set(snapshots) <= {
        ("old-1", "old-2"),
        ("new-1", "new-2"),
    }


def test_search_engine_reads_active_generation(tmp_path):
    _write(tmp_path)
    engine = SearchEngine(tmp_path)
    assert [entry.id for entry in engine.entries] == ["a-1", "a-2"]
    assert engine.metadata["generation_id"] == "g1"


def test_info_reads_generation_manifest(tmp_path):
    _write(tmp_path)
    result = runner.invoke(app, ["info", str(tmp_path)])
    assert result.exit_code == 0
    assert "generation g1" in result.output
    assert "test-model" in result.output
    assert "notes: 1" in result.output
