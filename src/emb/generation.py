"""Immutable, atomically activated embedding-index generations.

A lane root contains immutable generations and one small activation pointer::

    lane/
      generations/<generation-id>/{generation.json,entries.jsonl,embeddings.npy}
      current.json

``generation.json`` is the only metadata authority.  Writers stage beside the
final generation, validate the complete payload, rename it atomically, and only
then activate it.  Readers resolve ``current.json`` once and pin that immutable
generation for the lifetime of the returned data.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Set

import numpy as np

from emb.schema import Entry


SCHEMA_VERSION = 1
_GENERATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_RECIPE_FIELDS = {
    "recipe_id",
    "model",
    "dimension",
    "backend",
    "task_type",
    "normalization",
    "input_modality",
    "caption_inclusion",
    "version",
}


class GenerationError(ValueError):
    """Base class for stable, caller-distinguishable generation failures."""

    code = "GENERATION_ERROR"

    def __init__(self, message: str):
        super().__init__(message)
        self.reason_code = self.code


class GenerationBuildError(GenerationError):
    code = "GENERATION_BUILD_ERROR"


class GenerationExistsError(GenerationBuildError):
    code = "GENERATION_EXISTS"


class CrossFilesystemStagingError(GenerationBuildError):
    code = "CROSS_FILESYSTEM_STAGING"


class PointerError(GenerationError):
    code = "POINTER_ERROR"


class PointerNotFoundError(PointerError):
    code = "POINTER_NOT_FOUND"


class PointerMalformedError(PointerError):
    code = "POINTER_MALFORMED"


class UnsafeGenerationIdError(PointerError):
    code = "UNSAFE_GENERATION_ID"


class GenerationNotFoundError(PointerError):
    code = "GENERATION_NOT_FOUND"


class ManifestValidationError(GenerationError):
    code = "MANIFEST_INVALID"


class UnsupportedSchemaError(ManifestValidationError):
    code = "UNSUPPORTED_SCHEMA"


class ManifestDigestMismatchError(ManifestValidationError):
    code = "MANIFEST_DIGEST_MISMATCH"


class PayloadDigestMismatchError(GenerationError):
    code = "PAYLOAD_DIGEST_MISMATCH"


class EntryValidationError(GenerationError):
    code = "ENTRY_INVALID"


class DuplicateEntryIdError(EntryValidationError):
    code = "DUPLICATE_ENTRY_ID"


class EmbeddingValidationError(GenerationError):
    code = "EMBEDDING_INVALID"


class RecipeMismatchError(GenerationError):
    code = "RECIPE_MISMATCH"


@dataclass(frozen=True)
class Generation:
    """A fully validated and pinned generation."""

    generation_id: str
    path: Path
    manifest: dict[str, Any]
    entries: list[Entry]
    embeddings: np.ndarray


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(handle: Any) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_generation_id(generation_id: object) -> str:
    if not isinstance(generation_id, str):
        raise UnsafeGenerationIdError("generation_id must be a string")
    if (
        generation_id in {".", ".."}
        or generation_id.endswith(".building")
        or not _GENERATION_ID_RE.fullmatch(generation_id)
    ):
        raise UnsafeGenerationIdError(
            f"unsafe generation_id {generation_id!r}; path separators and "
            ".building targets are forbidden"
        )
    return generation_id


def _entry_dict(entry: Entry | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(entry, Entry):
        return entry.to_dict()
    if isinstance(entry, Mapping):
        return dict(entry)
    raise EntryValidationError(f"entry must be Entry or mapping, got {type(entry)!r}")


def _validate_raw_entries(
    raw_entries: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, int]]:
    ids: list[str] = []
    seen: set[str] = set()
    source_counts: dict[str, int] = {}
    for row_number, raw in enumerate(raw_entries, start=1):
        entry_id = raw.get("id")
        text = raw.get("text")
        if not isinstance(entry_id, str) or not entry_id.strip():
            raise EntryValidationError(f"row {row_number} has a blank or non-string id")
        if not isinstance(text, str) or not text.strip():
            raise EntryValidationError(f"row {row_number} has blank or non-string text")
        if entry_id in seen:
            raise DuplicateEntryIdError(f"duplicate entry id {entry_id!r}")
        seen.add(entry_id)
        ids.append(entry_id)
        source = raw.get("source") or "unknown"
        if not isinstance(source, str):
            raise EntryValidationError(f"row {row_number} has non-string source")
        source_counts[source] = source_counts.get(source, 0) + 1
    return ids, source_counts


def _validate_recipes(
    recipes: Sequence[Mapping[str, Any]],
    *,
    embedding_model: str,
    embedding_dim: int,
) -> list[dict[str, Any]]:
    if not recipes:
        raise RecipeMismatchError("at least one complete recipe is required")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, recipe_value in enumerate(recipes):
        recipe = dict(recipe_value)
        missing = sorted(_RECIPE_FIELDS - recipe.keys())
        if missing:
            raise RecipeMismatchError(
                f"recipe {index} is missing identity fields: {', '.join(missing)}"
            )
        recipe_id = recipe["recipe_id"]
        if not isinstance(recipe_id, str) or not recipe_id.strip():
            raise RecipeMismatchError(f"recipe {index} has a blank recipe_id")
        if recipe_id in seen:
            raise RecipeMismatchError(f"duplicate recipe_id {recipe_id!r}")
        seen.add(recipe_id)
        for field in (
            "model",
            "backend",
            "task_type",
            "normalization",
            "input_modality",
        ):
            if not isinstance(recipe[field], str) or not recipe[field].strip():
                raise RecipeMismatchError(
                    f"recipe {recipe_id!r} field {field} must be a nonblank string"
                )
        if not isinstance(recipe["caption_inclusion"], bool):
            raise RecipeMismatchError(
                f"recipe {recipe_id!r} caption_inclusion must be boolean"
            )
        if not isinstance(recipe["version"], (str, int)) or isinstance(
            recipe["version"], bool
        ):
            raise RecipeMismatchError(
                f"recipe {recipe_id!r} version must be a string or integer"
            )
        if recipe["model"] != embedding_model:
            raise RecipeMismatchError(
                f"recipe {recipe_id!r} model {recipe['model']!r} does not match "
                f"generation model {embedding_model!r}"
            )
        if recipe["dimension"] != embedding_dim:
            raise RecipeMismatchError(
                f"recipe {recipe_id!r} dimension {recipe['dimension']!r} does not "
                f"match generation dimension {embedding_dim}"
            )
        normalized.append(recipe)
    normalized.sort(key=lambda item: item["recipe_id"])
    return normalized


def _validate_entry_recipes(
    raw_entries: Sequence[Mapping[str, Any]], recipes: Sequence[Mapping[str, Any]]
) -> None:
    recipe_ids = {str(recipe["recipe_id"]) for recipe in recipes}
    if len(recipe_ids) == 1:
        expected = next(iter(recipe_ids))
        for row_number, raw in enumerate(raw_entries, start=1):
            metadata = raw.get("metadata") or {}
            if not isinstance(metadata, Mapping):
                raise EntryValidationError(f"row {row_number} metadata must be an object")
            row_recipe = metadata.get("recipe_id")
            if row_recipe is not None and row_recipe != expected:
                raise RecipeMismatchError(
                    f"row {row_number} recipe {row_recipe!r} does not match "
                    f"homogeneous recipe {expected!r}"
                )
        return

    for row_number, raw in enumerate(raw_entries, start=1):
        metadata = raw.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise EntryValidationError(f"row {row_number} metadata must be an object")
        row_recipe = metadata.get("recipe_id")
        if not isinstance(row_recipe, str) or row_recipe not in recipe_ids:
            raise RecipeMismatchError(
                f"row {row_number} must name one of the generation recipe IDs "
                f"{sorted(recipe_ids)}"
            )


def _load_manifest(
    generation_path: Path, expected_manifest_sha256: Optional[str] = None
) -> tuple[dict[str, Any], bytes]:
    manifest_path = generation_path / "generation.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
    except FileNotFoundError as exc:
        raise ManifestValidationError(f"missing {manifest_path}") from exc
    if expected_manifest_sha256 is not None:
        actual = _sha256_bytes(manifest_bytes)
        if actual != expected_manifest_sha256:
            raise ManifestDigestMismatchError(
                f"generation.json digest mismatch: expected {expected_manifest_sha256}, "
                f"got {actual}"
            )
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"malformed generation manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ManifestValidationError("generation manifest must be a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise UnsupportedSchemaError(
            f"unsupported generation schema {manifest.get('schema_version')!r}"
        )
    return manifest, manifest_bytes


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        raw_bytes = path.read_bytes()
    except FileNotFoundError as exc:
        raise EntryValidationError(f"missing {path}") from exc
    raw_lines = raw_bytes.splitlines()
    rows: list[dict[str, Any]] = []
    for row_number, line in enumerate(raw_lines, start=1):
        if not line.strip():
            raise EntryValidationError(f"blank JSONL row {row_number}")
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EntryValidationError(f"malformed JSONL row {row_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise EntryValidationError(f"JSONL row {row_number} must be an object")
        rows.append(row)
    return rows


def validate_generation(
    generation_path: Path,
    *,
    expected_manifest_sha256: Optional[str] = None,
    mmap: bool = True,
) -> Generation:
    """Validate every manifest and payload invariant and return pinned data."""

    generation_path = Path(generation_path)
    manifest, _ = _load_manifest(generation_path, expected_manifest_sha256)
    generation_id = _validate_generation_id(manifest.get("generation_id"))
    expected_dir_names = {generation_id, f"{generation_id}.building"}
    if generation_path.name not in expected_dir_names:
        raise ManifestValidationError(
            f"manifest generation_id {generation_id!r} does not match directory "
            f"{generation_path.name!r}"
        )

    required = {
        "lane",
        "created_at",
        "entry_count",
        "embedding",
        "recipes",
        "sources",
        "source_manifest_digest",
        "stable_id_digest",
        "ordered_entry_digest",
        "files",
        "generator",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ManifestValidationError(
            f"generation manifest is missing fields: {', '.join(missing)}"
        )
    generator = manifest.get("generator")
    if not isinstance(generator, dict):
        raise ManifestValidationError("generator must be an object")
    for field in ("app_git_commit", "emb_git_commit", "generator_version"):
        if not isinstance(generator.get(field), str) or not generator[field].strip():
            raise ManifestValidationError(f"generator.{field} must be nonblank")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ManifestValidationError("files must be an object")
    entries_path = generation_path / "entries.jsonl"
    embeddings_path = generation_path / "embeddings.npy"
    for name, path in (("entries.jsonl", entries_path), ("embeddings.npy", embeddings_path)):
        expected = files.get(name, {}).get("sha256") if isinstance(files.get(name), dict) else None
        if not isinstance(expected, str):
            raise ManifestValidationError(f"missing digest for {name}")
        try:
            actual = _sha256_file(path)
        except FileNotFoundError as exc:
            raise PayloadDigestMismatchError(f"missing payload {path}") from exc
        if actual != expected:
            raise PayloadDigestMismatchError(
                f"{name} digest mismatch: expected {expected}, got {actual}"
            )

    raw_entries = _load_jsonl(entries_path)
    ids, source_counts = _validate_raw_entries(raw_entries)
    entry_count = manifest.get("entry_count")
    if entry_count != len(raw_entries):
        raise EntryValidationError(
            f"entry count mismatch: manifest={entry_count!r}, JSONL={len(raw_entries)}"
        )
    if manifest.get("stable_id_digest") != _sha256_bytes(_canonical_json(sorted(ids))):
        raise PayloadDigestMismatchError("stable ID digest mismatch")
    if manifest.get("ordered_entry_digest") != _sha256_bytes(_canonical_json(raw_entries)):
        raise PayloadDigestMismatchError("ordered entry digest mismatch")

    sources = manifest.get("sources")
    if (
        not isinstance(sources, dict)
        or sources.get("counts") != source_counts
        or sources.get("ids") != sorted(source_counts)
    ):
        raise ManifestValidationError(
            f"source counts disagree with entries: manifest={sources!r}, actual={source_counts!r}"
        )
    if manifest.get("source_manifest_digest") != _sha256_bytes(_canonical_json(sources)):
        raise PayloadDigestMismatchError("source manifest digest mismatch")

    embedding = manifest.get("embedding")
    if not isinstance(embedding, dict):
        raise ManifestValidationError("embedding must be an object")
    model = embedding.get("model")
    dimension = embedding.get("dimension")
    if not isinstance(model, str) or not model.strip():
        raise ManifestValidationError("embedding.model must be nonblank")
    if not isinstance(dimension, int) or dimension < 0:
        raise ManifestValidationError("embedding.dimension must be a nonnegative integer")
    try:
        array = np.load(
            embeddings_path, mmap_mode="r" if mmap else None, allow_pickle=False
        )
    except Exception as exc:
        raise EmbeddingValidationError(f"cannot load embeddings.npy: {exc}") from exc
    expected_shape = [len(raw_entries), dimension]
    if embedding.get("shape") != expected_shape or list(array.shape) != expected_shape:
        raise EmbeddingValidationError(
            f"embedding shape mismatch: manifest={embedding.get('shape')!r}, "
            f"expected={expected_shape}, file={list(array.shape)}"
        )
    expected_dtype = embedding.get("dtype")
    if expected_dtype != "float32" or str(array.dtype) != expected_dtype:
        raise EmbeddingValidationError(
            f"embedding dtype mismatch: manifest={expected_dtype!r}, file={array.dtype}"
        )
    if not np.isfinite(array).all():
        raise EmbeddingValidationError("embeddings contain NaN or infinite values")

    recipes_value = manifest.get("recipes")
    if not isinstance(recipes_value, list):
        raise RecipeMismatchError("recipes must be a list")
    recipes = _validate_recipes(
        recipes_value, embedding_model=model, embedding_dim=dimension
    )
    recipe_ids = [recipe["recipe_id"] for recipe in recipes]
    if len(recipes) == 1:
        if manifest.get("recipe_id") != recipe_ids[0] or "recipe_ids" in manifest:
            raise RecipeMismatchError("homogeneous generation recipe identity disagrees")
    elif manifest.get("recipe_ids") != recipe_ids or "recipe_id" in manifest:
        raise RecipeMismatchError("mixed generation recipe set disagrees")
    _validate_entry_recipes(raw_entries, recipes)

    try:
        entries = [Entry.from_dict(row) for row in raw_entries]
    except (KeyError, TypeError, ValueError) as exc:
        raise EntryValidationError(f"entry schema failure: {exc}") from exc
    embeddings = np.asarray(array, dtype=np.float32)
    return Generation(generation_id, generation_path, manifest, entries, embeddings)


def _pointer(lane_root: Path) -> tuple[str, str]:
    pointer_path = lane_root / "current.json"
    try:
        pointer_bytes = pointer_path.read_bytes()
    except FileNotFoundError as exc:
        raise PointerNotFoundError(f"missing activation pointer {pointer_path}") from exc
    try:
        pointer = json.loads(pointer_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PointerMalformedError(f"malformed activation pointer: {exc}") from exc
    if not isinstance(pointer, dict) or set(pointer) != {
        "schema_version",
        "generation_id",
        "generation_manifest_sha256",
    }:
        raise PointerMalformedError(
            "current.json must contain only schema_version, generation_id, and "
            "generation_manifest_sha256"
        )
    if pointer.get("schema_version") != SCHEMA_VERSION:
        raise UnsupportedSchemaError(
            f"unsupported pointer schema {pointer.get('schema_version')!r}"
        )
    generation_id = _validate_generation_id(pointer.get("generation_id"))
    expected_digest = pointer.get("generation_manifest_sha256")
    if not isinstance(expected_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise PointerMalformedError("pointer manifest digest must be lowercase SHA-256")
    return generation_id, expected_digest


def read_generation(
    lane_root: Path,
    *,
    sources: Optional[Set[str]] = None,
    mmap: bool = True,
) -> tuple[list[Entry], np.ndarray, dict[str, Any]]:
    """Resolve the activation pointer once and read one validated generation."""

    lane_root = Path(lane_root)
    generation_id, expected_digest = _pointer(lane_root)
    generation_path = lane_root / "generations" / generation_id
    if generation_path.is_symlink() or not generation_path.is_dir():
        raise GenerationNotFoundError(
            f"active generation {generation_id!r} does not exist under {lane_root}"
        )
    generation = validate_generation(
        generation_path,
        expected_manifest_sha256=expected_digest,
        mmap=mmap,
    )
    if sources is None:
        return generation.entries, generation.embeddings, generation.manifest
    indices = [
        index for index, entry in enumerate(generation.entries) if entry.source in sources
    ]
    entries = [generation.entries[index] for index in indices]
    if indices:
        embeddings = np.asarray(generation.embeddings[indices], dtype=np.float32)
    else:
        dimension = generation.manifest["embedding"]["dimension"]
        embeddings = np.empty((0, dimension), dtype=np.float32)
    return entries, embeddings, generation.manifest


def activate_generation(lane_root: Path, generation_id: str) -> dict[str, Any]:
    """Atomically activate an already-complete, fully validated generation."""

    lane_root = Path(lane_root)
    generation_id = _validate_generation_id(generation_id)
    generation_path = lane_root / "generations" / generation_id
    if not generation_path.is_dir():
        raise GenerationNotFoundError(f"generation {generation_id!r} does not exist")
    generation = validate_generation(generation_path)
    manifest_sha256 = _sha256_file(generation_path / "generation.json")
    pointer = {
        "schema_version": SCHEMA_VERSION,
        "generation_id": generation.generation_id,
        "generation_manifest_sha256": manifest_sha256,
    }
    lane_root.mkdir(parents=True, exist_ok=True)
    temp_path = lane_root / f".current.{uuid.uuid4().hex}.tmp"
    with temp_path.open("wb") as handle:
        handle.write(_canonical_json(pointer) + b"\n")
        _fsync_file(handle)
    os.replace(temp_path, lane_root / "current.json")
    _fsync_directory(lane_root)
    return pointer


def write_generation(
    lane_root: Path,
    entries: Sequence[Entry | Mapping[str, Any]],
    embeddings: np.ndarray,
    *,
    generation_id: str,
    lane: str,
    embedding_model: str,
    recipes: Sequence[Mapping[str, Any]],
    generator: Mapping[str, Any],
    parent_generation: Optional[str] = None,
    source_manifest: Optional[Mapping[str, Any]] = None,
    migration: Optional[Mapping[str, Any]] = None,
    activate: bool = True,
) -> Generation:
    """Build, validate, publish, and optionally activate an immutable generation."""

    lane_root = Path(lane_root)
    generation_id = _validate_generation_id(generation_id)
    if not isinstance(lane, str) or not lane.strip():
        raise ManifestValidationError("lane must be nonblank")
    if parent_generation is not None:
        _validate_generation_id(parent_generation)
    raw_entries = [_entry_dict(entry) for entry in entries]
    ids, source_counts = _validate_raw_entries(raw_entries)
    matrix = np.asarray(embeddings)
    if matrix.ndim != 2:
        raise EmbeddingValidationError(
            f"embeddings must be a 2D matrix, got shape {matrix.shape}"
        )
    if matrix.shape[0] != len(raw_entries):
        raise EmbeddingValidationError(
            f"entry/vector count mismatch: {len(raw_entries)} vs {matrix.shape[0]}"
        )
    if not np.issubdtype(matrix.dtype, np.number):
        raise EmbeddingValidationError(f"embeddings must be numeric, got {matrix.dtype}")
    matrix = np.asarray(matrix, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise EmbeddingValidationError("embeddings contain NaN or infinite values")
    dimension = int(matrix.shape[1])
    normalized_recipes = _validate_recipes(
        recipes, embedding_model=embedding_model, embedding_dim=dimension
    )
    _validate_entry_recipes(raw_entries, normalized_recipes)
    generator_identity = dict(generator)
    for field in ("app_git_commit", "emb_git_commit", "generator_version"):
        if not isinstance(generator_identity.get(field), str) or not generator_identity[field].strip():
            raise ManifestValidationError(f"generator.{field} must be nonblank")

    generations_path = lane_root / "generations"
    generations_path.mkdir(parents=True, exist_ok=True)
    target_path = generations_path / generation_id
    staging_path = generations_path / f"{generation_id}.building"
    if target_path.exists() or staging_path.exists():
        raise GenerationExistsError(
            f"generation or staging directory already exists for {generation_id!r}"
        )
    staging_path.mkdir()
    if staging_path.stat().st_dev != generations_path.stat().st_dev:
        raise CrossFilesystemStagingError(
            "generation staging must be on the same filesystem as generations/"
        )

    entries_path = staging_path / "entries.jsonl"
    with entries_path.open("wb") as handle:
        for raw in raw_entries:
            handle.write(_canonical_json(raw) + b"\n")
        _fsync_file(handle)

    embeddings_path = staging_path / "embeddings.npy"
    with embeddings_path.open("wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
        _fsync_file(handle)

    sources: dict[str, Any] = {"ids": sorted(source_counts), "counts": source_counts}
    if source_manifest:
        for key, value in source_manifest.items():
            if key == "counts" and value != source_counts:
                raise ManifestValidationError(
                    f"provided source counts {value!r} disagree with entries {source_counts!r}"
                )
            if key == "ids" and value != sorted(source_counts):
                raise ManifestValidationError(
                    f"provided source IDs {value!r} disagree with entries "
                    f"{sorted(source_counts)!r}"
                )
            if key not in {"counts", "ids"}:
                sources[key] = value

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "lane": lane,
        "generation_id": generation_id,
        "parent_generation": parent_generation,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entry_count": len(raw_entries),
        "embedding": {
            "model": embedding_model,
            "dimension": dimension,
            "dtype": "float32",
            "shape": [len(raw_entries), dimension],
        },
        "recipes": normalized_recipes,
        "sources": sources,
        "source_manifest_digest": _sha256_bytes(_canonical_json(sources)),
        "stable_id_digest": _sha256_bytes(_canonical_json(sorted(ids))),
        "ordered_entry_digest": _sha256_bytes(_canonical_json(raw_entries)),
        "files": {
            "entries.jsonl": {"sha256": _sha256_file(entries_path)},
            "embeddings.npy": {"sha256": _sha256_file(embeddings_path)},
        },
        "generator": generator_identity,
    }
    recipe_ids = [recipe["recipe_id"] for recipe in normalized_recipes]
    if len(recipe_ids) == 1:
        manifest["recipe_id"] = recipe_ids[0]
    else:
        manifest["recipe_ids"] = recipe_ids
    if migration is not None:
        manifest["migration"] = dict(migration)

    manifest_path = staging_path / "generation.json"
    with manifest_path.open("wb") as handle:
        handle.write(_canonical_json(manifest) + b"\n")
        _fsync_file(handle)
    _fsync_directory(staging_path)

    validate_generation(staging_path, mmap=False)
    os.replace(staging_path, target_path)
    _fsync_directory(generations_path)
    if activate:
        activate_generation(lane_root, generation_id)
    return validate_generation(target_path, mmap=True)


def default_text_recipe(model: str, dimension: int) -> dict[str, Any]:
    """Return the explicit recipe used by the stock sentence-transformer CLI."""

    return {
        "recipe_id": f"sentence-transformers:{model}:retrieval-document:v1",
        "model": model,
        "dimension": dimension,
        "backend": "sentence-transformers",
        "task_type": "retrieval_document",
        "normalization": "l2",
        "input_modality": "text",
        "caption_inclusion": False,
        "version": 1,
    }
