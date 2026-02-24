"""Tests for CLI commands (version, info, merge).

embed and search require model loading, so they are integration-level only.
"""
import json
from typer.testing import CliRunner
from emb.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "emb" in result.output
    assert "0.1.0" in result.output


def test_no_command_shows_help():
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "embed" in result.output or "Embed" in result.output


def test_info(tmp_path):
    idx = tmp_path / "test.json"
    idx.write_text(json.dumps({
        'metadata': {
            'embedding_model': 'test-model',
            'embedding_dim': 4,
            'generated_at': '2025-01-01T00:00:00',
            'sources': {'chatgpt': 3, 'git': 2},
        },
        'entries': [
            {'id': f'e{i}', 'text': f't{i}', 'embedding': [0] * 4}
            for i in range(5)
        ],
    }))
    result = runner.invoke(app, ["info", str(idx)])
    assert result.exit_code == 0
    assert "5" in result.output  # entry count
    assert "test-model" in result.output
    assert "chatgpt" in result.output
    assert "git" in result.output


def test_info_minimal_metadata(tmp_path):
    """Info should handle indices with minimal metadata gracefully."""
    idx = tmp_path / "test.json"
    idx.write_text(json.dumps({
        'metadata': {},
        'entries': [{'id': 'a', 'text': 'hello', 'embedding': [0, 0]}],
    }))
    result = runner.invoke(app, ["info", str(idx)])
    assert result.exit_code == 0
    assert "1" in result.output  # entry count
    assert "unknown" in result.output  # missing model


def test_merge(tmp_path):
    idx1 = tmp_path / "a.json"
    idx2 = tmp_path / "b.json"
    out = tmp_path / "merged.json"

    idx1.write_text(json.dumps({
        'metadata': {'embedding_model': 'test', 'embedding_dim': 4},
        'entries': [{'id': 'e1', 'text': 't1', 'source': 'a', 'embedding': [0] * 4}],
    }))
    idx2.write_text(json.dumps({
        'metadata': {'embedding_model': 'test', 'embedding_dim': 4},
        'entries': [{'id': 'e2', 'text': 't2', 'source': 'b', 'embedding': [0] * 4}],
    }))

    result = runner.invoke(app, ["merge", str(idx1), str(idx2), "-o", str(out)])
    assert result.exit_code == 0

    with open(out) as f:
        merged = json.load(f)
    assert merged['metadata']['total_entries'] == 2
    assert len(merged['entries']) == 2
    assert merged['metadata']['embedding_model'] == 'test'
    assert merged['metadata']['embedding_dim'] == 4


def test_merge_deduplicates(tmp_path):
    idx1 = tmp_path / "a.json"
    idx2 = tmp_path / "b.json"
    out = tmp_path / "merged.json"

    # Same id in both
    idx1.write_text(json.dumps({
        'metadata': {'embedding_model': 'test', 'embedding_dim': 4},
        'entries': [{'id': 'same', 'text': 't1', 'embedding': [0] * 4}],
    }))
    idx2.write_text(json.dumps({
        'metadata': {'embedding_model': 'test', 'embedding_dim': 4},
        'entries': [{'id': 'same', 'text': 't2', 'embedding': [0] * 4}],
    }))

    result = runner.invoke(app, ["merge", str(idx1), str(idx2), "-o", str(out)])
    assert result.exit_code == 0

    with open(out) as f:
        merged = json.load(f)
    assert len(merged['entries']) == 1
    assert merged['metadata']['total_entries'] == 1


def test_merge_preserves_first_on_dedup(tmp_path):
    """When deduplicating, keep the first occurrence."""
    idx1 = tmp_path / "a.json"
    idx2 = tmp_path / "b.json"
    out = tmp_path / "merged.json"

    idx1.write_text(json.dumps({
        'metadata': {'embedding_model': 'test', 'embedding_dim': 4},
        'entries': [{'id': 'dup', 'text': 'first version', 'embedding': [1, 0, 0, 0]}],
    }))
    idx2.write_text(json.dumps({
        'metadata': {'embedding_model': 'test', 'embedding_dim': 4},
        'entries': [{'id': 'dup', 'text': 'second version', 'embedding': [0, 1, 0, 0]}],
    }))

    result = runner.invoke(app, ["merge", str(idx1), str(idx2), "-o", str(out)])
    assert result.exit_code == 0

    with open(out) as f:
        merged = json.load(f)
    assert merged['entries'][0]['text'] == 'first version'


def test_merge_records_source_files(tmp_path):
    idx1 = tmp_path / "a.json"
    idx2 = tmp_path / "b.json"
    out = tmp_path / "merged.json"

    idx1.write_text(json.dumps({
        'metadata': {'embedding_model': 'test', 'embedding_dim': 4},
        'entries': [{'id': 'e1', 'text': 't1', 'embedding': [0] * 4}],
    }))
    idx2.write_text(json.dumps({
        'metadata': {'embedding_model': 'test', 'embedding_dim': 4},
        'entries': [{'id': 'e2', 'text': 't2', 'embedding': [0] * 4}],
    }))

    result = runner.invoke(app, ["merge", str(idx1), str(idx2), "-o", str(out)])
    assert result.exit_code == 0

    with open(out) as f:
        merged = json.load(f)
    assert 'merged_from' in merged['metadata']
    assert len(merged['metadata']['merged_from']) == 2


def test_merge_model_mismatch_warning(tmp_path):
    """Merge should warn when models differ but still proceed."""
    idx1 = tmp_path / "a.json"
    idx2 = tmp_path / "b.json"
    out = tmp_path / "merged.json"

    idx1.write_text(json.dumps({
        'metadata': {'embedding_model': 'model-a', 'embedding_dim': 4},
        'entries': [{'id': 'e1', 'text': 't1', 'embedding': [0] * 4}],
    }))
    idx2.write_text(json.dumps({
        'metadata': {'embedding_model': 'model-b', 'embedding_dim': 4},
        'entries': [{'id': 'e2', 'text': 't2', 'embedding': [0] * 4}],
    }))

    result = runner.invoke(app, ["merge", str(idx1), str(idx2), "-o", str(out)])
    assert result.exit_code == 0
    assert "mismatch" in result.output.lower() or "Warning" in result.output

    # Should still produce output
    with open(out) as f:
        merged = json.load(f)
    assert len(merged['entries']) == 2


def test_merge_source_counts(tmp_path):
    """Source counts in metadata should reflect all entries before dedup."""
    idx1 = tmp_path / "a.json"
    idx2 = tmp_path / "b.json"
    out = tmp_path / "merged.json"

    idx1.write_text(json.dumps({
        'metadata': {'embedding_model': 'test', 'embedding_dim': 4},
        'entries': [
            {'id': 'e1', 'text': 't1', 'source': 'chatgpt', 'embedding': [0] * 4},
            {'id': 'e2', 'text': 't2', 'source': 'git', 'embedding': [0] * 4},
        ],
    }))
    idx2.write_text(json.dumps({
        'metadata': {'embedding_model': 'test', 'embedding_dim': 4},
        'entries': [
            {'id': 'e3', 'text': 't3', 'source': 'chatgpt', 'embedding': [0] * 4},
        ],
    }))

    result = runner.invoke(app, ["merge", str(idx1), str(idx2), "-o", str(out)])
    assert result.exit_code == 0

    with open(out) as f:
        merged = json.load(f)
    assert merged['metadata']['sources']['chatgpt'] == 2
    assert merged['metadata']['sources']['git'] == 1


def test_merge_creates_output_directory(tmp_path):
    """Merge should create parent dirs for output if needed."""
    idx1 = tmp_path / "a.json"
    out = tmp_path / "subdir" / "deep" / "merged.json"

    idx1.write_text(json.dumps({
        'metadata': {'embedding_model': 'test', 'embedding_dim': 4},
        'entries': [{'id': 'e1', 'text': 't1', 'embedding': [0] * 4}],
    }))

    result = runner.invoke(app, ["merge", str(idx1), "-o", str(out)])
    assert result.exit_code == 0
    assert out.exists()


def test_search_no_query_no_interactive_fails(tmp_path):
    """search without query or --interactive should fail."""
    idx = tmp_path / "test.json"
    idx.write_text(json.dumps({
        'metadata': {'embedding_model': 'test', 'embedding_dim': 4},
        'entries': [{'id': 'a', 'text': 'hello', 'embedding': [0] * 4}],
    }))
    result = runner.invoke(app, ["search", str(idx)])
    assert result.exit_code == 1
