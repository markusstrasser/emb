"""Tests for JSONL reading/writing utilities."""
import json
import io
from emb.io import read_jsonl, write_jsonl
from emb.schema import Entry


def test_read_jsonl_from_file(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text('{"id": "a", "text": "hello"}\n{"id": "b", "text": "world"}\n')
    entries = read_jsonl(str(f))
    assert len(entries) == 2
    assert entries[0].id == "a"
    assert entries[0].text == "hello"
    assert entries[1].id == "b"
    assert entries[1].text == "world"


def test_read_jsonl_with_all_fields(tmp_path):
    f = tmp_path / "test.jsonl"
    line = json.dumps({
        "id": "x", "text": "hello", "source": "chatgpt",
        "title": "Test", "date": "2024-01-01",
        "metadata": {"key": "val"}
    })
    f.write_text(line + "\n")
    entries = read_jsonl(str(f))
    assert len(entries) == 1
    assert entries[0].source == "chatgpt"
    assert entries[0].title == "Test"
    assert entries[0].date == "2024-01-01"
    assert entries[0].metadata == {"key": "val"}


def test_read_jsonl_skips_invalid_json(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text('{"id": "a", "text": "ok"}\nnot json\n{"id": "c", "text": "also ok"}\n')
    entries = read_jsonl(str(f))
    assert len(entries) == 2
    assert entries[0].id == "a"
    assert entries[1].id == "c"


def test_read_jsonl_skips_missing_id(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text('{"id": "a", "text": "ok"}\n{"text": "no id"}\n')
    entries = read_jsonl(str(f))
    assert len(entries) == 1
    assert entries[0].id == "a"


def test_read_jsonl_skips_missing_text(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text('{"id": "a", "text": "ok"}\n{"id": "b"}\n')
    entries = read_jsonl(str(f))
    assert len(entries) == 1


def test_read_jsonl_skips_empty_lines(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text('\n{"id": "a", "text": "ok"}\n\n{"id": "b", "text": "ok2"}\n\n')
    entries = read_jsonl(str(f))
    assert len(entries) == 2


def test_read_jsonl_from_file_like_object():
    data = '{"id": "a", "text": "hello"}\n{"id": "b", "text": "world"}\n'
    fh = io.StringIO(data)
    entries = read_jsonl(fh)
    assert len(entries) == 2
    assert entries[0].id == "a"


def test_read_jsonl_empty_file(tmp_path):
    f = tmp_path / "empty.jsonl"
    f.write_text("")
    entries = read_jsonl(str(f))
    assert entries == []


def test_read_jsonl_path_object(tmp_path):
    """read_jsonl should accept pathlib.Path objects."""
    f = tmp_path / "test.jsonl"
    f.write_text('{"id": "a", "text": "hello"}\n')
    entries = read_jsonl(f)
    assert len(entries) == 1


def test_write_jsonl_to_file(tmp_path):
    f = tmp_path / "out.jsonl"
    entries = [Entry(id="x", text="hello", source="test")]
    count = write_jsonl(entries, str(f))
    assert count == 1
    lines = f.read_text().strip().split('\n')
    assert len(lines) == 1
    d = json.loads(lines[0])
    assert d['id'] == 'x'
    assert d['text'] == 'hello'
    assert d['source'] == 'test'


def test_write_jsonl_multiple_entries(tmp_path):
    f = tmp_path / "out.jsonl"
    entries = [
        Entry(id="a", text="first"),
        Entry(id="b", text="second", source="git"),
        Entry(id="c", text="third", title="Title C"),
    ]
    count = write_jsonl(entries, str(f))
    assert count == 3
    lines = f.read_text().strip().split('\n')
    assert len(lines) == 3
    assert json.loads(lines[0])['id'] == 'a'
    assert json.loads(lines[1])['source'] == 'git'
    assert json.loads(lines[2])['title'] == 'Title C'


def test_write_jsonl_to_file_like_object():
    fh = io.StringIO()
    entries = [Entry(id="a", text="hello")]
    count = write_jsonl(entries, fh)
    assert count == 1
    output = fh.getvalue()
    d = json.loads(output.strip())
    assert d['id'] == 'a'


def test_write_jsonl_empty_list(tmp_path):
    f = tmp_path / "out.jsonl"
    count = write_jsonl([], str(f))
    assert count == 0
    assert f.read_text() == ""


def test_write_jsonl_excludes_none_fields(tmp_path):
    f = tmp_path / "out.jsonl"
    entries = [Entry(id="x", text="hello")]  # source, title, date are None
    write_jsonl(entries, str(f))
    d = json.loads(f.read_text().strip())
    assert 'source' not in d
    assert 'title' not in d
    assert 'date' not in d
    assert 'embedding' not in d


def test_roundtrip(tmp_path):
    """Write then read should preserve data."""
    original = [
        Entry(id="a", text="hello", source="test", title="T", date="2024-01-01"),
        Entry(id="b", text="world", metadata={"k": "v"}),
    ]
    f = tmp_path / "roundtrip.jsonl"
    write_jsonl(original, str(f))
    loaded = read_jsonl(str(f))
    assert len(loaded) == 2
    assert loaded[0].id == "a"
    assert loaded[0].source == "test"
    assert loaded[0].title == "T"
    assert loaded[1].metadata == {"k": "v"}
