from emb.chunking import chunk_text, chunk_entries
from emb.schema import Entry

def test_short_text_no_chunking():
    chunks = chunk_text("Hello world. This is short.", chunk_tokens=500)
    assert len(chunks) == 1
    assert chunks[0] == "Hello world. This is short."

def test_long_text_chunked():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, chunk_tokens=500, overlap_tokens=50)
    assert len(chunks) >= 2
    for c in chunks:
        words = c.split()
        assert len(words) <= 600  # allow some tolerance

def test_sentence_boundary_respected():
    sentences = ["This is sentence one."] * 250
    text = " ".join(sentences)
    chunks = chunk_text(text, chunk_tokens=200, overlap_tokens=20)
    assert len(chunks) >= 2
    # Each chunk except possibly the last should end with a period
    for c in chunks[:-1]:
        assert c.rstrip().endswith(".")

def test_overlap_exists():
    text = " ".join(f"unique{i}." for i in range(500))
    chunks = chunk_text(text, chunk_tokens=100, overlap_tokens=20)
    assert len(chunks) >= 3
    # Check that consecutive chunks share some content
    for i in range(len(chunks) - 1):
        words_a = set(chunks[i].split()[-30:])
        words_b = set(chunks[i+1].split()[:30])
        overlap = words_a & words_b
        assert len(overlap) > 0, f"No overlap between chunks {i} and {i+1}"

def test_chunk_entries_creates_child_entries():
    entry = Entry(
        id="podcast_abc",
        text=" ".join(f"word{i}." for i in range(1000)),
        source="podcast",
        title="Episode 42",
        date="2024-01-01",
        metadata={"channel": "test"}
    )
    chunked = chunk_entries([entry], chunk_tokens=200, overlap_tokens=20)
    assert len(chunked) > 1
    for i, c in enumerate(chunked):
        assert c.id == f"podcast_abc__chunk_{i}"
        assert c.metadata.get("parent_id") == "podcast_abc"
        assert c.metadata.get("chunk_index") == i
        assert c.metadata.get("total_chunks") == len(chunked)
        assert c.source == "podcast"
        assert c.date == "2024-01-01"

def test_short_entry_passes_through():
    entry = Entry(id="note_1", text="Short note.")
    chunked = chunk_entries([entry], chunk_tokens=500)
    assert len(chunked) == 1
    assert chunked[0].id == "note_1"
    assert "parent_id" not in chunked[0].metadata

def test_no_text_entry_passes_through():
    entry = Entry(id="empty", text="")
    chunked = chunk_entries([entry], chunk_tokens=500)
    assert len(chunked) == 1

def test_fallback_to_word_splitting():
    """Text with no sentence boundaries should still chunk."""
    text = " ".join(f"word{i}" for i in range(1000))  # no periods
    chunks = chunk_text(text, chunk_tokens=300, overlap_tokens=30)
    assert len(chunks) >= 2
