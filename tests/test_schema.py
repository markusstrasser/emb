from emb.schema import Entry, Index, validate_entry

def test_minimal_entry():
    e = Entry(id="test_1", text="Hello world")
    assert e.id == "test_1"
    assert e.text == "Hello world"
    assert e.source is None
    assert e.title is None
    assert e.date is None
    assert e.metadata == {}
    assert e.embedding is None

def test_full_entry():
    e = Entry(
        id="chatgpt_123",
        text="How do I use React hooks?",
        source="chatgpt",
        title="React Hooks Discussion",
        date="2024-03-15",
        metadata={"conversation_id": "abc"}
    )
    assert e.source == "chatgpt"
    assert e.metadata["conversation_id"] == "abc"

def test_validate_entry_requires_id_and_text():
    assert validate_entry({"id": "x", "text": "y"}) is True
    assert validate_entry({"id": "x"}) is False
    assert validate_entry({"text": "y"}) is False
    assert validate_entry({}) is False

def test_entry_to_dict():
    e = Entry(id="x", text="hello", source="test")
    d = e.to_dict()
    assert d["id"] == "x"
    assert d["text"] == "hello"
    assert d["source"] == "test"
    assert "embedding" not in d  # None fields excluded

def test_entry_from_dict():
    d = {"id": "x", "text": "hello", "source": "test", "extra_field": "ignored"}
    e = Entry.from_dict(d)
    assert e.id == "x"
    assert e.source == "test"

def test_index_metadata():
    idx = Index(
        entries=[],
        metadata={"embedding_model": "test", "embedding_dim": 768}
    )
    assert idx.metadata["embedding_dim"] == 768
