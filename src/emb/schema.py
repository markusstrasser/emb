"""Entry schema for emb indices.

Required fields: id, text
Optional first-class fields: source, title, date (used by search features)
Everything else: metadata dict (passthrough)
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class Entry:
    id: str
    text: str
    source: Optional[str] = None
    title: Optional[str] = None
    date: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_hash: Optional[str] = None
    embedding: Optional[List[float]] = None

    def to_dict(self) -> dict:
        """Convert to dict, excluding None fields."""
        d = {"id": self.id, "text": self.text}
        if self.source is not None:
            d["source"] = self.source
        if self.title is not None:
            d["title"] = self.title
        if self.date is not None:
            d["date"] = self.date
        if self.metadata:
            d["metadata"] = self.metadata
        if self.content_hash is not None:
            d["content_hash"] = self.content_hash
        if self.embedding is not None:
            d["embedding"] = self.embedding
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Entry":
        """Create Entry from dict, ignoring unknown fields."""
        return cls(
            id=d["id"],
            text=d["text"],
            source=d.get("source"),
            title=d.get("title"),
            date=d.get("date"),
            metadata=d.get("metadata", {}),
            content_hash=d.get("content_hash"),
            embedding=d.get("embedding"),
        )


@dataclass
class Index:
    entries: List[Entry]
    metadata: Dict[str, Any] = field(default_factory=dict)


def validate_entry(d: dict) -> bool:
    """Check that a dict has required entry fields."""
    return bool(d.get("id") and d.get("text"))
