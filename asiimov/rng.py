"""Counter-based RNG.

Every draw is a pure function of (seed, stream, index), so a chance outcome can be
recomputed from its log entry alone. Only integer arithmetic is used, which keeps
draws identical across platforms and Python versions.
"""
import hashlib


class Rng:
    def __init__(self, seed: int):
        self.seed = seed
        self.counters: dict[str, int] = {}

    def raw(self, stream: str, index: int) -> int:
        digest = hashlib.sha256(f"{self.seed}|{stream}|{index}".encode()).digest()
        return int.from_bytes(digest[:8], "big")

    def draw(self, stream: str, lo: int, hi: int) -> tuple[int, int]:
        """Return (index, value) with lo <= value <= hi, advancing the stream."""
        if hi < lo:
            raise ValueError("empty range")
        index = self.counters.get(stream, 0)
        self.counters[stream] = index + 1
        return index, lo + self.raw(stream, index) % (hi - lo + 1)

    def normal_milli(self, stream: str, sd: int) -> int:
        """Approximate N(0, sd) in integers (Irwin-Hall sum of 12 uniforms)."""
        total = sum(self.draw(stream, 0, 9999)[1] for _ in range(12))
        return (total - 60000) * sd // 10000
