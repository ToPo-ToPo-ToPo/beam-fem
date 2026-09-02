"""Encodings between discrete structural states and QUBO bits."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class OneHotEncoding:
    domain_sizes: tuple[int, ...]

    def __post_init__(self):
        if not self.domain_sizes or any(int(size) < 1 for size in self.domain_sizes):
            raise ValueError("all domain sizes must be positive")

    @property
    def n_bits(self) -> int:
        return sum(self.domain_sizes)

    @property
    def slices(self) -> tuple[slice, ...]:
        result, start = [], 0
        for size in self.domain_sizes:
            result.append(slice(start, start + size)); start += size
        return tuple(result)
    def encode(self, states: Sequence[int]) -> tuple[int, ...]:
        if len(states) != len(self.domain_sizes):
            raise ValueError("state length does not match encoding")
        bits = [0] * self.n_bits
        for state, region, size in zip(states, self.slices, self.domain_sizes):
            if not 0 <= int(state) < size:
                raise ValueError("state outside domain")
            bits[region.start + int(state)] = 1
        return tuple(bits)

    def decode(self, bits: Sequence[int], repair: bool = True) -> tuple[int, ...]:
        x = np.asarray(bits, dtype=int)
        if x.shape != (self.n_bits,):
            raise ValueError("bit length does not match encoding")
        states = []
        for region in self.slices:
            active = np.flatnonzero(x[region])
            if len(active) != 1 and not repair:
                raise ValueError("invalid one-hot sample")
            states.append(int(active[0]) if len(active) else 0)
        return tuple(states)

    def constraint_penalty(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return coefficients for sum_g (sum(x_g)-1)^2."""
        linear = np.zeros(self.n_bits)
        quadratic = np.zeros((self.n_bits, self.n_bits))
        constant = float(len(self.domain_sizes))
        for region in self.slices:
            indices = range(region.start, region.stop)
            for i in indices:
                linear[i] -= 1.0
                for j in range(i + 1, region.stop):
                    quadratic[i, j] += 2.0
        return linear, quadratic, constant


@dataclass(frozen=True)
class BinaryEncoding:
    domain_sizes: tuple[int, ...]

    @property
    def widths(self) -> tuple[int, ...]:
        return tuple(max(1, ceil(log2(size))) for size in self.domain_sizes)

    @property
    def n_bits(self) -> int:
        return sum(self.widths)

    def encode(self, states: Sequence[int]) -> tuple[int, ...]:
        bits = []
        for state, size, width in zip(states, self.domain_sizes, self.widths):
            if not 0 <= int(state) < size:
                raise ValueError("state outside domain")
            bits.extend((int(state) >> bit) & 1 for bit in range(width))
        return tuple(bits)

    def decode(self, bits: Sequence[int], repair: bool = True) -> tuple[int, ...]:
        if len(bits) != self.n_bits:
            raise ValueError("bit length does not match encoding")
        result, start = [], 0
        for size, width in zip(self.domain_sizes, self.widths):
            value = sum(int(bits[start + bit]) << bit for bit in range(width))
            if value >= size:
                if not repair:
                    raise ValueError("binary sample maps outside domain")
                value = size - 1
            result.append(value); start += width
        return tuple(result)
