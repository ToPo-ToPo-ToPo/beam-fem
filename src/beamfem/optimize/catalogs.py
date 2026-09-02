"""離散最適化で使用する断面カタログ（SI単位）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..material import Material, Section


@dataclass(frozen=True)
class SectionOption:
    """カタログ内の1候補。

    ``section=None`` は部材を配置しない OFF 状態を表す。強度は Pa、密度は
    ``material.rho`` [kg/m3] とし、構造解析本体と同じSI単位を用いる。
    """

    name: str
    section: Section | None
    material: Material | None = None
    tensile_strength: float | None = None
    compressive_strength: float | None = None
    cost_per_kg: float = 0.0
    carbon_per_kg: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("断面候補名は空にできません")
        if self.section is None and self.material is not None:
            raise ValueError("OFF候補に材料は指定できません")
        for label, value in (
            ("tensile_strength", self.tensile_strength),
            ("compressive_strength", self.compressive_strength),
        ):
            if value is not None and value <= 0.0:
                raise ValueError(f"{label} は正値でなければなりません")

    @property
    def active(self) -> bool:
        return self.section is not None


@dataclass(frozen=True)
class SectionCatalog:
    """名称付き断面候補の不変カタログ。"""

    name: str
    options: tuple[SectionOption, ...]

    def __init__(self, name: str, options: Iterable[SectionOption]):
        opts = tuple(options)
        if not opts:
            raise ValueError("断面カタログは1候補以上必要です")
        names = [o.name for o in opts]
        if len(set(names)) != len(names):
            raise ValueError("断面候補名はカタログ内で一意でなければなりません")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "options", opts)

    def __len__(self) -> int:
        return len(self.options)

    def __getitem__(self, index: int) -> SectionOption:
        return self.options[index]

    def index(self, name: str) -> int:
        for i, option in enumerate(self.options):
            if option.name == name:
                return i
        raise KeyError(f"断面候補 {name!r} はカタログ {self.name!r} にありません")

    @property
    def off_index(self) -> int | None:
        return next((i for i, o in enumerate(self.options) if not o.active), None)
