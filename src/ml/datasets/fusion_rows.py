"""Join OOF predictions by explicit row identities, never by accidental array length."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.domain.features import PillarType
from src.ml.datasets.oof import OOFResult


@dataclass(frozen=True, slots=True)
class FusionRows:
    row_indices: tuple[int, ...]
    pillar_probabilities: Mapping[PillarType, NDArray[np.float64]]
    labels: NDArray[np.int64]


def assemble_fusion_rows(
    results: Mapping[PillarType, OOFResult], labels: NDArray[np.int64]
) -> FusionRows:
    if not results:
        raise ValueError("No OOF expert results")
    common = set.intersection(*(set(result.row_indices) for result in results.values()))
    if not common:
        raise ValueError("No common eligible OOF rows")
    rows = tuple(sorted(common))
    values = {}
    for pillar, result in results.items():
        offsets = {row: index for index, row in enumerate(result.row_indices)}
        values[pillar] = result.probabilities[[offsets[row] for row in rows]].copy()
        values[pillar].setflags(write=False)
    outcomes = labels[np.asarray(rows, dtype=np.int64)].copy()
    outcomes.setflags(write=False)
    return FusionRows(rows, values, outcomes)
