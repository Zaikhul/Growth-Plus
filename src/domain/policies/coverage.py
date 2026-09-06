"""Exact input-mask contracts, independently of model/publication approval."""

from collections.abc import Iterable

from src.domain.errors import DomainError
from src.domain.features import PillarType, SourceCoverageMode


class UnsupportedSourceMaskError(DomainError):
    def __init__(self) -> None:
        super().__init__(
            message="No approved input mode for this source mask",
            code="UNSUPPORTED_SOURCE_MASK",
            details={"status": "UNAVAILABLE", "label": None, "confidence": None},
        )


def resolve_coverage_mode(pillars: Iterable[PillarType]) -> SourceCoverageMode:
    mask = frozenset(pillars)
    technical_macro = frozenset((PillarType.TECHNICAL, PillarType.MACRO))
    modes = {
        technical_macro: SourceCoverageMode.TECH_MACRO,
        technical_macro | {PillarType.NEWS}: SourceCoverageMode.CORE_NO_ETF,
        technical_macro | {PillarType.NEWS, PillarType.ETF}: SourceCoverageMode.FULL,
    }
    if mask not in modes:
        raise UnsupportedSourceMaskError()
    return modes[mask]
