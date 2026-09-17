from dataclasses import dataclass, field
from typing import Any, Dict, Tuple
import numpy as np


@dataclass
class Observation:
    t: float
    z: float
    source: str | None = None
    variance: float | None = None
    quality: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PosteriorSummary:
    x_mean: np.ndarray
    x_cov: np.ndarray
    y_mean: float
    y_var: float
    innovation: float
    innovation_var: float
    loglik: float
    ci68: Tuple[float, float]
    ci95: Tuple[float, float]
    probability_of_event: float
    noise_velocity: float
    dt: float
    mode: str
    diag: Dict[str, Any]
