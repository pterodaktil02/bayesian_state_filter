from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import random
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "bayesian_state_filter"
    / "core"
    / "noise_detection.py"
)
SPEC = importlib.util.spec_from_file_location("bsf_noise_detection_test", MODULE_PATH)
noise_detection = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(noise_detection)


def _poisson_sample(rng: random.Random, lam: float) -> int:
    limit = math.exp(-lam)
    product = 1.0
    k = 0
    while product > limit:
        k += 1
        product *= rng.random()
    return k - 1


class NoiseDetectionTests(unittest.TestCase):
    def test_stationary_quantized_scaled_poisson(self):
        rng = random.Random(12345)
        # Mean ~= 14, variance scale ~= 0.35, quantized to integer output.
        # The p10-p90 span stays below the 8-bin broad-span requirement, so
        # classification must come from the stationary moment fallback.
        values = [
            float(round(0.35 * _poisson_sample(rng, 40.0)))
            for _ in range(6000)
        ]
        result = noise_detection.detect_noise_model(
            {"radiation": [(float(i), v) for i, v in enumerate(values)]}
        )
        self.assertEqual(result.family, "poisson")
        self.assertEqual(result.reason, "stationary_scaled_poisson")
        self.assertEqual(result.quantization_step, 1.0)
        self.assertGreater(result.confidence, 0.70)
        self.assertIsNotNone(result.poisson_scale)
        self.assertGreater(result.poisson_scale, 0.20)
        self.assertLess(result.poisson_scale, 0.50)

    def test_stationary_quantized_gaussian_is_not_poisson(self):
        rng = random.Random(54321)
        values = [float(round(200.0 + rng.gauss(0.0, 2.0))) for _ in range(6000)]
        result = noise_detection.detect_noise_model(
            {"pressure": [(float(i), v) for i, v in enumerate(values)]}
        )
        self.assertEqual(result.family, "gaussian")
        self.assertEqual(result.quantization_step, 1.0)
        self.assertNotEqual(result.reason, "stationary_scaled_poisson")

    def test_negative_quantized_signal_is_not_count_like(self):
        values = [float((i % 17) - 3) for i in range(800)]
        result = noise_detection.detect_noise_model(
            {"pressure": [(float(i), v) for i, v in enumerate(values)]}
        )
        self.assertEqual(result.family, "gaussian")
        self.assertEqual(result.reason, "not_count_like")


if __name__ == "__main__":
    unittest.main()
