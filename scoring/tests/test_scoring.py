"""Unit tests for the shared scoring package."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scoring.api import score, scientific_scores, sports_scores
from scoring.columns import points_col
from scoring.mercier import mercier_points
from scoring.purdy import purdy_from_distance_time, purdy_points
from scoring.vdot import vdot_from_distance_time, vdot_points
from scoring.wa import load_coefficients, wa_points


class TestWA(unittest.TestCase):
    def test_club_100m_men(self):
        coeffs = load_coefficients()
        pts = wa_points(coeffs, "men", "100m", 11.50)
        # 2025 outdoor tables; legacy NRCD CSVs may differ by a few points
        self.assertIsNotNone(pts)
        self.assertTrue(730 <= pts <= 760)

    def test_women_100mh_mapping(self):
        coeffs = load_coefficients()
        pts = wa_points(coeffs, "women", "100mH", 14.50)
        self.assertIsNotNone(pts)
        self.assertGreater(pts, 0)


class TestVDOT(unittest.TestCase):
    def test_1500_reasonable(self):
        # ~4:00 1500m should be mid-60s VDOT range for club/college
        v = vdot_from_distance_time(1500, 240.0)
        self.assertIsNotNone(v)
        self.assertTrue(55 < v < 80)

    def test_field_none(self):
        self.assertIsNone(vdot_points("men", "LJ", 7.0))


class TestPurdy(unittest.TestCase):
    def test_950ish_at_portuguese_100(self):
        # Portuguese 100m speed 10.86 m/s → ~9.208 s straight; with slowdown ~950 pts
        pts = purdy_from_distance_time(100.0, 10.5)
        self.assertIsNotNone(pts)
        self.assertGreater(pts, 800)

    def test_field_none(self):
        self.assertIsNone(purdy_points("men", "SP", 15.0))


class TestMercier(unittest.TestCase):
    def test_greene_wr(self):
        self.assertEqual(mercier_points("men", "100m", 9.79), 1015)

    def test_flojo_wr(self):
        self.assertEqual(mercier_points("women", "100m", 10.49), 1077)

    def test_powell_lj(self):
        self.assertEqual(mercier_points("men", "LJ", 8.95), 1069)


class TestAPI(unittest.TestCase):
    def test_score_keys(self):
        out = score(11.50, "100m", "men")
        self.assertEqual(set(out), {"wa", "vdot", "purdy", "mercier"})
        self.assertIsNotNone(out["wa"])
        self.assertTrue(730 <= out["wa"] <= 760)
        self.assertIsNotNone(out["vdot"])
        self.assertIsNotNone(out["purdy"])
        self.assertIsNotNone(out["mercier"])

    def test_framing_helpers(self):
        sci = scientific_scores(11.50, "100m", "men")
        spo = sports_scores(11.50, "100m", "men")
        self.assertEqual(set(sci), {"purdy", "mercier"})
        self.assertEqual(set(spo), {"wa", "vdot"})

    def test_points_col(self):
        self.assertEqual(points_col("Men", "wa"), "World_Athletics_Points_Men")
        self.assertEqual(points_col("Women", "vdot"), "VDOT_Women")
        self.assertEqual(points_col("men", "purdy"), "Purdy_Points_Men")
        self.assertEqual(points_col("w", "mercier"), "Mercier_Points_Women")


if __name__ == "__main__":
    unittest.main()
