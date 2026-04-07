from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecm_organoid_agent.formulation import (
    recommend_campaign_formulations,
    recommend_formulation,
    recommend_formulations_from_design_payload,
)


class FormulationMappingTests(unittest.TestCase):
    def test_recommend_formulation_returns_recipe_template(self) -> None:
        candidate = {
            "rank": 1,
            "score": 0.25,
            "feasible": True,
            "parameters": {
                "fiber_density": 0.42,
                "fiber_stiffness": 10.5,
                "bending_stiffness": 0.24,
                "crosslink_prob": 0.66,
                "domain_size": 1.0,
            },
            "features": {
                "stiffness_mean": 8.0,
                "anisotropy": 0.22,
                "connectivity": 1.0,
                "stress_propagation": 0.86,
                "risk_index": 0.42,
            },
        }
        recommendation = recommend_formulation(candidate)
        self.assertEqual(recommendation["candidate_rank"], 1)
        self.assertIn("material_family", recommendation)
        self.assertIn("primary_recipe", recommendation)
        self.assertIn("polymer_wt_percent", recommendation["primary_recipe"])
        self.assertGreater(len(recommendation["experimental_checks"]), 0)

    def test_recommend_formulations_from_design_payload_uses_top_candidates(self) -> None:
        payload = {
            "top_candidates": [
                {
                    "rank": 1,
                    "score": 0.2,
                    "feasible": True,
                    "parameters": {
                        "fiber_density": 0.35,
                        "fiber_stiffness": 8.0,
                        "bending_stiffness": 0.2,
                        "crosslink_prob": 0.45,
                        "domain_size": 1.0,
                    },
                    "features": {
                        "stiffness_mean": 7.8,
                        "anisotropy": 0.2,
                        "connectivity": 1.0,
                        "stress_propagation": 0.9,
                        "risk_index": 0.5,
                    },
                },
                {
                    "rank": 2,
                    "score": 0.3,
                    "feasible": False,
                    "parameters": {
                        "fiber_density": 0.3,
                        "fiber_stiffness": 7.0,
                        "bending_stiffness": 0.18,
                        "crosslink_prob": 0.4,
                        "domain_size": 1.0,
                    },
                    "features": {
                        "stiffness_mean": 6.5,
                        "anisotropy": 0.18,
                        "connectivity": 1.0,
                        "stress_propagation": 0.8,
                        "risk_index": 0.6,
                    },
                },
            ]
        }
        recommendations = recommend_formulations_from_design_payload(payload, max_candidates=2)
        self.assertEqual(len(recommendations), 2)
        self.assertEqual(recommendations[0]["candidate_rank"], 1)

    def test_recommend_campaign_formulations_returns_one_per_target(self) -> None:
        campaign_results = [
            {
                "target_stiffness": 6.0,
                "best_candidate": {
                    "rank": 1,
                    "score": 0.2,
                    "feasible": True,
                    "parameters": {
                        "fiber_density": 0.3,
                        "fiber_stiffness": 7.0,
                        "bending_stiffness": 0.18,
                        "crosslink_prob": 0.42,
                        "domain_size": 1.0,
                    },
                    "features": {
                        "stiffness_mean": 5.9,
                        "anisotropy": 0.15,
                        "connectivity": 1.0,
                        "stress_propagation": 0.72,
                        "risk_index": 0.31,
                    },
                },
            },
            {
                "target_stiffness": 10.0,
                "best_candidate": {
                    "rank": 1,
                    "score": 0.25,
                    "feasible": True,
                    "parameters": {
                        "fiber_density": 0.45,
                        "fiber_stiffness": 12.0,
                        "bending_stiffness": 0.28,
                        "crosslink_prob": 0.7,
                        "domain_size": 0.95,
                    },
                    "features": {
                        "stiffness_mean": 9.8,
                        "anisotropy": 0.22,
                        "connectivity": 1.0,
                        "stress_propagation": 0.88,
                        "risk_index": 0.5,
                    },
                },
            },
        ]
        recommendations = recommend_campaign_formulations(campaign_results)
        self.assertEqual(len(recommendations), 2)
        self.assertEqual(recommendations[0]["target_stiffness"], 6.0)
        self.assertIn("template_name", recommendations[1])


if __name__ == "__main__":
    unittest.main()
