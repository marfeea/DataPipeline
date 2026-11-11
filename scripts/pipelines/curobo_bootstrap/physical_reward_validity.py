"""Keep candidate scoring separate from full reward evidence for training."""

import numpy as np

from .pick_place_artifact import save_json


def gate_reward_validity(rows, report):
    """Object contacts do not establish absence of illegal arm/world contact.

    This MVP observes object/finger/support forces but does not have a validated
    per-frame arm/world collision channel or geometric-overlap query. Preserve
    the v2 numerical scores, and mask the complete reward for training.
    """
    coverage = {
        "physical_dynamics": True,
        "grasp_drop_support": bool(all(report.get("contact_positive_controls", (False,)))),
        "illegal_contact": False,
        "geometry_overlap": False,
    }
    complete = all(coverage.values())
    for row in rows:
        row["candidate_reward_valid"] = bool(row.get("candidate_reward_valid", row["reward_valid"]))
        row["reward_valid"] = bool(row["candidate_reward_valid"] and complete and row["transition_valid"])
    report["reward_evidence_coverage"] = coverage
    report["candidate_valid_reward_transitions"] = sum(r["candidate_reward_valid"] for r in rows)
    report["valid_reward_transitions"] = sum(r["reward_valid"] for r in rows)
    report["reward_valid_scope"] = "full v2 reward; missing illegal-contact/overlap evidence masks training loss"
    return np.asarray([r["candidate_reward_valid"] for r in rows], dtype=bool)


def finalize_reward_files(output, rows, report):
    candidate = gate_reward_validity(rows, report)
    save_json(output / "rewards.json", rows)
    with np.load(output / "rewards.npz", allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    arrays["candidate_reward_valid"] = candidate
    arrays["reward_valid"] = np.asarray([r["reward_valid"] for r in rows], dtype=bool)
    np.savez_compressed(output / "rewards.npz", **arrays)
