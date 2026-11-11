"""Reward v2: keep v1 shaping and add one explicit controller IK-abort event.

An IK rejection happens before a physics step. It is an event, not an invented
transition, and must remain separate from training transition rewards.
"""

from dataclasses import dataclass

from scripts.pick_place_reward_candidate import PickPlaceReward, RewardConfig

VERSION = "pick-place-candidate/2"


@dataclass(frozen=True)
class RewardConfigV2(RewardConfig):
    ik_failure_cost: float = 15.0


class PickPlaceRewardV2(PickPlaceReward):
    def __init__(self, task, initial, config=None):
        if config is not None and not isinstance(config, RewardConfigV2):
            raise ValueError("v2 requires RewardConfigV2")
        super().__init__(task, initial, config or RewardConfigV2())

    def abort_ik_failure(self, execution_stage):
        """Charge once after an explicit IK abort, without advancing observation/time.

        The caller must establish controller failure evidence. Historical adapters
        validate the original exception and stage; missing IK solutions, timeouts,
        planning world gates, and velocity gates do not establish this event.
        """
        if self.done:
            raise ValueError("episode already terminated")
        if execution_stage not in {"legacy_approach_pick", "legacy_lift", "legacy_place_approach"}:
            raise ValueError("unsupported IK execution stage")
        self.done = True
        return {
            "event_type": "controller_ik_abort",
            "failure": "ik_failed",
            "execution_stage": execution_stage,
            "reward": -self.config.ik_failure_cost,
            "terms": {"ik_failure": -self.config.ik_failure_cost},
            "terminated": True,
            "reward_valid": False,
            "last_observed_time_s": self.previous.time_s,
            "event_time_s": None,
            "physical_transition_created": False,
        }
