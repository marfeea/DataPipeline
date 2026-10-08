"""Observe PhysX contact pairs without changing collision or acceptance rules."""

import numpy as np


class PhysicsContactReports:
    def __init__(self):
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools

        self._decode = PhysicsSchemaTools.intToSdfPath
        self._window = {}
        self.total_headers = 0
        self.total_points = 0
        self.pairs = set()
        self._subscription = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact)

    def _on_contact(self, headers, data):
        for header in headers:
            actors = tuple(sorted((str(self._decode(header.actor0)), str(self._decode(header.actor1)))))
            self.total_headers += 1
            self.pairs.add(actors)
            pair = self._window.setdefault(
                actors, dict(actors=list(actors), events=0, points=0, impulse_sum_Ns=0.0, min_separation_m=None)
            )
            pair["events"] += 1
            for index in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data):
                point = data[index]
                separation = float(point.separation)
                pair["points"] += 1
                self.total_points += 1
                pair["impulse_sum_Ns"] += float(np.linalg.norm(np.asarray(point.impulse)))
                previous = pair["min_separation_m"]
                pair["min_separation_m"] = separation if previous is None else min(previous, separation)
                if hasattr(point, "position"):
                    pair["last_contact_position_w"] = np.asarray(point.position).tolist()

    def begin_control_step(self):
        self._window = {}

    def sample(self):
        return list(self._window.values())

    def finish(self):
        self._subscription = None
        return dict(
            total_headers=self.total_headers,
            total_points=self.total_points,
            observed_pairs=[list(pair) for pair in sorted(self.pairs)],
            scope="Observed physics contact reports; absent reports do not establish collision freedom",
        )
