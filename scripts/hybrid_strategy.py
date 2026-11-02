"""The trained policy attacks; the rule engine's defence answers what crosses.

Measured on 116 live matches (2026-10-27): the policy (55% won) lost when
enemy units reached our towers - twice as often in losses as in wins - and
answered an enemy entering our half after a median 4 s, having spent its
elixir at the bridge. Training it harder in the simulator did not move the
live record (43% over 90 matches against stronger scripted opponents).

The beatdown rules already defend well enough to keep: assess the threat
(small fry left to the towers), skip it when our troops are on it or when
its cost is already committed, pick a defender, place it in the threat's
path. `HybridStrategy` asks the policy every frame - it keeps the hand order
and tower memory current - and lets that defence override the plan while a
threat stands. With no defender ready the plan is to wait: elixir stays for
the next answer instead of going to the bridge.

`rules_factory` picks the rule set: the Giant beatdown by default, or the
user's P.E.K.K.A plan (`PekkaStrategy`), whose rules also bind the policy -
tanks answered from the bridge, Arrows only on groups of small troops, and
no P.E.K.K.A while a threat stands.
"""

from __future__ import annotations

from dataclasses import replace

from beatdown_strategy import BeatdownStrategy, Plan


class HybridStrategy:
    name = "hybride"

    def __init__(self, policy, cluster_spells: bool = False, rules_factory=BeatdownStrategy):
        self.policy = policy
        self.rules_factory = rules_factory
        self.rules = rules_factory()
        # Rules that cast their own spells (the P.E.K.K.A plan's Arrows) also
        # decide which of the policy's spells may go.
        self.rule_spells = bool(getattr(self.rules, "handles_spells", False))
        # The policy cast Fireball 0.2 and Arrows 0.4-0.7 times a match live.
        # With cluster_spells the runner's spell-on-a-real-cluster rule plays
        # before the policy is asked (handles_spells False hands it back).
        self.handles_spells = not cluster_spells

    def start_match(self, now: float) -> None:
        self.policy.start_match(now)
        self.rules = self.rules_factory()

    def played(self, card: str, tile, now: float) -> None:
        self.policy.played(card, tile, now)
        self.rules.played(card, tile, now)

    def decide(self, state, now: float) -> Plan:
        plan = self.policy.decide(state, now)
        spell = self.rules.spell_plan(state) if self.rule_spells else None
        threats = self.rules.threats(state)
        if spell is not None:
            plan = spell
        else:
            plan = self._bind_policy(plan, threats)
            if threats:
                plan = self.rules.defence_plan(state, threats, now, plan)
                plan = self._bind_policy(plan, threats)
        self.rules.last_plan = plan
        return plan

    def _bind_policy(self, plan: Plan, threats: list) -> Plan:
        """The rule set's hard limits on what the policy may play."""
        if plan.card is None:
            return plan
        if self.rule_spells and plan.card == "arrows" and plan.phase != "sort":
            return Plan(phase=plan.phase, reason=f"Flèches gardées pour un groupe de petites "
                                                 f"troupes · {plan.reason}")
        # A tank dropped while a threat stands would turn on it, wherever it
        # lands (the P.E.K.K.A at the bridge aggroes a Valkyrie at our tower).
        if plan.card == self.rules.tank and not self.rules.tank_defends and threats:
            return replace(plan, card=None, slot=None, tile=None,
                           reason=f"Jamais le {self.rules.tank_label} en défense · {plan.reason}")
        return plan
