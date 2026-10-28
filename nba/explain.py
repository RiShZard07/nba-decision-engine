"""Plain-language explanation of one recommendation.

Feature contributions use one-at-a-time substitution: the customer's value
for a feature is replaced by the population median and the change in the
predicted uplift of the action is recorded. This is cheap and model agnostic,
but it is not a Shapley decomposition and it ignores interactions. No LLM is
involved; the text is assembled from templates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from nba import simulate as sim
from nba.policy import Decision


@dataclass
class Explanation:
    customer_id: int
    action: str
    expected_uplift_points: float
    expected_net_value: float
    cost: float
    margin_at_stake: float
    binding_constraint: str
    top_features: list[dict]
    alternatives: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        drivers = ", ".join(
            f"{f['feature']} = {f['value']:g} ({f['contribution']:+.1f} points)" for f in self.top_features
        )
        if self.action == sim.ACTIONS[sim.NO_ACTION]:
            best = self.alternatives[0] if self.alternatives else None
            why = {
                "below_threshold": "no action cleared the minimum net value threshold",
                "ineligible": "the actions worth taking failed an eligibility rule",
                "contact_cap": "the customer already hit the contact frequency cap",
                "budget": "the budgets for the actions worth taking were used up",
            }.get(self.binding_constraint, "no action was selected")
            tail = f" The best candidate was {best['action']} at a net value of {best['net_value']:.2f}." if best else ""
            return f"Customer {self.customer_id}: no action, because {why}.{tail}"
        alt = ""
        if self.alternatives:
            first = self.alternatives[0]
            if self.binding_constraint == "budget_fallback":
                alt = f" The first choice, {first['action']} at {first['net_value']:.2f}, was out of budget."
            else:
                alt = f" Next best option: {first['action']} at {first['net_value']:.2f}."
        return (
            f"Customer {self.customer_id}: recommend {self.action}. The model expects retention "
            f"probability to rise by {self.expected_uplift_points:.1f} points, worth "
            f"{self.expected_net_value:.2f} in net value after a cost of {self.cost:.2f} "
            f"(margin at stake {self.margin_at_stake:.2f}). Strongest drivers versus the typical "
            f"customer: {drivers}. Binding constraint: {self.binding_constraint}.{alt}"
        )


def feature_contributions(model, df: pd.DataFrame, row: int, action: int,
                          reference: pd.DataFrame | None = None, top_k: int = 3) -> list[dict]:
    """Top features by absolute one-at-a-time contribution, in retention points."""
    reference = df if reference is None else reference
    medians = reference[list(sim.FEATURES)].median()
    x = sim.feature_matrix(df.iloc[[row]])
    base = model.predict_uplift(x)[0, action]
    contributions = []
    for j, name in enumerate(sim.FEATURES):
        x_alt = x.copy()
        x_alt[0, j] = medians[name]
        delta = base - model.predict_uplift(x_alt)[0, action]
        contributions.append({"feature": name, "value": float(x[0, j]),
                              "contribution": round(100.0 * float(delta), 2)})
    contributions.sort(key=lambda c: -abs(c["contribution"]))
    return contributions[:top_k]


def explain_recommendation(df: pd.DataFrame, customer_id: int, model, decision: Decision,
                           reference: pd.DataFrame | None = None, top_k: int = 3) -> Explanation:
    """Build an Explanation for one customer from a population-level Decision."""
    row = int(np.flatnonzero(df["customer_id"].to_numpy() == customer_id)[0])
    action = int(decision.actions[row])
    net = decision.net_values[row]
    focus = action if action != sim.NO_ACTION else int(net[1:].argmax() + 1)
    x = sim.feature_matrix(df.iloc[[row]])
    uplift = float(model.predict_uplift(x)[0, focus])
    ranked = [a for a in np.argsort(-net) if a not in (focus, sim.NO_ACTION)]
    alternatives = [{"action": sim.ACTIONS[a], "net_value": round(float(net[a]), 2)} for a in ranked]
    if action == sim.NO_ACTION:
        alternatives.insert(0, {"action": sim.ACTIONS[focus], "net_value": round(float(net[focus]), 2)})
    return Explanation(
        customer_id=int(customer_id),
        action=sim.ACTIONS[action],
        expected_uplift_points=round(100.0 * uplift, 2),
        expected_net_value=round(float(net[focus]), 2),
        cost=float(sim.COSTS[focus]),
        margin_at_stake=round(float(sim.quarterly_margin(df.iloc[[row]])[0]), 2),
        binding_constraint=str(decision.binding[row]),
        top_features=feature_contributions(model, df, row, focus, reference, top_k),
        alternatives=alternatives,
    )
