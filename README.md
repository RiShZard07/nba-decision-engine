# nba-decision-engine

A next-best-action decision engine for a synthetic subscription business. Given a customer's features it decides which retention or upsell action to take, or whether to do nothing, using:

1. uplift models that estimate each action's incremental effect per customer (T-learner, S-learner, X-learner on scikit-learn),
2. contextual bandits that balance exploration and exploitation online (LinUCB and linear Thompson sampling),
3. offline policy evaluation so a new policy can be judged on logged data before it ships (IPS, self-normalized IPS, direct method, doubly robust, each with a bootstrap interval),
4. guardrails that sit outside the model: eligibility rules, per-action budget caps, a contact frequency cap, and a minimum net value threshold with "no action" as the default.

## Honest status

This is an AI-assisted educational prototype built with Claude Code. Every customer, action and outcome is synthetic and comes from a seeded simulator in `nba/simulate.py`. There is no production use, no real customers and no real company behind the numbers. The simulator's treatment effects were designed by hand, so the models are recovering effects that were written down, not discovered. The point of the project is to show the full loop (estimate, decide under constraints, evaluate offline, gate, explain) end to end and to be able to defend every piece of it.

## How to run

```
pip install -r requirements.txt
python -m pytest -q                 # 63 offline tests, about 15 seconds
python -m nba demo --check          # full experiment into output/, exits 1 if the gate fails
python -m nba recommend --customer-id 42 --json
```

Other commands: `python -m nba simulate` writes `customers.csv` and `logged.csv`; `python -m nba fit --learner x` fits one learner on `logged.csv`; `python -m nba evaluate --n 5000 --rounds 5000` runs the experiment at a different size. Everything goes into `output/` (ignored by git). The demo takes about 11 seconds on two cores.

## Decision flow

```
features x  -->  uplift model  -->  net value per action        -->  constraints  -->  action
                 tau_a(x)           tau_a(x) * margin(x) - cost_a     eligibility
                                                                      contact cap
                                                                      min net value
                                                                      budget per action
                                                                      (default: no action)

logged data (x, a, propensity, reward)  -->  OPE (IPS / SNIPS / DM / DR + CI)  -->  ship / no ship
                                              compared against simulator truth
                                              because the truth is known here

online: context  -->  bandit (LinUCB / Thompson)  -->  action  -->  reward  -->  update
```

## Simulator design

`generate_customers(n, seed)` draws about 20,000 synthetic customers with these features: `tenure_months`, `monthly_spend`, `plan_tier` (0, 1, 2), `support_contacts_90d`, `usage_trend` (in [-1, 1], negative means declining usage), `device_age_months`, `offers_last_90d` (prior contacts, used by the contact cap) and `churn_risk_score`, which is the true churn probability plus noise, the way an upstream risk model would score it.

True churn probability with no action is a logistic function of the features:

```
logit p_churn = -1.0 + 0.55 * support_contacts - 1.3 * usage_trend - 0.012 * tenure
                + 0.015 * device_age - 0.35 * plan_tier
```

Four actions plus `no_action`, with fixed costs:

| action | cost | true effect on retention probability (tau) | responds negatively when |
|---|---|---|---|
| retention_discount | 12 | clip(0.45 * p_churn - 0.06, -0.06, 0.36) | churn risk is low (the discount reopens a decision that was already settled) |
| device_upgrade_offer | 8 | clip(0.007 * (device_age - 18), -0.12, 0.21) | the device is newer than 18 months |
| data_addon_offer | 4 | clip(0.16 * usage_trend + 0.04 * (tier < 2) - 0.03, -0.20, 0.20) | usage is declining or the customer is on the top tier |
| proactive_support_call | 6 | clip(0.08 * min(contacts, 4) - 0.05, -0.05, 0.27) | there were no recent support contacts (the call is an intrusion) |

Outcome: `retained ~ Bernoulli(1 - p_churn + tau_a)`, `margin = retained * monthly_spend * 3 * 0.45`, `reward = margin - cost_a`. The expected incremental value of action a is `tau_a(x) * margin(x) - cost_a`, which is what the policy layer maximizes. Because the response model is known, the simulator answers counterfactual queries exactly: `expected_rewards(df)` gives the expected reward of every action for every customer, and `policy_value(df, actions)` is the true value of any action vector.

Logging policy: epsilon-greedy (epsilon = 0.3) over a rule-based heuristic that targets the discount by churn score, the upgrade by device age, the support call by recent contacts and the add-on by usage trend. The rules point in the right direction but each is a little too broad, which is what hand-written rules usually look like. Propensities `P_log(a | x) = 0.7 * 1[a = rule(x)] + 0.3 / 5` are recorded on every logged row.

## Models and formulas

Treatment `t` is an integer with 0 meaning no action. Every learner exposes `fit(X, treatment, outcome)` and `predict_uplift(X)`, which returns an array of shape (n, n_actions) where column j is the estimated effect of action j relative to action 0.

- T-learner: one outcome model per arm, `tau_a(x) = mu_a(x) - mu_0(x)`.
- S-learner: a single model `mu(x, a)` on the features plus a one-hot of the arm, `tau_a(x) = mu(x, a) - mu(x, 0)`. With trees and small effects the arm indicator is rarely split on, so the S-learner tends to shrink effects toward zero; the demo shows this.
- X-learner: for each action a, fit `mu_0` on controls and `mu_a` on treated, impute `D_treated = y - mu_0(x)` and `D_control = mu_a(x) - y`, regress each on x to get `tau_treated(x)` and `tau_control(x)`, then blend with the propensity `g(x) = P(T = a | x, T in {0, a})`: `tau_a(x) = g(x) * tau_control(x) + (1 - g(x)) * tau_treated(x)`.

Base regression model everywhere (the learners and the OPE outcome model): `HistGradientBoostingRegressor` with depth 2, 150 iterations and a minimum leaf size of 200, which is deliberately heavily regularized because uplift signals are small. The X-learner's propensity model is a small `HistGradientBoostingClassifier`.

- LinUCB: per action keep `A_a = lambda I + sum x x^T` and `b_a = sum r x`; `theta_a = A_a^-1 b_a`; choose `argmax_a theta_a . x + alpha * sqrt(x^T A_a^-1 x)`. The inverse is kept up to date with a Sherman-Morrison step.
- Linear Thompson sampling: same statistics; sample `theta_a ~ N(A_a^-1 b_a, v^2 A_a^-1)` and choose `argmax_a theta_a . x`.
- The bandit context is the standardized features, an intercept, and five spend interaction terms, because the true reward differences between actions are products of spend and one other feature.

Offline policy evaluation with `w_i = pi(a_i | x_i) / p_i`, clipped at 20, and `q(x, a)` a cross-fitted regression of reward on features and action:

```
IPS    V = mean( w_i r_i )
SNIPS  V = sum( w_i r_i ) / sum( w_i )
DM     V = mean( sum_a pi(a | x_i) q(x_i, a) )
DR     V = mean( sum_a pi(a | x_i) q(x_i, a) + w_i ( r_i - q(x_i, a_i) ) )
```

Each estimator reports a 95 percent percentile bootstrap interval from 200 resamples of the logged rows. Evaluating the logging policy on its own log gives `w_i = 1` for every row, so IPS and SNIPS must reproduce the observed mean reward exactly; the demo checks this.

## Metric definitions

- Qini curve: sort by score, and at the top k rows compute `Q(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k)`. The Qini coefficient is the area under this curve minus the area under the straight line from the origin to the curve's final point (random targeting). Units are incremental retained customers.
- Uplift curve and AUUC: at the top k rows, `gain(k) = (mean_t(k) - mean_c(k)) * k`, zero while either arm is empty; AUUC is the area under this curve with the x axis as the targeted fraction.
- Both are computed per action against the control arm on a separate, fully randomized synthetic log, because Qini on a confounded log is biased. The report also shows the correlation between estimated and true effects, which only a simulator can provide.
- True policy value: `mean_i sum_a pi(a | x_i) E[reward | x_i, a]` from the simulator.
- OPE error: `|estimate - true value|`.
- Effective sample size: `(sum w)^2 / sum w^2`; small values warn that a few heavily weighted rows drive the estimate.
- Regret: `sum_t max_a E[r | x_t, a] - E[r | x_t, a_t]`. The sublinearity test used by the gate: per-round regret in the final third of the run must be at most 0.6 times the per-round regret in the first third. A linear curve scores 1.0, a sqrt(T) curve about 0.32.

## Constraints

`nba/policy.py` applies, in order: eligibility rules (tenure of at least 3 months for the discount, device at least 12 months old for the upgrade, not on the top tier for the add-on, at least one recent contact for the call), the contact cap (customers at or above `contact_cap` prior offers get nothing), the minimum net value threshold, and then a greedy allocation in descending net value under per-action budget caps. A customer whose first choice is out of budget gets their next best eligible action if it clears the threshold, otherwise no action. Each decision records which constraint was binding, which the explanation uses. `check_constraints` audits any decision independently and is what the gate relies on.

There are no demographic or protected-looking attributes in the synthetic data, so the fairness-style guardrail here is structural: the policy can only see the eight operational features above, extra columns in a data frame are ignored by construction, `assert_no_protected_features` fails loudly if a protected-looking name (age, gender, race, zip code and so on) ever enters the feature list, and the same constraints apply to everyone. A real deployment would need an actual fairness audit, which this project does not do.

## Results from the demo (synthetic, seed 7)

Expected net reward per customer on the 8,000 held-out synthetic customers. "True value" is simulator truth; the estimates use only the logged data.

| policy | true value | DR estimate | DR 95% CI | IPS | SNIPS | DM |
|---|---|---|---|---|---|---|
| logging (epsilon-greedy rules) | 63.47 | 63.80 | [62.63, 65.06] | 63.83 | 63.83 | 63.70 |
| heuristic rules, no caps | 65.17 | 65.69 | [64.39, 67.13] | 66.65 | 66.10 | 64.31 |
| uplift_t (budget capped) | 65.43 | 65.43 | [62.07, 68.62] | 66.31 | 64.68 | 66.82 |
| uplift_s (budget capped) | 64.71 | 63.18 | [59.73, 66.34] | 62.41 | 60.36 | 65.64 |
| uplift_x (budget capped) | 65.65 | 64.51 | [61.36, 67.39] | 66.00 | 62.57 | 66.39 |
| oracle (true effects, capped) | 66.76 | 67.08 | [64.25, 70.22] | 68.46 | 65.40 | 67.39 |

Things worth noticing. The X-learner has the best true value among the learned policies but DR picked the T-learner, because the intervals are several units wide while the two policies differ by 0.2; offline evaluation of a deterministic policy on an epsilon-greedy log with 8,000 rows simply cannot resolve that gap, and the honest conclusion is "both beat the logging policy, pick either and confirm with a holdout". The S-learner actions far fewer customers (78 percent no action) because it shrinks the effects. DM has by far the narrowest intervals (about 1.7 wide against 6 for DR) and the worst coverage (4 of the 6 intervals contain the truth, against 6 of 6 for DR), which is the usual warning about model-based estimates: they are confident whether or not they are right.

Uplift quality per action, computed on a separate randomized synthetic log:

| model | action | Qini | AUUC | corr with true effect |
|---|---|---|---|---|
| t | retention_discount | 27.5 | 223.6 | 0.70 |
| t | device_upgrade_offer | 39.6 | 132.5 | 0.77 |
| t | data_addon_offer | 11.7 | 43.8 | 0.69 |
| t | proactive_support_call | 17.0 | 68.6 | 0.54 |
| s | retention_discount | 38.9 | 244.5 | 0.81 |
| s | device_upgrade_offer | 35.6 | 122.1 | 0.29 |
| s | data_addon_offer | 16.1 | 52.6 | 0.85 |
| s | proactive_support_call | 30.1 | 94.0 | 0.80 |
| x | retention_discount | 30.8 | 231.9 | 0.81 |
| x | device_upgrade_offer | 40.7 | 134.3 | 0.84 |
| x | data_addon_offer | 17.0 | 53.9 | 0.76 |
| x | proactive_support_call | 19.4 | 73.4 | 0.63 |

Online bandit simulation, 30,000 rounds on the full synthetic population, rewards divided by 100:

| player | final cumulative regret | regret ratio (last third / first third) | sublinear |
|---|---|---|---|
| LinUCB (alpha 2.0) | 494.6 | 0.39 | yes |
| Thompson sampling (v 0.3) | 682.9 | 0.48 | yes |
| uniform random | 2396.0 | 1.00 | no |
| oracle | 0.0 | 0.00 | yes |

The bandits learn slowly here because retention is a Bernoulli outcome multiplied by a wide margin distribution, so per-round reward noise is about ten times the gap between actions. That is a property of retention problems, not of the algorithms.

Full tables, including action distributions, budget use and the binding-constraint counts, are in `output/report.md` and `output/metrics.json` after a run.

## The regression gate

`python -m nba demo --check` exits nonzero unless all of these hold:

1. the uplift policy chosen by the DR estimate has a higher true value than the logging policy,
2. the DR interval for that policy contains the true value,
3. every budget cap, the contact cap and every eligibility rule held in every constrained policy's decision,
4. LinUCB and Thompson sampling both pass the sublinearity test above.

CI runs the tests and then this gate on Python 3.12.

## Explanations

`python -m nba recommend --customer-id 42` prints something like:

```
Customer 42: recommend device_upgrade_offer. The model expects retention probability to rise
by 30.5 points, worth 9.29 in net value after a cost of 8.00 (margin at stake 56.77).
Strongest drivers versus the typical customer: device_age_months = 46 (+16.7 points),
monthly_spend = 42.05 (+10.5 points), churn_risk_score = 0.268 (-4.1 points).
Binding constraint: none. Next best option: data_addon_offer at 3.30.
```

Feature contributions come from one-at-a-time substitution to the population median, which is cheap and model agnostic but ignores interactions and is not a Shapley decomposition. When no action is taken, the text names the constraint that blocked it. No language model is involved; an LLM could rewrite these for agents, but that is not implemented.

## Limitations

- The synthetic effects are designed, not discovered. Nothing here says anything about how real customers respond.
- Outcomes are immediate. There are no delayed rewards, no repeated decisions over time and no customer state that changes in response to past actions.
- The cost model is a fixed cost per action. A real discount changes margin for as long as it lasts, and the margin itself is assumed known from spend.
- The bandits use linear payoff models with hand-chosen interaction terms and constant exploration parameters.
- There is no fairness audit beyond the structural constraints, and the synthetic data has no protected attributes to audit against.
- Offline evaluation of deterministic policies on a log with 30 percent exploration gives wide intervals, as the results show.
- Single machine, in-memory pandas; nothing is built for scale.
- Feature contributions are one-at-a-time substitutions, not Shapley values.

## Reproduce and explain before using on a resume

Run `python -m pytest -q` and `python -m nba demo --check` yourself and read `output/report.md`. Then make sure you can answer, without notes, the twelve items in `LEARNING_GUIDE.md`: why uplift rather than churn propensity, how each learner works and why the S-learner shrinks, what Qini measures and why it needs randomized data, why propensities must be logged, why IPS is noisy and what DR fixes, and how a policy gets shipped safely. The numbers in `VERIFIED_RESULTS.md` are the ones to quote, and they should always be described as synthetic.

## Layout

```
nba/simulate.py   synthetic population, ground-truth response model, logging policy
nba/uplift.py     T/S/X learners, Qini and uplift curves, AUUC
nba/bandit.py     LinUCB, linear Thompson sampling, online simulation, regret test
nba/policy.py     constraints and the greedy allocation
nba/ope.py        IPS, SNIPS, DM, DR with bootstrap intervals
nba/evaluate.py   experiment runner, report, gate
nba/explain.py    plain-language explanation of one recommendation
nba/__main__.py   CLI
tests/            63 pytest tests, all offline
```
