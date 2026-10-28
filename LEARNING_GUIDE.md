# Learning guide: twelve things to be able to explain

Each item is a question an interviewer could ask about this project, a plain answer, and where the idea lives in the code. All numbers mentioned come from the synthetic demo run.

## 1. Why model uplift instead of propensity to churn?

A churn model answers "who is likely to leave". A decision needs "who will stay because of this action". Those are different customers. High-risk customers may be unreachable (the discount does nothing), low-risk customers may be harmed (the simulator's discount effect is negative when churn risk is low, because the offer reopens a decision that was already settled), and some customers respond to one action but not another. Targeting by risk spends budget on the first two groups. Uplift `tau_a(x) = E[y | x, a] - E[y | x, no action]` is the quantity the policy maximizes, multiplied by margin and net of cost. See `nba/simulate.py::treatment_effects` for the designed effects and `nba/policy.py::uplift_scorer` for the net value.

## 2. T-learner versus S-learner versus X-learner

T-learner: one model per arm, uplift is the difference of two predictions. Simple, but each model is fit on only its arm's rows and the difference of two noisy fits is noisier still. S-learner: one model with the arm as an input feature. It uses all the data but a tree model with a small effect rarely splits on the arm indicator, so estimates shrink toward zero; in the demo the S-learner policy actions only a fifth of customers and has the lowest true value among the learners. X-learner: fits the arm models, imputes individual effects on each side, regresses those, and blends with the propensity, so the side with more data informs the side with less. It had the best true value in the demo. Code: `nba/uplift.py`.

## 3. Qini versus AUUC, and why both need randomized data

Both rank customers by estimated uplift and ask how much incremental outcome the top k capture. The Qini curve is `Y_t(k) - Y_c(k) * N_t(k) / N_c(k)`, the Qini coefficient is the area above the random-targeting line, and the units are incremental retained customers. The uplift (cumulative gain) curve scales the difference in means by k and AUUC is its area. Qini is more common and easier to read; AUUC is smoother at the top of the ranking. Either one is biased if the treated and control groups differ in features, which they do on a log produced by a targeting rule, so the demo computes them on a separate randomized synthetic log. Code: `nba/uplift.py::qini_curve`, `uplift_curve`, `nba/evaluate.py::uplift_quality`.

## 4. Why propensities must be logged

Offline evaluation reweights each logged row by `pi(a | x) / p_log(a | x)`. If the logging probability is not recorded you have to estimate it afterwards, and any error goes straight into the weights. Worse, if the logging policy gave some action zero probability for some customers, no estimator can say what the new policy would do there. The simulator records `propensity` on every row and `tests/test_simulate.py` checks that `E[1(a_i = a) / p_i] = 1` for every action, which is the bookkeeping identity that must hold.

## 5. Why IPS has high variance, and what clipping and SNIPS do

With epsilon = 0.3 over five actions the smallest propensity is 0.06, so a deterministic new policy can put weight 16.7 on a single row whose reward can be as large as 300. A few such rows dominate the estimate; the effective sample size in the demo is around 700 out of 8,000 rows. Clipping caps the weights (bias for variance). Self-normalized IPS divides by the sum of the weights, which removes the sensitivity to the weight scale and is usually more stable, with intervals about a quarter narrower than IPS in the demo. It is not always more accurate on a single run, though: across the six policies in the demo its mean absolute error (1.8) was worse than plain IPS (1.2), while in a four-seed check during development it was better by a factor of two. One run does not settle estimator rankings. Code: `nba/ope.py::importance_weights`, `evaluate_policy`.

## 6. What doubly robust fixes, and what the direct method gets wrong

The direct method replaces rewards by a regression `q(x, a)` and asks the model what the new policy would earn. It has tiny variance and whatever bias the model has, and it cannot tell you about the bias: in the demo DM's intervals are about 1.7 wide against 6 for DR, and they are the only ones that miss the truth more than once (4 of 6 cover it, against 6 of 6 for DR). DR uses the model as a baseline and corrects it with the importance-weighted residual `w (r - q(x, a))`. If the model is right the residuals are small and the variance is low; if the propensities are right the correction removes the model's bias. It is consistent if either one is right, which is where the name comes from. Cross-fitting `q` keeps the residual term honest. Code: `nba/ope.py::fit_outcome_model`, `evaluate_policy`.

## 7. Exploration versus exploitation, LinUCB versus Thompson sampling

Always taking the current best action means never learning whether a rarely tried action is better. LinUCB adds a bonus `alpha * sqrt(x^T A^-1 x)` that is large where the model has seen little data, so optimism drives exploration and the bonus shrinks as data arrives. Thompson sampling draws a parameter vector from the posterior and acts as if it were true, so exploration is proportional to posterior uncertainty and is randomized, which also makes its propensities loggable. In the demo both bring per-round regret in the final third of 30,000 rounds to under half of its first-third value (ratios 0.39 and 0.48), and both beat uniform random by a factor of three and a half to five, with LinUCB ahead. Code: `nba/bandit.py`.

## 8. Why constraints sit outside the model

Budgets, eligibility rules, contact caps and thresholds change on a business timescale, often weekly. If they were baked into the training target every change would need a retrain and a re-evaluation. Keeping them in `nba/policy.py` means the model estimates one thing (incremental effect), the policy layer composes it with the rules, and the rules can be audited directly: `check_constraints` is an independent function the gate runs on every decision. It also keeps the explanation honest, because the decision records which constraint was binding.

## 9. How to ship a policy safely: OPE first, then a holdout

Step one is offline: evaluate the candidate on the existing log with DR and SNIPS, look at the interval and the effective sample size, and compare against the logging policy evaluated the same way (which must reproduce the observed mean). If the interval does not clear the incumbent, do not ship. Step two is online with a holdout: route a random slice of customers to the old policy, the rest to the new one, and compare realized outcomes, because OPE intervals on epsilon-greedy logs are wide (several units in the demo, more than the gap between the two best learned policies). Step three is keeping exploration on, so the next candidate can be evaluated offline too. `nba/evaluate.py::gate_checks` is the offline part of that procedure as a CI gate.

## 10. How A/B testing and OPE relate

An A/B test is the gold standard because the randomization is exactly the one you want, but it costs real customers and time and can only compare the policies you actually ran. OPE reuses one log to score any number of candidate policies for free, at the price of variance (and bias, if the overlap or the model is poor). The right pairing is OPE to pick which few candidates deserve an A/B test, and the A/B test to confirm. The logged exploration of an epsilon-greedy or Thompson policy is what makes the OPE step possible at all.

## 11. What changes with delayed outcomes

Retention is observed a quarter later. In this project the simulator returns it immediately, which is a real simplification. With delay, the bandit must act for a long time before any feedback arrives, so it needs either a fast proxy signal (engagement in the first week) with a model that maps proxy to outcome, or batched updates where exploration is planned per batch rather than per round. Uplift models need the full outcome window to have elapsed before training rows are complete, so there is always a lag between the behavior you learn from and the behavior you act on. OPE needs each logged row's reward to be the final one, not a partial one.

## 12. What the synthetic simulator can and cannot tell you

It can tell you whether the machinery works: whether the estimators recover effects they are given, whether OPE intervals cover known truths at the stated rate, whether the constraints hold, whether the bandit's regret curve bends. It cannot tell you anything about real response rates, real costs, or whether the features available in a real system carry any uplift signal. On real data the first things to establish are overlap (does every action have positive probability for every segment), a randomized holdout to make Qini honest, and whether the realized value of a shipped policy matches what OPE predicted, which is the only check on the whole chain that does not depend on a model.
