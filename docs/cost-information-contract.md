# Cost-information contract

_Last updated: 2026-09-16_

This document defines which cost signals may be used before routing and which signals are available only after an action executes.

## Pre-routing information

The router may use static provider metadata:

- model identifier and description;
- input/output price table kappa(m);
- exact input-token count when the query and prompt are already formed;
- a pre-execution output-length prior or predictor, if explicitly enabled.

A pre-routing estimate is denoted by

RhatC(q,a) = kappa_in(m) T_in(q,a) + kappa_out(m) That_out(q,a).

If no output-length predictor is enabled, the scorer must not receive the realized output length or realized cost.

## Post-execution information

After an action executes, the system may record output-token count, realized monetary cost, normalized realized cost C(q,a), task performance P(q,a), and deployment reward R(q,a)=P(q,a)-lambda C(q,a).

These values are offline labels and evaluation outcomes. They must not be passed to the scorer for the decision that caused the execution.

## Few-shot calibration

For an unseen candidate, a separate calibration set may be executed before the test phase. Its results can estimate a candidate-level prior such as expected output length or cost. Calibration queries must be disjoint from test queries, and calibration outcomes must not update router parameters unless a separate adaptation experiment is explicitly defined.

## Validation requirements

Every experiment must record the exact candidate metadata available before routing, whether cost was estimated/fixed/omitted from scorer inputs, calibration-query IDs if any, test-query IDs, and the source of every reported P, C, and R value.

A result is invalid if realized test performance, tokens, cost, or reward are used to choose the action before execution.
