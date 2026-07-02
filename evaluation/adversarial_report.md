# Adversarial robustness report

Every JD skill + a buzzword summary + fake 95/100 assessment scores are injected into candidates that should not rank; we then check the promotion rate into the top-100.

| cohort | n | full system | naive keyword |
|---|---|---|---|
| off-domain roles | 40 | **0%** | 100% |
| generic engineers | 4 | **100%** | 100% |

Lower is more robust. The naive keyword ranker is trivially promoted by stuffing.

## Interpretation

- **Off-domain roles are fully robust (0% vs 100%):** the disqualifier gate keys on the (unchanged) title and trajectory keys on the (unchanged) career history, so injecting a skill list changes nothing.
- **Residual finding — generic engineers (n=4):** a non-disqualified engineer stuffed with *every* JD skill + fake 95/100 assessments + a buzzword summary can be promoted (100%). The skill/semantic/attention terms trust self-reported signals. Exposure is limited (the semantic funnel already contains very few evidence-free generic engineers, hence the tiny n), but the honest mitigation is **skill grounding**: discount claimed skills that are never corroborated in the career-history text. Logged as future work.

Reporting a weakness the test surfaced is deliberate — adversarial evaluation is only useful if we act on what it finds.
