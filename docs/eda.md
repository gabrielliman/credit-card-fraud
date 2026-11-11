# EDA Findings (2026-11-11)

Reproduce: `conda run -n fraud python -m src.eda_profile data`

## Data health
- 699,960 train / 299,984 test rows, 25 features, **no missing values, no duplicate rows**.
- Target `is_fraud`: 11,689 positives = **1.67%** (≈ 1:59 imbalance).
- **Train and test look identically distributed:** adversarial validation AUC = **0.500**, standardized mean shift < 0.005 on every feature, and no category appears in test that is missing from train. → Implication: stratified K-fold CV should reflect the test set well (supports D003).
- `id` does not predict the target (AUC 0.499), so the rows aren't ordered by fraud.

## Signal (single-feature AUC / fraud rate by level)
| Strong | Medium | ~None |
|---|---|---|
| merchant_risk_score 0.75, velocity_score 0.71, cvv_retry_count 0.71 | customer_age 0.63, txn_count_24h 0.62, hours_since_last_txn 0.57 (fewer hours → more fraud), amount 0.55, prior_disputes 0.53 | distance_from_home, card_age, account_balance, time_of_day, day_of_week (≈0.50–0.51) |

- **Flags are strong:** ip_country_mismatch 16.5% fraud when True vs 1.4% when False; is_ai_generated_scam 15.2%; billing_shipping_mismatch 13.0%; foreign 12.6%; VPN 6.5%; new merchant 4.7%.
- **auth_method:** "No Authentication" 6.4% → Biometric 0.5% (13× range).
- **merchant_category:** Gift Cards 7.3%, Crypto 6.5%, Gaming 5.5% vs ~1.1% for everyday purchases.
- **No signal on their own:** card_type, channel and device_type, whose fraud rates are flat at about 1.6–1.7%. They might still matter in combination with other features, which tree models would capture.

## Correlations
- velocity_score ↔ txn_count_last_24h (ρ = 0.61), velocity_score ↔ hours_since_last_txn (0.44): velocity looks partly derived from these two.
- Everything else is close to independent (|ρ| < 0.15).

## Open questions / hypotheses for experiments
1. Do the features with no signal on their own help through interactions, or should we drop them?
2. velocity_score may be a function of txn_count and hours: is there leftover signal in the residual or in a ratio?
3. Several numeric features have only a few values (cvv_retry, prior_disputes, txn_count): treat them as categories or as numbers?
