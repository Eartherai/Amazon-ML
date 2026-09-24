# Train–test distribution shift

Exact full-source histograms; no hidden labels. Country-conditioned comparisons avoid composition confounding. France has no training counterpart, so no within-France train/test divergence is fabricated.

JSD is divergence in bits (squared SciPy Jensen–Shannon distance); PSI floors each probability at 1e-6 then renormalizes; Wasserstein uses raw feature units. Binning for all three metrics is exact integer feature values. PSI depends on this choice and is descriptive, not a universal alarm threshold.

| Source | Country | Field | Variable | JSD bits | PSI | Wasserstein |
|---|---|---|---|---:|---:|---:|
| S2 | India | business_name | tokens | 0.001986 | 0.011025 | 0.1030 |
| S3 | India | business_name | tokens | 0.001826 | 0.010134 | 0.1048 |
| S2 | India | business_name | length | 0.001747 | 0.009689 | 0.8312 |
| S2 | US | business_name | tokens | 0.001567 | 0.008692 | 0.1088 |
| S3 | India | business_name | length | 0.001496 | 0.008285 | 0.8309 |
| S3 | US | business_name | tokens | 0.001468 | 0.008143 | 0.0951 |
| S2 | US | business_name | length | 0.001341 | 0.007433 | 0.7311 |
| S3 | US | business_name | length | 0.001119 | 0.006196 | 0.6448 |
| S2 | US | business_address | ascii_numeric_tokens | 0.000774 | 0.004295 | 0.0235 |
| S3 | India | business_address | ascii_numeric_tokens | 0.000761 | 0.004219 | 0.0303 |
| S2 | India | business_address | ascii_numeric_tokens | 0.000755 | 0.004183 | 0.0341 |
| S3 | US | business_address | ascii_numeric_tokens | 0.000656 | 0.003645 | 0.0234 |
| S2 | US | business_address | tokens | 0.000475 | 0.002638 | 0.0585 |
| S3 | US | business_address | tokens | 0.000376 | 0.002081 | 0.0582 |
| S2 | US | business_address | length | 0.000372 | 0.002055 | 0.3123 |
| S3 | US | business_address | length | 0.000344 | 0.001891 | 0.3927 |
| S2 | India | business_address | length | 0.000314 | 0.001683 | 0.5504 |
| S3 | India | business_address | length | 0.000304 | 0.001630 | 0.3159 |
| S2 | India | business_address | tokens | 0.000278 | 0.001533 | 0.1064 |
| S3 | India | business_address | tokens | 0.000261 | 0.001434 | 0.0799 |
| S1 | India | business_address | length | 0.000100 | 0.000449 | 0.0328 |
| S1 | India | business_name | length | 0.000038 | 0.000168 | 0.0142 |
| S2 | India | business_name | ascii_numeric_tokens | 0.000034 | 0.000187 | 0.0023 |
| S1 | US | business_address | length | 0.000034 | 0.000162 | 0.0131 |
| S1 | US | business_name | length | 0.000028 | 0.000142 | 0.0098 |
| S3 | India | business_name | ascii_numeric_tokens | 0.000025 | 0.000134 | 0.0020 |
| S2 | US | business_name | ascii_numeric_tokens | 0.000018 | 0.000099 | 0.0023 |
| S3 | US | business_name | ascii_numeric_tokens | 0.000017 | 0.000092 | 0.0024 |
| S1 | India | business_address | tokens | 0.000016 | 0.000074 | 0.0061 |
| S1 | India | business_address | ascii_numeric_tokens | 0.000009 | 0.000034 | 0.0020 |
| S1 | India | business_name | tokens | 0.000007 | 0.000021 | 0.0008 |
| S1 | India | business_name | ascii_numeric_tokens | 0.000007 | 0.000034 | 0.0001 |
| S1 | US | business_address | tokens | 0.000006 | 0.000014 | 0.0009 |
| S1 | US | business_name | tokens | 0.000005 | 0.000028 | 0.0026 |
| S1 | US | business_address | ascii_numeric_tokens | 0.000003 | 0.000012 | 0.0002 |
| S1 | US | business_name | ascii_numeric_tokens | 0.000003 | 0.000014 | 0.0003 |

See `TRAIN_TEST_SHIFT.csv` for machine-readable values and the visual report for country composition, script and missingness shifts. Test profiles are descriptive, not a supervised fit corpus.

Reproduce: `.venv/bin/python scripts/render_preprocessing_report.py`
