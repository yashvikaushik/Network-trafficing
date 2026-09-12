# CSE-CIC-IDS2018 dataset analysis

## Scope and method

This is a read-only analysis of `ai/data/raw/cse-cic-ids2018/`. All ten CSV files were fully scanned one row at a time with a streaming CSV reader; no raw file was changed and no file was loaded in full into memory. The scan checked row widths, headers, timestamps, labels, blank cells, numeric parseability, and non-finite numeric values.

## Inventory

| File | Size | Rows | Columns |
|---|---:|---:|---:|
| Friday-02-03-2018 | 352.37 MB | 1,048,575 | 80 |
| Friday-16-02-2018 | 333.72 MB | 1,048,575 | 80 |
| Friday-23-02-2018 | 382.84 MB | 1,048,575 | 80 |
| Thuesday-20-02-2018 | 4.05 GB | 7,948,748 | 84 |
| Thursday-01-03-2018 | 107.84 MB | 331,125 | 80 |
| Thursday-15-02-2018 | 375.95 MB | 1,048,575 | 80 |
| Thursday-22-02-2018 | 382.64 MB | 1,048,575 | 80 |
| Wednesday-14-02-2018 | 358.22 MB | 1,048,575 | 80 |
| Wednesday-21-02-2018 | 328.89 MB | 1,048,575 | 80 |
| Wednesday-28-02-2018 | 209.25 MB | 613,104 | 80 |
| **Total** | **6.89 GB** | **16,233,002** | — |

`Thuesday-20-02-2018…` is spelled that way in the source filename.

## Schema and network fields

Nine files have the same 80-column schema: 78 numeric flow features plus `Timestamp` and `Label`. The timestamp format is `dd/MM/yyyy HH:mm:ss`.

The common feature set contains destination port and protocol; flow duration; forward/backward packet and byte counts; packet-length statistics; flow, forward, and backward inter-arrival-time statistics; TCP flag counts; headers, windows, directional ratios and bulk-rate fields; subflow counts/bytes; and active/idle statistics. The full observed metadata is recorded in [dataset_metadata.json](dataset_metadata.json).

One file has an incompatible 84-column schema: `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv` additionally has `Flow ID`, `Src IP`, `Src Port`, and `Dst IP`. `Src Port` is numeric; the other three are identifiers. No source/destination IP or flow identifier exists in the other nine files, so entity-level state construction cannot rely on these fields across the complete dataset.

## Time coverage and ordering

The valid intended collection dates are 14, 15, 16, 20, 21, 22, 23, and 28 February, plus 1 and 2 March 2018. Each file covers a portion of its named day, usually from approximately 01:00 to 12:59.

The source row order is not temporal: **5,249,787** adjacent timestamp reversals were found. Every usable time-based operation must parse and sort records first, separately within each collection day.

There are 14 semantically implausible but format-valid timestamps: five 1970 dates in `Wednesday-14-02-2018…` and nine in `Thursday-22-02-2018…`. They must be treated as invalid timestamp records for this collection. The report does not change them.

## Labels and class distribution

`Label` is the supplied technical target. There are 15 valid classes plus 59 embedded repeated header rows, which appear as `Label = "Label"` and are not observations.

| Label | Rows |
|---|---:|
| Benign | 13,484,708 |
| DDOS attack-HOIC | 686,012 |
| DDoS attacks-LOIC-HTTP | 576,191 |
| DoS attacks-Hulk | 461,912 |
| Bot | 286,191 |
| FTP-BruteForce | 193,360 |
| SSH-Bruteforce | 187,589 |
| Infilteration | 161,934 |
| DoS attacks-SlowHTTPTest | 139,890 |
| DoS attacks-GoldenEye | 41,508 |
| DoS attacks-Slowloris | 10,990 |
| DDOS attack-LOIC-UDP | 1,730 |
| Brute Force -Web | 611 |
| Brute Force -XSS | 230 |
| SQL Injection | 87 |

Benign traffic is 83.07% of all rows. The `Infilteration` spelling is retained exactly as supplied; any future canonical naming must be explicitly documented and applied after raw-data preservation.

## Missing and invalid data

- No blank cells or malformed-width rows were found.
- 59 embedded repeated header rows occur: 1 on 16 February, 25 on 1 March, and 33 on 28 February. They also account for all 59 unparseable timestamps and numeric parse failures.
- `Flow Byts/s` and `Flow Pkts/s` each contain 95,760 non-finite values (191,520 cells total, representing 95,760 rows where both fields are non-finite).
- The 14 1970 timestamp outliers described above are additional semantic validity failures.

## Leakage assessment

There is no separate obvious label-duplicate column in the common schema, but there are substantial leakage risks:

- `Label` is the outcome and must never enter model-ready features.
- Attack labels are concentrated in named collection days. Random row-level splits would put near-identical records and the same attack windows into every split; file/day membership and exact timestamps would then become proxies for the target.
- The one 84-column file exposes IP addresses and `Flow ID`; those fields are unavailable elsewhere and may encode environment-specific identities. They must not silently become features in a cross-file dataset.
- Many flow statistics, including duration, byte counts, packet counts, rates, and active/idle values, describe the complete flow. If `Timestamp` marks flow start, using a row before that flow has completed would include future information. A later pipeline must establish timestamp semantics and use completed flows only (or a documented cutoff-safe representation).

## Suitability for temporal network-state forecasting

**Qualified yes.** The dataset provides timestamps, labeled network-flow observations, and a broad common set of flow features, which is enough to construct chronological aggregate network states and technical next-behaviour targets.

It is not, by itself, evidence of real multi-stage enterprise attack chains: it is a scheduled/labeled collection with day-level scenarios and gaps between collection days. Forecasting should therefore be framed initially as the next observed malicious behaviour in a time window, not as a proven attacker campaign stage.

Any future work should remove the known invalid rows, sort by parsed timestamp, retain collection-day boundaries, and use chronological leakage-safe evaluation. No pipeline implementation has been created in this analysis.
