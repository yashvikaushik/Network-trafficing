# Temporal semantics of CSE-CIC-IDS2018 flow CSVs

## Scope

This report is a read-only investigation of the ten CSVs in `ai/data/raw/cse-cic-ids2018/`. No raw file was changed, and no preprocessing, feature extraction, or modelling was implemented.

## Conclusion

`Timestamp` should be treated as the **flow start time**: the time of the first packet assigned to the bidirectional flow. It is not a flow-end time, packet/event time for every packet, or the time at which the CSV row became available.

The record’s flow statistics are **complete-flow summaries**. They are calculated as packets are added to a flow and emitted after the flow is closed/expired; consequently, they can include packets and timing intervals after the displayed `Timestamp`.

The safe temporal anchor for a completed-flow record is therefore:

```text
flow_available_time = flow_start_timestamp + Flow Duration (microseconds)
```

This is a necessary working interpretation, subject to the timestamp ambiguity described below.

## Evidence for timestamp interpretation

The CSE-CIC-IDS2018 headers match CICFlowMeter’s flow export schema. In the CICFlowMeter source, the first packet sets `flowStartTime`; the exported `Timestamp` is formatted directly from that value, while `Flow Duration` is computed as `flowLastSeen - flowStartTime`. [BasicFlow.java, lines 114–121 and 555–570](https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/src/main/java/cic/cs/unb/ca/jnetpcap/BasicFlow.java)

`flowLastSeen` is updated whenever a subsequent packet is added to the flow. [BasicFlow.java, lines 165–209](https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/src/main/java/cic/cs/unb/ca/jnetpcap/BasicFlow.java)

The local 84-column CSV has the same leading identity fields and ordering as CICFlowMeter’s exported feature enumeration, while the 80-column CSVs retain the same schema after those identifiers are removed. [FlowFeature.java](https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/src/main/java/cic/cs/unb/ca/jnetpcap/FlowFeature.java)

## Important timestamp limitation

CICFlowMeter’s formatter in the matching source uses `dd/MM/yyyy hh:mm:ss`: lowercase `hh` is a 12-hour clock, and the text includes no AM/PM marker. [BasicFlow.java, lines 810–815](https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/src/main/java/cic/cs/unb/ca/jnetpcap/BasicFlow.java)

All locally observed valid timestamps have hours from 01 through 12, and the prior full scan found 5,249,787 adjacent timestamp reversals. This is consistent with a 12-hour ambiguity and also means that sorting the raw timestamp text alone cannot reliably recover an all-day chronological sequence. The source’s exact version/configuration for these files is not bundled in the dataset, so this is strong matching-source evidence rather than a version-pinned proof.

## Complete-flow feature interpretation

The exporter serializes a flow only after using the accumulated packet lists, counters, byte totals, and timing summaries. It calculates rates from total bytes/packets and total flow duration; IAT statistics are accumulated between packets; and the output uses accumulated TCP-flag counters. [BasicFlow.java, lines 587–633](https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/src/main/java/cic/cs/unb/ca/jnetpcap/BasicFlow.java)

Active/idle values are updated from later packet timestamps and emitted from the completed active/idle summary statistics. [BasicFlow.java, lines 533–554 and 657–674](https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/src/main/java/cic/cs/unb/ca/jnetpcap/BasicFlow.java)

| Requested feature family | Can include information after `Timestamp`? | Why |
|---|---|---|
| `Flow Duration` | Yes | It is last-seen time minus first-packet time. |
| Total Fwd/Backward Packets | Yes | Counts all packets accumulated in each direction through flow closure. |
| Total Length of Fwd/Bwd Packets | Yes | Sums packet payload lengths accumulated through flow closure. |
| `Flow Bytes/s`, `Flow Packets/s` | Yes | Ratios of complete-flow totals to complete-flow duration. |
| Flow IAT statistics | Yes | They summarize intervals between subsequent packets in the full flow. |
| Active/Idle statistics | Yes | They depend on later activity gaps and, for timeout closure, can include the timeout-derived idle period. |
| TCP flag counts | Yes | They count flags over all packets seen before flow closure. |

Thus, none of the requested flow-summary families is safe if a record is placed in a state at its **start** timestamp and intended to represent knowledge available at that instant.

## Schema differences

The nine 80-column files begin:

```text
Dst Port, Protocol, Timestamp, Flow Duration, …, Label
```

The 84-column file, `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv`, begins:

```text
Flow ID, Src IP, Src Port, Dst IP, Dst Port, Protocol, Timestamp, Flow Duration, …, Label
```

The four additional fields are:

1. `Flow ID`
2. `Src IP`
3. `Src Port`
4. `Dst IP`

No fields are missing from the 84-column schema relative to the 80-column schema; the other 80 fields are shared. These four fields are unavailable in nine files, so they cannot define a consistent all-file feature/state schema.

## Preliminary safe/unsafe recommendations

| Usage context | Recommendation |
|---|---|
| State is anchored at the displayed flow-start time | **Unsafe:** exclude all complete-flow statistics in this report. They reveal later packets or later inactivity. |
| State window ends at or after a defensibly derived flow end/availability time | **Conditionally safe:** completed-flow statistics may be aggregated only after the entire flow is available. This needs a resolved, unambiguous timestamp timeline. |
| `Dst Port`, `Protocol` | Potentially available from the first packet, but only after validating their direction/field semantics; they are not complete-flow summaries. |
| `Timestamp` | Use only for temporal ordering/availability, never as a raw predictive feature. Its 12-hour ambiguity must first be resolved. |
| `Flow ID`, `Src IP`, `Src Port`, `Dst IP` | Do not use in a common all-file feature table. They exist in one file only; IP/flow identifiers also create environment-identity and split-leakage risk. |
| `Label` | Target only; never an input feature or state aggregate available before the target window. |

## Potential temporal leakage

Using a row’s start `Timestamp` together with its full-flow feature vector leaks the interval from the second packet through the last packet. The leakage is especially direct for duration, packet/byte totals, rates, IAT features, active/idle statistics, and flag counts.

Even end-time anchoring needs caution: if a flow is emitted only after an inactivity timeout, the exact availability time may be later than `start + Flow Duration`. The CSV contains no explicit end time, export time, timeout setting, timezone, or AM/PM information. A forecasting workflow must not claim real-time start-of-flow prediction from these completed-flow rows.

## Unresolved questions

- Which exact CICFlowMeter version and runtime configuration produced these CSVs?
- What flow and activity timeout values were used, and when was a closed/expired flow emitted?
- What timezone applies to the formatted timestamp?
- Can AM/PM (or a 24-hour time) be recovered from the original PCAP/logs or dataset documentation? Without it, all-day ordering remains ambiguous.
- Does the dataset’s `Timestamp` reflect the first packet for every exported file/version, as the matching CICFlowMeter source indicates, or was any collection-specific transformation applied?
- Are labels assigned at flow close, by a separate event interval, or by another procedure? This affects only target timing, but must be resolved before next-window targets are defined.
