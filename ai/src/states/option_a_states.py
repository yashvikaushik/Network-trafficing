"""One-minute, end-anchored network-state aggregation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any


FEATURE_COLUMNS = [
    "flow_count", "total_flow_duration_us", "mean_flow_duration_us",
    "total_fwd_packets", "total_bwd_packets", "total_fwd_bytes",
    "total_bwd_bytes", "mean_flow_bytes_per_s", "mean_flow_packets_per_s",
    "mean_flow_iat_us", "mean_active_us", "mean_idle_us", "fin_flag_count",
    "syn_flag_count", "rst_flag_count", "psh_flag_count", "ack_flag_count",
    "tcp_flow_count", "udp_flow_count", "other_protocol_flow_count",
    "unique_dst_port_count",
]


def floor_minute(timestamp: datetime) -> datetime:
    return timestamp.replace(second=0, microsecond=0)


class WindowAccumulator:
    """Accumulate only records already available by a one-minute window."""

    def __init__(self) -> None:
        self.flow_count = 0
        self.sums = Counter()
        self.ports: set[int] = set()
        self.labels: Counter[str] = Counter()

    def add(self, record: dict[str, Any]) -> None:
        self.flow_count += 1
        self.labels[str(record["label"])] += 1
        self.ports.add(int(record["Dst Port"]))
        protocol = int(record["Protocol"])
        if protocol == 6:
            self.sums["tcp_flow_count"] += 1
        elif protocol == 17:
            self.sums["udp_flow_count"] += 1
        else:
            self.sums["other_protocol_flow_count"] += 1
        for source, destination in (
            ("Flow Duration", "total_flow_duration_us"),
            ("Tot Fwd Pkts", "total_fwd_packets"),
            ("Tot Bwd Pkts", "total_bwd_packets"),
            ("TotLen Fwd Pkts", "total_fwd_bytes"),
            ("TotLen Bwd Pkts", "total_bwd_bytes"),
            ("FIN Flag Cnt", "fin_flag_count"),
            ("SYN Flag Cnt", "syn_flag_count"),
            ("RST Flag Cnt", "rst_flag_count"),
            ("PSH Flag Cnt", "psh_flag_count"),
            ("ACK Flag Cnt", "ack_flag_count"),
            ("Flow Byts/s", "_flow_bytes_per_s"),
            ("Flow Pkts/s", "_flow_packets_per_s"),
            ("Flow IAT Mean", "_flow_iat_us"),
            ("Active Mean", "_active_us"),
            ("Idle Mean", "_idle_us"),
        ):
            self.sums[destination] += float(record[source])

    def window_label(self) -> str:
        """Benign only when all flows are benign; otherwise dominant malicious label.

        Ties are resolved alphabetically, making the multi-attack rule deterministic.
        """
        malicious = {label: count for label, count in self.labels.items() if label != "Benign"}
        if not malicious:
            return "Benign"
        maximum = max(malicious.values())
        return sorted(label for label, count in malicious.items() if count == maximum)[0]

    def feature_row(self) -> dict[str, float | int]:
        count = self.flow_count
        return {
            "flow_count": count,
            "total_flow_duration_us": self.sums["total_flow_duration_us"],
            "mean_flow_duration_us": self.sums["total_flow_duration_us"] / count,
            "total_fwd_packets": self.sums["total_fwd_packets"],
            "total_bwd_packets": self.sums["total_bwd_packets"],
            "total_fwd_bytes": self.sums["total_fwd_bytes"],
            "total_bwd_bytes": self.sums["total_bwd_bytes"],
            "mean_flow_bytes_per_s": self.sums["_flow_bytes_per_s"] / count,
            "mean_flow_packets_per_s": self.sums["_flow_packets_per_s"] / count,
            "mean_flow_iat_us": self.sums["_flow_iat_us"] / count,
            "mean_active_us": self.sums["_active_us"] / count,
            "mean_idle_us": self.sums["_idle_us"] / count,
            "fin_flag_count": self.sums["fin_flag_count"],
            "syn_flag_count": self.sums["syn_flag_count"],
            "rst_flag_count": self.sums["rst_flag_count"],
            "psh_flag_count": self.sums["psh_flag_count"],
            "ack_flag_count": self.sums["ack_flag_count"],
            "tcp_flow_count": self.sums["tcp_flow_count"],
            "udp_flow_count": self.sums["udp_flow_count"],
            "other_protocol_flow_count": self.sums["other_protocol_flow_count"],
            "unique_dst_port_count": len(self.ports),
        }
