"""Shared paths, feature schema, and display constants."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

TRAIN_CSV = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_CSV = DATA_DIR / "UNSW_NB15_testing-set.csv"

MODEL_PATH = MODELS_DIR / "model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

TARGET_COL = "label"
DROP_COLS = ("id", "attack_cat")

CATEGORICAL_FEATURES = ("proto", "service", "state")

NUMERIC_FEATURES = (
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
)

FEATURE_COLUMNS = list(CATEGORICAL_FEATURES) + list(NUMERIC_FEATURES)

LABEL_NAMES = {0: "Normal", 1: "Attack"}

PROTO_OPTIONS = ("tcp", "udp", "icmp", "arp", "ospf", "sctp")
SERVICE_OPTIONS = ("-", "http", "dns", "smtp", "ftp", "ftp-data", "ssh", "ssl", "pop3", "dhcp")
STATE_OPTIONS = ("FIN", "INT", "CON", "REQ", "RST", "ECO", "CLO", "ACC")

FEATURE_HELP = {
    "dur": "Record duration in seconds",
    "proto": "Transaction protocol",
    "service": "Application service (use '-' if unused)",
    "state": "Protocol state of the flow",
    "spkts": "Source-to-destination packet count",
    "dpkts": "Destination-to-source packet count",
    "sbytes": "Source-to-destination bytes",
    "dbytes": "Destination-to-source bytes",
    "rate": "Packets per second",
    "sttl": "Source TTL",
    "dttl": "Destination TTL",
    "sload": "Source bits per second",
    "dload": "Destination bits per second",
    "sloss": "Source packets retransmitted or dropped",
    "dloss": "Destination packets retransmitted or dropped",
    "sinpkt": "Source inter-packet arrival time (ms)",
    "dinpkt": "Destination inter-packet arrival time (ms)",
    "sjit": "Source jitter (ms)",
    "djit": "Destination jitter (ms)",
    "swin": "Source TCP window advertisement",
    "stcpb": "Source TCP base sequence number",
    "dtcpb": "Destination TCP base sequence number",
    "dwin": "Destination TCP window advertisement",
    "tcprtt": "TCP connection setup round-trip time",
    "synack": "Time between SYN and SYN-ACK",
    "ackdat": "Time between SYN-ACK and ACK",
    "smean": "Mean of the source packet size",
    "dmean": "Mean of the destination packet size",
    "trans_depth": "Pipelined depth into the HTTP connection",
    "response_body_len": "Size of the HTTP response body",
    "ct_srv_src": "Connections with the same service and source in 100 connections",
    "ct_state_ttl": "State / TTL combination count",
    "ct_dst_ltm": "Connections to the same destination in 100 connections",
    "ct_src_dport_ltm": "Same source and destination port in 100 connections",
    "ct_dst_sport_ltm": "Same destination and source port in 100 connections",
    "ct_dst_src_ltm": "Same source and destination address in 100 connections",
    "is_ftp_login": "FTP session with a user/password (0/1)",
    "ct_ftp_cmd": "FTP command flows in the session",
    "ct_flw_http_mthd": "HTTP methods in the flow",
    "ct_src_ltm": "Connections from the same source in 100 connections",
    "ct_srv_dst": "Same service and destination in 100 connections",
    "is_sm_ips_ports": "Source/destination IP and port are equal (0/1)",
}
