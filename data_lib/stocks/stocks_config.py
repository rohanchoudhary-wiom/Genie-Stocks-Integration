"""
Stock-level configuration — GBRESH integration.
Import into lib/config.py or use standalone.
"""

# =========================================================================
# G — GATEKEEPER
# =========================================================================

# Density router: BallTree count within this radius per lead
GATE_DENSITY_RADIUS_M = 100.0
# Leads in areas with fewer than this many settled decisions → Regime 2
GATE_DENSITY_THRESHOLD = 30

# Non-responder: auto-block partners above this decline rate (30d rolling)
GATE_DECLINE_RATE_BLOCK = 0.85
# Minimum decisions in window before decline rate is trusted
GATE_DECLINE_RATE_MIN_OBS = 20

# Capacity gate: block partner when pending >= this multiple of expected daily slots
GATE_CAPACITY_OVERLOAD_FACTOR = 3.0

# Time-of-day: system notification window (IST hours, 24h format)
GATE_NOTIFY_HOUR_START = 10
GATE_NOTIFY_HOUR_END = 21

# Lookback windows for gate queries (days)
GATE_DECLINE_RATE_WINDOW_DAYS = 30
GATE_SLOTS_LOOKBACK_DAYS = 45


# =========================================================================
# B_OPERATIONAL — PARTNER OPS VECTOR
# =========================================================================

# Lead lifecycle: how many days back to look for pending leads
OPS_PENDING_LEADS_LOOKBACK_DAYS = 30

# Ticket tasks
OPS_TICKET_TYPES = ("service", "cash_collection", "router_pickup")
OPS_TICKET_LOOKBACK_DAYS = 30

# Slot confirmation
OPS_SLOT_LOOKBACK_DATE = "2025-10-01"   # align with your test window

# Per-partner SE window
OPS_SE_WINDOW_DAYS = 30

# Expected daily slots: rolling average window
OPS_SLOTS_ROLLING_DAYS = 30

# Response latency
OPS_RESPONSE_LOOKBACK_DAYS = 30


# =========================================================================
# S — SHOCK LEDGER
# =========================================================================

# Outage severity threshold: partners at or above are shocked
SHOCK_OUTAGE_SEVERITY_THRESHOLD = 3
# Outage recency: only consider outages within this many days
SHOCK_OUTAGE_RECENCY_DAYS = 7

# Decline spike: if 7d rate exceeds 30d baseline by this many pp → transient shock
SHOCK_DECLINE_SPIKE_PP = 0.30

# Internal capacity shock: pending > this × expected_daily_slots
SHOCK_CAPACITY_FACTOR = 3.0

# Ticket overload: active tickets above this count → shock
SHOCK_TICKET_OVERLOAD = 5


# =========================================================================
# E — ACTIVE EXPOSURE
# =========================================================================

# Max concurrent promises per partner before R throttles
EXPOSURE_MAX_CONCURRENT = 10
# E concentration: if pending > factor × expected_slots → skip/downgrade
EXPOSURE_CONCENTRATION_FACTOR = 2.0


# =========================================================================
# R — PROMISE GOVERNOR (COMPOSITE SCORING)
# =========================================================================

# Composite feature weights (initial — override via Dirichlet optimiser)
R_WEIGHT_FIELD = 0.30
R_WEIGHT_CONTESTED = 0.15
R_WEIGHT_EVIDENCE = 0.20
R_WEIGHT_OPS_CAPACITY = 0.15
R_WEIGHT_OPS_RELIABILITY = 0.10
R_WEIGHT_SE_30D = 0.10

# Floor: no weight below this after optimisation
R_MIN_WEIGHT = 0.05

# Confidence tier boundaries (on normalised composite 0–1)
R_TIER_DECLINE = 0.1
R_TIER_LOW = 0.2
R_TIER_MOD = 0.35
# above MOD → HIGH

# R fusion: partner_score = spatial_shrunk × operational_score
# operational_score floor (so a decent spatial score isn't zeroed by sparse ops data)
R_OPS_FLOOR = 0.20


# =========================================================================
# H — CALIBRATION MEMORY (logging enrichment)
# =========================================================================

# Fields to snapshot into test_scores output for reproducibility
H_SNAPSHOT_FIELDS = [
    "FIELD_THRESHOLD", "CONTEST_FIELD_THRESHOLD", "LAMBDA_DECAY",
    "PARENT_TOTAL_THRESHOLD", "INDETERMINATE_INSTALLS_CUTOFF",
    "SHRINKAGE_K", "MIN_SHRINKAGE_RATIO",
    "TRAIN_START_DATE", "TRAIN_END_DATE",
    "TEST_START_DATE", "TEST_END_DATE",
]