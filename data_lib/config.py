# EXPERIMENT CONFIGURATION
# Modify these values to run different experiments

# 1. Hexagon Generation

# DECIDING BAD AND MID SE RANGES based on portfolio deciles:
BAD_SE = [1, 2, 3]
MID_SE = [4, 5, 6, 7]

# The list of sizes (in km) to test for finding the best separation
HEX_GRID_SIZES = [0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.25]
DEFAULT_HEX_SIZE = 0.25
# How far out to tile hexes around the partner center (km)
HEX_TILING_RADIUS_KM = 3.0

# 2. Desirability Field Physics
# Adaptive H (Influence Radius)
USE_ADAPTIVE_H = True
ADAPTIVE_H_MIN = 20.0
ADAPTIVE_H_MAX = 150.0
ADAPTIVE_H_NEIGHBOR_K = 3

# CONSTANT 'h' - The impact radius (meters) where influence drops to ~60%
H_INSTALL = 60.0  # Impact radius for positive decisions
H_DECLINE = 60.0  # Impact radius for negative decisions

# Weights for different decision types
WEIGHT_INSTALL = 1.0
WEIGHT_DECLINE = -1.5  # Stronger negative signal

# Time Decay (Lambda)
# w_t = w_0 * exp(-lambda * days_old)
LAMBDA_DECAY = 0.005

# 3. Geometric Features
# Radius to search for local geometric patterns (gullies, clusters) around a lead
GEOM_SEARCH_RADIUS_M = 100.0
# If 1, compute geometry using only installed historical points
GEOM_INSTALL_FILTER = 1
# Regime thresholds for geometry-based context
GEOM_DENSE_THRESHOLD = 0.60
GEOM_GULLY_THRESHOLD = 0.70
GEOM_SPARSE_THRESHOLD = 0.40

# 4. Boundary & Competition
# Search radius to find nearby clusters (degrees)
COMPETITION_SEARCH_RADIUS_DEG = 0.027  # ~3km

# Toggle boundary filtering in lib_find_boundary.py: use p30 filteration to exclude hexagons less than p30 installs
ENABLE_BOUNDARY_FILTER = 1

# 5. Scoring Logic
MIN_DIST_CUTOFF_M = 500.0


# 6. Scoring Thresholds
FIELD_THRESHOLD = -0.1
CONTEST_FIELD_THRESHOLD = -0.1
INDETERMINATE_INSTALLS_CUTOFF = 4
SPARSE_DENSITY_THRESHOLD = 0.5
DEPTH_SCORE_THRESHOLD = 200
PARENT_TOTAL_THRESHOLD = 10



# 7. Fixed Date Windows (inclusive)
# Playground uses fixed calendar windows for reproducibility.
TRAIN_START_DATE = "2024-10-20"
TRAIN_END_DATE = "2025-10-19"
# TRAIN_START_DATE = "2025-09-15"
# TRAIN_END_DATE = "2025-12-15"
TEST_START_DATE = "2025-10-20"
TEST_END_DATE = "2025-11-09"

# G1 logs are allowed an earlier lookback than test start.
G1_LOOKBACK_EXTRA_DAYS = 15

# 8. TOGGLING BETWEEN USING ONLY DEFINITE DECISIONS FOR SCORING OR NOT:
DEFINITE_DECISIONS = 0

SHRINKAGE_K = 4
MIN_SHRINKAGE_RATIO = 0.45

EARTH_RADIUS_KILOMETER = 6371
EARTH_RADIUS_METER = 6371000
METERS_PER_DEG_LAT = 111320.0
TEMPORAL_WINDOWS = [30, 60, 365]
