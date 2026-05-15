-- ============================================================
-- Macedonia Border Crossing Data Schema
-- PostgreSQL — database: border_crossing
-- ============================================================

-- ── Static reference data ────────────────────────────────────

CREATE TABLE IF NOT EXISTS crossings (
    id           SERIAL PRIMARY KEY,
    name         TEXT   NOT NULL UNIQUE,
    display_name TEXT   NOT NULL,
    neighbor     TEXT   NOT NULL,
    borderalarm_slug TEXT,
    lane_config  JSONB  NOT NULL DEFAULT '{}'::jsonb
);

ALTER TABLE crossings
    ADD COLUMN IF NOT EXISTS borderalarm_slug TEXT,
    ADD COLUMN IF NOT EXISTS lane_config JSONB NOT NULL DEFAULT '{}'::jsonb;

INSERT INTO crossings (name, display_name, neighbor, borderalarm_slug, lane_config)
VALUES
    (
        'bogorodica',
        'Bogorodica (МК–ГР)',
        'Greece',
        'bogorodica-evzoni',
        $${
          "Bogorodica L1": [[0.32, 0.12], [0.39, 0.14], [0.00, 0.53], [0.00, 0.25]],
          "Bogorodica L2": [[0.39, 0.16], [0.43, 0.16], [0.05, 0.94], [0.00, 0.68]],
          "Bogorodica L3": [[0.43, 0.16], [0.46, 0.16], [0.26, 0.94], [0.05, 0.94]],
          "Bogorodica L4": [[0.50, 0.16], [0.54, 0.16], [0.93, 0.94], [0.73, 0.94]],
          "Bogorodica L5": [[0.54, 0.16], [0.59, 0.16], [1.00, 0.70], [0.93, 0.94]]
        }$$::jsonb
    ),
    (
        'blace',
        'Blace (МК–КС)',
        'Kosovo',
        'blace-merdare',
        $${
          "Blace L1": [[0.480, 0.135], [0.435, 0.211], [0.368, 0.343], [0.308, 0.474], [0.250, 0.625], [0.196, 0.769], [0.140, 0.918], [0.117, 0.993], [0.003, 0.997], [0.000, 0.720], [0.100, 0.542], [0.193, 0.394], [0.264, 0.301], [0.347, 0.214], [0.438, 0.123]],
          "Blace L2": [[0.488, 0.133], [0.519, 0.135], [0.523, 0.207], [0.551, 0.510], [0.573, 0.745], [0.596, 0.993], [0.204, 0.992], [0.270, 0.731], [0.345, 0.483], [0.410, 0.291], [0.458, 0.185]],
          "Blace L3": [[0.579, 0.132], [0.910, 0.995], [0.625, 0.989], [0.548, 0.263], [0.536, 0.137]]
        }$$::jsonb
    ),
    (
        'tabanovce',
        'Tabanovce (МК–СР)',
        'Serbia',
        'tabanovce-presevo',
        $${
          "Tabanovce L1": [[0.516, 0.177], [0.494, 0.137], [0.356, 0.161], [0.217, 0.215], [0.006, 0.389], [0.003, 0.641], [0.203, 0.384], [0.311, 0.297]],
          "Tabanovce L2": [[0.003, 0.684], [0.233, 0.416], [0.346, 0.324], [0.377, 0.309], [0.414, 0.368], [0.308, 0.523], [0.221, 0.666], [0.084, 0.995], [0.001, 0.991]],
          "Tabanovce L3": [[0.145, 0.993], [0.421, 0.997], [0.490, 0.368], [0.415, 0.376], [0.353, 0.449]]
        }$$::jsonb
    ),
    (
        'deve_bair',
        'Deve Bair (МК–БГ)',
        'Bulgaria',
        NULL,
        $${
          "DeveBair L1": [[0.406, 0.168], [0.396, 0.234], [0.345, 0.340], [0.062, 0.992], [0.423, 0.996], [0.494, 0.353], [0.507, 0.168]],
          "DeveBair L2": [[0.573, 0.179], [0.578, 0.342], [0.645, 0.996], [0.995, 0.994], [0.997, 0.658], [0.840, 0.360], [0.814, 0.290], [0.782, 0.170]]
        }$$::jsonb
    ),
    (
        'kafasan',
        'Kafasan (МК–АЛ)',
        'Albania',
        'kjafasan-qafe-thane',
        $${
          "Kafasan L1": [[0.511, 0.243], [0.508, 0.341], [0.233, 0.994], [0.006, 0.994], [0.004, 0.716], [0.414, 0.246]],
          "Kafasan L2": [[0.519, 0.243], [0.523, 0.338], [0.557, 0.516], [0.612, 0.670], [0.740, 0.997], [0.998, 0.996], [0.991, 0.706], [0.602, 0.244]]
        }$$::jsonb
    ),
    (
        'medzitlija',
        'Megjitlija (МК–ГР)',
        'Greece',
        'medjitlija-niki',
        $${
          "Medzitlija L1": [[0.366, 0.220], [0.002, 0.506], [0.000, 0.332], [0.236, 0.222], [0.333, 0.193]],
          "Medzitlija L2": [[-0.001, 0.533], [0.723, 0.262], [0.956, 0.345], [0.995, 0.547], [0.999, 0.995], [0.000, 0.995]]
        }$$::jsonb
    )
ON CONFLICT (name) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    neighbor = EXCLUDED.neighbor,
    borderalarm_slug = EXCLUDED.borderalarm_slug,
    lane_config = EXCLUDED.lane_config;

-- ── Raw vehicle count snapshots (periodic overview) ──────────

CREATE TABLE IF NOT EXISTS snapshots (
    id               BIGSERIAL PRIMARY KEY,
    crossing_id      INTEGER     NOT NULL REFERENCES crossings(id),
    captured_at      TIMESTAMPTZ NOT NULL,
    interval_minutes INTEGER     NOT NULL,
    total_vehicles   INTEGER     NOT NULL,
    cars             INTEGER     NOT NULL DEFAULT 0,
    motorcycles      INTEGER     NOT NULL DEFAULT 0,
    buses            INTEGER     NOT NULL DEFAULT 0,
    trucks           INTEGER     NOT NULL DEFAULT 0,
    lane_breakdown   JSONB,
    fps              REAL
);

CREATE INDEX IF NOT EXISTS idx_snapshots_crossing_time
    ON snapshots (crossing_id, captured_at DESC);

CREATE INDEX IF NOT EXISTS idx_snapshots_time
    ON snapshots (captured_at DESC);

CREATE INDEX IF NOT EXISTS idx_snapshots_lane_breakdown
    ON snapshots USING GIN (lane_breakdown);

-- ── Per-vehicle tracking (entry = first seen, exit = left frame / reached booth) ──

CREATE TABLE IF NOT EXISTS vehicle_crossings (
    id            BIGSERIAL PRIMARY KEY,
    crossing_id   INTEGER     NOT NULL REFERENCES crossings(id),
    track_id      INTEGER     NOT NULL,
    vehicle_type  TEXT,                         -- car / bus / truck / motorcycle
    lane          TEXT,                         -- e.g. "Bogorodica L1"
    entered_at    TIMESTAMPTZ NOT NULL,          -- first frame vehicle appeared
    exited_at     TIMESTAMPTZ,                   -- last frame vehicle was seen
    duration_sec  REAL,                          -- exited_at - entered_at in seconds
    was_reassigned BOOLEAN    DEFAULT FALSE,     -- tracker lost/reacquired the ID
    frame_count   INTEGER     DEFAULT 0,         -- how many frames the vehicle was tracked
    avg_confidence REAL,                         -- mean YOLO confidence across its frames
    notes         TEXT                           -- e.g. "lane switch detected"
);

CREATE INDEX IF NOT EXISTS idx_vehicle_crossings_crossing_time
    ON vehicle_crossings (crossing_id, entered_at DESC);

CREATE INDEX IF NOT EXISTS idx_vehicle_crossings_lane
    ON vehicle_crossings (lane, entered_at DESC);

CREATE INDEX IF NOT EXISTS idx_vehicle_crossings_duration
    ON vehicle_crossings (duration_sec);

-- ── LLM-generated wait time estimates ────────────────────────

CREATE TABLE IF NOT EXISTS wait_time_estimates (
    id                     BIGSERIAL PRIMARY KEY,
    crossing_id            INTEGER     NOT NULL REFERENCES crossings(id),
    snapshot_id            BIGINT      REFERENCES snapshots(id),
    estimated_at           TIMESTAMPTZ NOT NULL,
    estimated_wait_minutes REAL,
    confidence             REAL,
    model_version          TEXT,
    context_json           JSONB
);

CREATE INDEX IF NOT EXISTS idx_estimates_crossing_time
    ON wait_time_estimates (crossing_id, estimated_at DESC);

CREATE INDEX IF NOT EXISTS idx_estimates_snapshot_id
    ON wait_time_estimates (snapshot_id);

-- Dedicated wait_estimator_v3 output linked 1:1 with snapshots
CREATE TABLE IF NOT EXISTS wait_estimator_v3_results (
    id                     BIGSERIAL PRIMARY KEY,
    crossing_id            INTEGER     NOT NULL REFERENCES crossings(id),
    snapshot_id            BIGINT      NOT NULL UNIQUE REFERENCES snapshots(id),
    estimated_at           TIMESTAMPTZ NOT NULL,
    estimated_wait_minutes REAL,
    confidence             REAL,
    model_version          TEXT,
    result_json            JSONB       NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_v3_results_crossing_time
    ON wait_estimator_v3_results (crossing_id, estimated_at DESC);

-- ── Crowdsourced wait times (from borderalarm.com or similar) ──

CREATE TABLE IF NOT EXISTS crowdsourced_waits (
    id              BIGSERIAL PRIMARY KEY,
    crossing_id     INTEGER     NOT NULL REFERENCES crossings(id),
    reported_at     TIMESTAMPTZ NOT NULL,
    wait_minutes    INTEGER     NOT NULL,
    reported_by     TEXT,                        -- "anonymous_" or username
    source          TEXT        DEFAULT 'borderalarm',
    raw_text        TEXT                         -- original scraped text for audit
);

CREATE INDEX IF NOT EXISTS idx_crowdsourced_crossing_time
    ON crowdsourced_waits (crossing_id, reported_at DESC);

-- ── Queue depth multipliers used by wait estimator v3 ─────────────────────

CREATE TABLE IF NOT EXISTS crossing_queue_multipliers (
    id          BIGSERIAL PRIMARY KEY,
    crossing_id INTEGER NOT NULL UNIQUE REFERENCES crossings(id),
    multiplier  REAL    NOT NULL,
    notes       TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Views ─────────────────────────────────────────────────────

-- Latest snapshot per crossing
CREATE OR REPLACE VIEW v_latest_snapshots AS
SELECT DISTINCT ON (s.crossing_id)
    c.name,
    c.display_name,
    c.neighbor,
    s.captured_at,
    s.total_vehicles,
    s.cars,
    s.motorcycles,
    s.buses,
    s.trucks
FROM snapshots s
JOIN crossings c ON s.crossing_id = c.id
ORDER BY s.crossing_id, s.captured_at DESC;

-- Hourly averages from snapshots
CREATE OR REPLACE VIEW v_hourly_averages AS
SELECT
    c.name                                         AS crossing,
    DATE_TRUNC('hour', s.captured_at)              AS hour_utc,
    COUNT(*)                                       AS snapshots,
    ROUND(AVG(s.total_vehicles)::NUMERIC, 1)       AS avg_vehicles,
    MAX(s.total_vehicles)                          AS peak_vehicles,
    ROUND(AVG(s.cars)::NUMERIC, 1)                 AS avg_cars,
    ROUND(AVG(s.buses)::NUMERIC, 1)                AS avg_buses,
    ROUND(AVG(s.trucks)::NUMERIC, 1)               AS avg_trucks
FROM snapshots s
JOIN crossings c ON s.crossing_id = c.id
GROUP BY c.name, DATE_TRUNC('hour', s.captured_at);

-- Average tracked crossing duration per lane per hour
CREATE OR REPLACE VIEW v_avg_crossing_times AS
SELECT
    c.name                                              AS crossing,
    vc.lane,
    DATE_TRUNC('hour', vc.entered_at)                  AS hour_utc,
    COUNT(*)                                            AS vehicles,
    ROUND(AVG(vc.duration_sec)::NUMERIC, 1)            AS avg_duration_sec,
    ROUND(MIN(vc.duration_sec)::NUMERIC, 1)            AS min_duration_sec,
    ROUND(MAX(vc.duration_sec)::NUMERIC, 1)            AS max_duration_sec,
    ROUND(AVG(vc.avg_confidence)::NUMERIC, 3)          AS avg_detection_confidence
FROM vehicle_crossings vc
JOIN crossings c ON vc.crossing_id = c.id
WHERE vc.duration_sec > 10
  AND vc.duration_sec < 7200
  AND vc.exited_at IS NOT NULL
GROUP BY c.name, vc.lane, DATE_TRUNC('hour', vc.entered_at);

-- Throughput per crossing per hour
CREATE OR REPLACE VIEW v_throughput AS
SELECT
    c.name                                              AS crossing,
    DATE_TRUNC('hour', vc.entered_at)                  AS hour_utc,
    COUNT(*)                                            AS vehicles_completed,
    ROUND(AVG(vc.duration_sec)::NUMERIC, 1)            AS avg_duration_sec,
    ROUND(AVG(vc.duration_sec / 60.0)::NUMERIC, 2)    AS avg_duration_min
FROM vehicle_crossings vc
JOIN crossings c ON vc.crossing_id = c.id
WHERE vc.duration_sec > 10
  AND vc.duration_sec < 7200
  AND vc.exited_at IS NOT NULL
GROUP BY c.name, DATE_TRUNC('hour', vc.entered_at);

-- Latest wait time estimate per crossing
CREATE OR REPLACE VIEW v_latest_estimates AS
SELECT DISTINCT ON (e.crossing_id)
    c.name,
    c.display_name,
    e.estimated_at,
    e.estimated_wait_minutes,
    e.confidence,
    e.model_version
FROM wait_time_estimates e
JOIN crossings c ON e.crossing_id = c.id
ORDER BY e.crossing_id, e.estimated_at DESC;

-- Combined current status view (useful for the LLM estimator)
CREATE OR REPLACE VIEW v_current_status AS
SELECT
    ls.name,
    ls.display_name,
    ls.neighbor,
    ls.captured_at                                      AS last_snapshot_at,
    ls.total_vehicles                                   AS current_queue,
    ls.cars,
    ls.buses,
    ls.trucks,
    le.estimated_wait_minutes                           AS last_estimated_wait,
    le.confidence                                       AS last_confidence,
    le.estimated_at                                     AS last_estimated_at,
    ct.avg_duration_sec                                 AS recent_avg_crossing_sec,
    ct.vehicles                                         AS vehicles_tracked_this_hour
FROM v_latest_snapshots ls
LEFT JOIN v_latest_estimates le  ON ls.name = le.name
LEFT JOIN v_avg_crossing_times ct ON ls.name = ct.crossing
    AND ct.hour_utc = DATE_TRUNC('hour', NOW());

-- Crowdsourced wait times aggregated per hour (for model training)
CREATE OR REPLACE VIEW v_crowdsourced_hourly AS
SELECT
    c.name                                              AS crossing,
    DATE_TRUNC('hour', cw.reported_at)                 AS hour_utc,
    COUNT(*)                                            AS reports,
    ROUND(AVG(cw.wait_minutes)::NUMERIC, 1)            AS avg_wait_minutes,
    MIN(cw.wait_minutes)                               AS min_wait_minutes,
    MAX(cw.wait_minutes)                               AS max_wait_minutes,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
          (ORDER BY cw.wait_minutes)::NUMERIC, 1)      AS median_wait_minutes
FROM crowdsourced_waits cw
JOIN crossings c ON cw.crossing_id = c.id
GROUP BY c.name, DATE_TRUNC('hour', cw.reported_at);

-- Training-ready view: joins camera observations with crowdsourced ground truth
CREATE OR REPLACE VIEW v_training_data AS
SELECT
    ha.crossing,
    ha.hour_utc,
    ha.avg_vehicles,
    ha.peak_vehicles,
    ha.avg_cars,
    ha.avg_buses,
    ha.avg_trucks,
    act.avg_duration_sec                                AS cam_avg_crossing_sec,
    act.vehicles                                        AS tracked_vehicles,
    EXTRACT(HOUR FROM ha.hour_utc)                     AS hour_of_day,
    EXTRACT(DOW FROM ha.hour_utc)                      AS day_of_week,
    csh.avg_wait_minutes                               AS ground_truth_wait_minutes,
    csh.reports                                        AS ground_truth_sample_size
FROM v_hourly_averages ha
LEFT JOIN v_avg_crossing_times act
    ON ha.crossing = act.crossing AND ha.hour_utc = act.hour_utc
LEFT JOIN v_crowdsourced_hourly csh
    ON ha.crossing = csh.crossing AND ha.hour_utc = csh.hour_utc
WHERE csh.avg_wait_minutes IS NOT NULL;  -- only rows with ground truth labels

-- ── Seed crossings ────────────────────────────────────────────

