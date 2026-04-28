-- ============================================================
-- Macedonia Border Crossing Data Schema
-- PostgreSQL — database: border_crossing
-- ============================================================

-- ── Static reference data ────────────────────────────────────

CREATE TABLE IF NOT EXISTS crossings (
    id           SERIAL PRIMARY KEY,
    name         TEXT   NOT NULL UNIQUE,
    display_name TEXT   NOT NULL,
    neighbor     TEXT   NOT NULL
);

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
    estimated_at           TIMESTAMPTZ NOT NULL,
    estimated_wait_minutes REAL,
    confidence             REAL,
    model_version          TEXT,
    context_json           JSONB
);

CREATE INDEX IF NOT EXISTS idx_estimates_crossing_time
    ON wait_time_estimates (crossing_id, estimated_at DESC);

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

INSERT INTO crossings (name, display_name, neighbor) VALUES
    ('bogorodica', 'Bogorodica (МК–ГР)', 'Greece'),
    ('blace',      'Blace (МК–КС)',      'Kosovo'),
    ('tabanovce',  'Tabanovce (МК–СР)',  'Serbia'),
    ('deve_bair',  'Deve Bair (МК–БГ)', 'Bulgaria'),
    ('kafasan',    'Kafasan (МК–АЛ)',    'Albania'),
    ('medzitlija', 'Megjitlija (МК–ГР)', 'Greece')
ON CONFLICT (name) DO NOTHING;