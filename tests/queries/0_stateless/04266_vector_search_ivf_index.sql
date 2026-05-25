-- Tags: no-fasttest, no-ordinary-database

SET enable_analyzer = 1;
SET query_plan_optimize_lazy_materialization = 0;
SET vector_search_with_rescoring = 0;
SET use_skip_indexes_for_top_k = 0;
SET use_top_k_dynamic_filtering = 0;

DROP TABLE IF EXISTS tab_ivf;

SELECT 'Create and query IVF index';

CREATE TABLE tab_ivf
(
    id UInt32,
    vec Array(Float32),
    INDEX idx vec TYPE vector_similarity('ivf', 'L2Distance', 2, 'i8', 2, 2)
)
ENGINE = MergeTree
ORDER BY id
SETTINGS index_granularity = 4;

INSERT INTO tab_ivf VALUES
    (0, [1.0, 0.0]), (1, [1.1, 0.0]), (2, [1.2, 0.0]), (3, [1.3, 0.0]),
    (4, [0.0, 2.0]), (5, [0.0, 2.1]), (6, [0.0, 2.2]), (7, [0.0, 2.3]),
    (8, [5.0, 5.0]), (9, [5.1, 5.0]), (10, [5.2, 5.0]), (11, [5.3, 5.0]);

WITH [0.0, 2.0] AS reference_vec
SELECT id
FROM tab_ivf
ORDER BY L2Distance(vec, reference_vec)
LIMIT 3;

WITH [0.0, 2.0] AS reference_vec
SELECT id, round(L2Distance(vec, reference_vec), 4)
FROM tab_ivf
ORDER BY L2Distance(vec, reference_vec)
LIMIT 3;

SELECT 'IVF rescoring';

WITH [0.0, 2.0] AS reference_vec
SELECT id, round(L2Distance(vec, reference_vec), 4)
FROM tab_ivf
ORDER BY L2Distance(vec, reference_vec)
LIMIT 3
SETTINGS vector_search_with_rescoring = 1;

SELECT 'IVF rescoring fetch multiplier below one';

WITH [0.0, 2.0] AS reference_vec
SELECT count()
FROM
(
    SELECT id
    FROM tab_ivf
    ORDER BY L2Distance(vec, reference_vec)
    LIMIT 4
    SETTINGS vector_search_with_rescoring = 1, vector_search_index_fetch_multiplier = 0.5
);

SELECT count() = 0
FROM
(
    EXPLAIN header = 1
    WITH [0.0, 2.0] AS reference_vec
    SELECT id, round(L2Distance(vec, reference_vec), 4)
    FROM tab_ivf
    ORDER BY L2Distance(vec, reference_vec)
    LIMIT 3
    SETTINGS vector_search_with_rescoring = 1
)
WHERE explain LIKE '%_distance%';

SELECT count() > 0
FROM
(
    EXPLAIN actions = 1
    WITH [0.0, 2.0] AS reference_vec
    SELECT id, round(L2Distance(vec, reference_vec), 4)
    FROM tab_ivf
    ORDER BY L2Distance(vec, reference_vec)
    LIMIT 3
)
WHERE explain LIKE '%_distance%';

EXPLAIN indexes = 1
WITH [0.0, 2.0] AS reference_vec
SELECT id
FROM tab_ivf
ORDER BY L2Distance(vec, reference_vec)
LIMIT 3;

WITH [0.0, 2.0] AS reference_vec
SELECT id
FROM tab_ivf
ORDER BY L2Distance(vec, reference_vec)
LIMIT 1
SETTINGS hnsw_candidate_list_size_for_search = 0;

DETACH TABLE tab_ivf SYNC;
ATTACH TABLE tab_ivf;

SELECT 'After detach/attach';

WITH [0.0, 2.0] AS reference_vec
SELECT id
FROM tab_ivf
ORDER BY L2Distance(vec, reference_vec)
LIMIT 3;

SELECT 'After merge';

INSERT INTO tab_ivf VALUES
    (12, [10.0, 10.0]), (13, [10.1, 10.0]), (14, [10.0, 10.1]), (15, [20.0, 20.0]);

OPTIMIZE TABLE tab_ivf FINAL;

WITH [10.0, 10.0] AS reference_vec
SELECT id
FROM tab_ivf
ORDER BY L2Distance(vec, reference_vec)
LIMIT 1;

DROP TABLE tab_ivf;

SELECT 'Automatic IVF parameters';

CREATE TABLE tab_ivf
(
    id UInt32,
    vec Array(Float32),
    INDEX idx vec TYPE vector_similarity('ivf', 'cosineDistance', 2)
)
ENGINE = MergeTree
ORDER BY id
SETTINGS index_granularity = 4;

INSERT INTO tab_ivf VALUES
    (0, [1.0, 0.0]), (1, [0.0, 1.0]), (2, [0.0, 2.0]), (3, [2.0, 0.0]);

WITH [0.0, 1.0] AS reference_vec
SELECT id
FROM tab_ivf
ORDER BY cosineDistance(vec, reference_vec)
LIMIT 2;

DROP TABLE tab_ivf;

SELECT 'Automatic IVF partitions with explicit probes';

CREATE TABLE tab_ivf
(
    id UInt32,
    vec Array(Float32),
    INDEX idx vec TYPE vector_similarity('ivf', 'L2Distance', 2, 'i8', 0, 4096)
)
ENGINE = MergeTree
ORDER BY id
SETTINGS index_granularity = 16;

INSERT INTO tab_ivf
SELECT number, [toFloat32(number), 0.0]
FROM numbers(16);

WITH
    [7.0, 0.0] AS reference_vec,
    (
        SELECT groupArray(id)
        FROM
        (
            SELECT id
            FROM tab_ivf
            ORDER BY L2Distance(vec, reference_vec)
            LIMIT 5
            SETTINGS ignore_data_skipping_indices = 'idx'
        )
    ) AS exact_ids,
    (
        SELECT groupArray(id)
        FROM
        (
            SELECT id
            FROM tab_ivf
            ORDER BY L2Distance(vec, reference_vec)
            LIMIT 5
        )
    ) AS approx_ids
SELECT length(arrayIntersect(exact_ids, approx_ids));

DROP TABLE tab_ivf;

SELECT 'Parallel IVF assignment';

CREATE TABLE tab_ivf
(
    id UInt32,
    vec Array(Float32),
    INDEX idx vec TYPE vector_similarity('ivf', 'L2Distance', 2, 'i8', 16, 16)
)
ENGINE = MergeTree
ORDER BY id
SETTINGS index_granularity = 8192;

INSERT INTO tab_ivf
SELECT number, [toFloat32(number), toFloat32(number % 17)]
FROM numbers(9001);

WITH [7777.0, 11.0] AS reference_vec
SELECT has(groupArray(id), 7777)
FROM
(
    SELECT id
    FROM tab_ivf
    ORDER BY L2Distance(vec, reference_vec)
    LIMIT 10
);

DROP TABLE tab_ivf;

SELECT 'IVF recall overlap';

CREATE TABLE tab_ivf
(
    id UInt32,
    vec Array(Float32),
    INDEX idx vec TYPE vector_similarity('ivf', 'L2Distance', 8, 'i8', 32, 16)
)
ENGINE = MergeTree
ORDER BY id
SETTINGS index_granularity = 8192;

INSERT INTO tab_ivf
SELECT
    number,
    arrayMap(i -> toFloat32(sin(number * 0.017 + i * 0.31) + cos(number * 0.005 + i * 0.13)), range(8))
FROM numbers(2048);

WITH
    arrayMap(i -> toFloat32(sin(777 * 0.017 + i * 0.31) + cos(777 * 0.005 + i * 0.13)), range(8)) AS reference_vec,
    (
        SELECT groupArray(id)
        FROM
        (
            SELECT id
            FROM tab_ivf
            ORDER BY L2Distance(vec, reference_vec)
            LIMIT 10
            SETTINGS ignore_data_skipping_indices = 'idx'
        )
    ) AS exact_ids,
    (
        SELECT groupArray(id)
        FROM
        (
            SELECT id
            FROM tab_ivf
            ORDER BY L2Distance(vec, reference_vec)
            LIMIT 10
        )
    ) AS approx_ids
SELECT length(arrayIntersect(exact_ids, approx_ids));

DROP TABLE tab_ivf;

SELECT 'Float64 IVF index';

CREATE TABLE tab_ivf
(
    id UInt32,
    vec Array(Float64),
    INDEX idx vec TYPE vector_similarity('ivf', 'L2Distance', 2, 'i8', 2, 2)
)
ENGINE = MergeTree
ORDER BY id
SETTINGS index_granularity = 4;

INSERT INTO tab_ivf VALUES
    (0, [1.0, 0.0]), (1, [2.0, 0.0]), (2, [3.0, 0.0]), (3, [0.0, 3.0]);

WITH [2.0, 0.0] AS reference_vec
SELECT id
FROM tab_ivf
ORDER BY L2Distance(vec, reference_vec)
LIMIT 2;

DROP TABLE tab_ivf;

SELECT 'BFloat16 IVF index';

CREATE TABLE tab_ivf
(
    id UInt32,
    vec Array(BFloat16),
    INDEX idx vec TYPE vector_similarity('ivf', 'L2Distance', 2, 'i8', 2, 2)
)
ENGINE = MergeTree
ORDER BY id
SETTINGS index_granularity = 4;

INSERT INTO tab_ivf VALUES
    (0, [1.0, 0.0]), (1, [0.0, 1.0]), (2, [0.0, 2.0]), (3, [2.0, 0.0]);

WITH [0.0, 2.0] AS reference_vec
SELECT id
FROM tab_ivf
ORDER BY L2Distance(vec, reference_vec)
LIMIT 2;

DROP TABLE tab_ivf;

SELECT 'Dot product IVF index';

CREATE TABLE tab_ivf
(
    id UInt32,
    vec Array(Float32),
    INDEX idx vec TYPE vector_similarity('ivf', 'dotProduct', 2, 'i8', 2, 2)
)
ENGINE = MergeTree
ORDER BY id
SETTINGS index_granularity = 4;

INSERT INTO tab_ivf VALUES
    (0, [1.0, 0.0]), (1, [2.0, 0.0]), (2, [0.0, 3.0]), (3, [-1.0, 0.0]);

WITH [1.0, 0.0] AS reference_vec
SELECT id
FROM tab_ivf
ORDER BY dotProduct(vec, reference_vec) DESC
LIMIT 2;

WITH [1.0, 0.0] AS reference_vec
SELECT id, round(dotProduct(vec, reference_vec), 4)
FROM tab_ivf
ORDER BY dotProduct(vec, reference_vec) DESC
LIMIT 2;

SELECT count() > 0
FROM
(
    EXPLAIN actions = 1
    WITH [1.0, 0.0] AS reference_vec
    SELECT id, round(dotProduct(vec, reference_vec), 4)
    FROM tab_ivf
    ORDER BY dotProduct(vec, reference_vec) DESC
    LIMIT 2
)
WHERE explain LIKE '%_distance%';

DROP TABLE tab_ivf;

SELECT 'Multi-granule dot product IVF index';

CREATE TABLE tab_ivf
(
    id UInt32,
    vec Array(Float32),
    INDEX idx vec TYPE vector_similarity('ivf', 'dotProduct', 2, 'i8', 4, 4) GRANULARITY 1
)
ENGINE = MergeTree
ORDER BY id
SETTINGS index_granularity = 8;

INSERT INTO tab_ivf
SELECT number, [toFloat32(number % 7), toFloat32(number)]
FROM numbers(64);

WITH [0.0, 1.0] AS reference_vec
SELECT id
FROM tab_ivf
ORDER BY dotProduct(vec, reference_vec) DESC
LIMIT 5;

DROP TABLE tab_ivf;

SELECT 'Materialized IVF index';

CREATE TABLE tab_ivf
(
    id UInt32,
    vec Array(Float32)
)
ENGINE = MergeTree
ORDER BY id
SETTINGS index_granularity = 4;

INSERT INTO tab_ivf VALUES
    (0, [1.0, 0.0]), (1, [1.1, 0.0]), (2, [0.0, 2.0]),
    (3, [0.0, 2.1]), (4, [5.0, 5.0]), (5, [5.1, 5.0]);

ALTER TABLE tab_ivf ADD INDEX idx vec TYPE vector_similarity('ivf', 'L2Distance', 2, 'i8', 2, 2);

SELECT count()
FROM system.data_skipping_indices
WHERE table = 'tab_ivf' AND name = 'idx' AND data_compressed_bytes > 0;

ALTER TABLE tab_ivf MATERIALIZE INDEX idx SETTINGS mutations_sync = 2;

SELECT count()
FROM system.data_skipping_indices
WHERE table = 'tab_ivf' AND name = 'idx' AND data_compressed_bytes > 0;

WITH [0.0, 2.0] AS reference_vec
SELECT id
FROM tab_ivf
ORDER BY L2Distance(vec, reference_vec)
LIMIT 2;

DROP TABLE tab_ivf;

SELECT 'Zero vector IVF index';

CREATE TABLE tab_ivf
(
    id UInt32,
    vec Array(Float32),
    INDEX idx vec TYPE vector_similarity('ivf', 'L2Distance', 2, 'i8', 2, 2)
)
ENGINE = MergeTree
ORDER BY id
SETTINGS index_granularity = 4;

INSERT INTO tab_ivf VALUES
    (0, [0.0, 0.0]), (1, [1.0, 0.0]), (2, [0.0, 1.0]), (3, [2.0, 2.0]);

WITH [0.0, 0.0] AS reference_vec
SELECT id
FROM tab_ivf
ORDER BY L2Distance(vec, reference_vec)
LIMIT 1;

DROP TABLE tab_ivf;

SELECT 'Low probe count fills LIMIT';

CREATE TABLE tab_ivf
(
    id UInt32,
    vec Array(Float32),
    INDEX idx vec TYPE vector_similarity('ivf', 'L2Distance', 2, 'i8', 8, 1)
)
ENGINE = MergeTree
ORDER BY id
SETTINGS index_granularity = 8;

INSERT INTO tab_ivf VALUES
    (0, [0.0, 0.0]), (1, [1.0, 0.0]), (2, [2.0, 0.0]), (3, [3.0, 0.0]),
    (4, [4.0, 0.0]), (5, [5.0, 0.0]), (6, [6.0, 0.0]), (7, [7.0, 0.0]);

WITH [0.0, 0.0] AS reference_vec
SELECT count()
FROM
(
    SELECT id
    FROM tab_ivf
    ORDER BY L2Distance(vec, reference_vec)
    LIMIT 3
);

DROP TABLE tab_ivf;

SELECT 'Stable IVF ties';

CREATE TABLE tab_ivf
(
    id UInt64,
    vec Array(Float32),
    INDEX idx vec TYPE vector_similarity('ivf', 'L2Distance', 2, 'i8', 2, 2) GRANULARITY 1
)
ENGINE = MergeTree
ORDER BY id
SETTINGS index_granularity = 8;

INSERT INTO tab_ivf VALUES
    (0, [1.0, 1.0]), (1, [1.0, 1.0]), (2, [1.0, 1.0]), (3, [1.0, 1.0]),
    (4, [1.0, 1.0]), (5, [1.0, 1.0]), (6, [1.0, 1.0]), (7, [1.0, 1.0]);

WITH [1.0, 1.0] AS reference_vec
SELECT groupArray(id)
FROM
(
    SELECT id
    FROM tab_ivf
    ORDER BY L2Distance(vec, reference_vec)
    LIMIT 4
);

DROP TABLE tab_ivf;

SELECT 'Multi-granule IVF index';

CREATE TABLE tab_ivf
(
    id UInt32,
    vec Array(Float32),
    INDEX idx vec TYPE vector_similarity('ivf', 'L2Distance', 2, 'i8', 4, 1) GRANULARITY 1
)
ENGINE = MergeTree
ORDER BY id
SETTINGS index_granularity = 8;

INSERT INTO tab_ivf
SELECT number, [toFloat32(number), toFloat32(number % 3)]
FROM numbers(64);

WITH
    [31.0, 1.0] AS reference_vec,
    (
        SELECT groupArray(id)
        FROM
        (
            SELECT id
            FROM tab_ivf
            ORDER BY L2Distance(vec, reference_vec)
            LIMIT 5
            SETTINGS ignore_data_skipping_indices = 'idx'
        )
    ) AS exact_ids,
    (
        SELECT groupArray(id)
        FROM
        (
            SELECT id
            FROM tab_ivf
            ORDER BY L2Distance(vec, reference_vec)
            LIMIT 5
        )
    ) AS approx_ids
SELECT length(arrayIntersect(exact_ids, approx_ids));

WITH
    [31.0, 1.0] AS reference_vec,
    (
        SELECT groupArray(id)
        FROM
        (
            SELECT id
            FROM tab_ivf
            ORDER BY L2Distance(vec, reference_vec)
            LIMIT 5
            SETTINGS ignore_data_skipping_indices = 'idx'
        )
    ) AS exact_ids,
    (
        SELECT groupArray(id)
        FROM
        (
            SELECT id
            FROM tab_ivf
            ORDER BY L2Distance(vec, reference_vec)
            LIMIT 5
            SETTINGS vector_search_with_rescoring = 1
        )
    ) AS rescored_ids
SELECT length(arrayIntersect(exact_ids, rescored_ids));

DROP TABLE tab_ivf;

SELECT 'IVF with another matching vector index';

CREATE TABLE tab_ivf
(
    id UInt32,
    vec Array(Float32),
    INDEX idx_hnsw vec TYPE vector_similarity('hnsw', 'L2Distance', 2) GRANULARITY 1,
    INDEX idx_ivf vec TYPE vector_similarity('ivf', 'L2Distance', 2, 'i8', 4, 2) GRANULARITY 1
)
ENGINE = MergeTree
ORDER BY id
SETTINGS index_granularity = 8;

INSERT INTO tab_ivf
SELECT number, [toFloat32(number), toFloat32(number % 3)]
FROM numbers(64);

WITH
    [31.0, 1.0] AS reference_vec,
    (
        SELECT groupArray(id)
        FROM
        (
            SELECT id
            FROM tab_ivf
            ORDER BY L2Distance(vec, reference_vec)
            LIMIT 5
            SETTINGS ignore_data_skipping_indices = 'idx_hnsw,idx_ivf'
        )
    ) AS exact_ids,
    (
        SELECT groupArray(id)
        FROM
        (
            SELECT id
            FROM tab_ivf
            ORDER BY L2Distance(vec, reference_vec)
            LIMIT 5
        )
    ) AS approx_ids
SELECT length(arrayIntersect(exact_ids, approx_ids));

DROP TABLE tab_ivf;

SELECT 'Ordered IVF seeding';

CREATE TABLE tab_ivf
(
    id UInt32,
    vec Array(Float32),
    INDEX idx vec TYPE vector_similarity('ivf', 'L2Distance', 2, 'i8', 4, 1)
)
ENGINE = MergeTree
ORDER BY id
SETTINGS index_granularity = 512;

INSERT INTO tab_ivf
SELECT number, [toFloat32(number % 7) / 1000, toFloat32(number % 5) / 1000] FROM numbers(224)
UNION ALL
SELECT 224 + number, [toFloat32(100 + number % 7) / 10, toFloat32(number % 5) / 1000] FROM numbers(16)
UNION ALL
SELECT 240 + number, [toFloat32(number % 7) / 1000, toFloat32(100 + number % 5) / 10] FROM numbers(8)
UNION ALL
SELECT 248 + number, [toFloat32(100 + number % 7) / 10, toFloat32(100 + number % 5) / 10] FROM numbers(8);

CREATE TEMPORARY TABLE event_before AS SELECT sumIf(value, event = 'IVFSearchComputedDistances') AS value FROM system.events;

WITH [10.0, 10.0] AS reference_vec
SELECT id
FROM tab_ivf
ORDER BY L2Distance(vec, reference_vec)
LIMIT 1;

SELECT throwIf(
    (SELECT sumIf(value, event = 'IVFSearchComputedDistances') FROM system.events)
    - (SELECT value FROM event_before) > 64,
    'IVF search should not scan most of a skewed ordered granule when farthest-point seeding is used');

DROP TABLE tab_ivf;

CREATE TABLE tab_ivf_invalid_quantization
(
    id UInt32,
    vec Array(Float32),
    INDEX idx vec TYPE vector_similarity('ivf', 'L2Distance', 2, 'f32', 2, 1)
)
ENGINE = MergeTree
ORDER BY id; -- { serverError INCORRECT_DATA }

CREATE TABLE tab_ivf_invalid_binary_quantization
(
    id UInt32,
    vec Array(Float32),
    INDEX idx vec TYPE vector_similarity('ivf', 'L2Distance', 8, 'b1', 2, 1)
)
ENGINE = MergeTree
ORDER BY id; -- { serverError INCORRECT_DATA }

CREATE TABLE tab_ivf_float64_range
(
    id UInt32,
    vec Array(Float64),
    INDEX idx vec TYPE vector_similarity('ivf', 'L2Distance', 2)
)
ENGINE = MergeTree
ORDER BY id;

INSERT INTO tab_ivf_float64_range VALUES (0, [1e39, 0.0]); -- { serverError INCORRECT_DATA }

INSERT INTO tab_ivf_float64_range VALUES (1, [1.0, 0.0]);

WITH [toFloat64(1e39), 0.0] AS reference_vec
SELECT id
FROM tab_ivf_float64_range
ORDER BY L2Distance(vec, reference_vec)
LIMIT 1; -- { serverError INCORRECT_QUERY }

DROP TABLE tab_ivf_float64_range;

CREATE TABLE tab_ivf_invalid_probes
(
    id UInt32,
    vec Array(Float32),
    INDEX idx vec TYPE vector_similarity('ivf', 'L2Distance', 2, 'i8', 2, 3)
)
ENGINE = MergeTree
ORDER BY id; -- { serverError INCORRECT_DATA }

CREATE TABLE tab_ivf_invalid_auto_probes
(
    id UInt32,
    vec Array(Float32),
    INDEX idx vec TYPE vector_similarity('ivf', 'L2Distance', 2, 'i8', 0, 4097)
)
ENGINE = MergeTree
ORDER BY id; -- { serverError INCORRECT_DATA }

CREATE TABLE tab_ivf_invalid_partitions
(
    id UInt32,
    vec Array(Float32),
    INDEX idx vec TYPE vector_similarity('ivf', 'L2Distance', 2, 'i8', 4097, 1)
)
ENGINE = MergeTree
ORDER BY id; -- { serverError INCORRECT_DATA }

CREATE TABLE tab_ivf_cosine_zero_vector
(
    id UInt32,
    vec Array(Float32),
    INDEX idx vec TYPE vector_similarity('ivf', 'cosineDistance', 2)
)
ENGINE = MergeTree
ORDER BY id;

INSERT INTO tab_ivf_cosine_zero_vector VALUES (0, [0.0, 0.0]); -- { serverError INCORRECT_DATA }

INSERT INTO tab_ivf_cosine_zero_vector VALUES (1, [1.0, 0.0]);

WITH [0.0, 0.0] AS reference_vec
SELECT id
FROM tab_ivf_cosine_zero_vector
ORDER BY cosineDistance(vec, reference_vec)
LIMIT 1; -- { serverError INCORRECT_QUERY }

DROP TABLE tab_ivf_cosine_zero_vector;
