#include <Storages/MergeTree/MergeTreeIndexVectorSimilarity.h>

#if USE_USEARCH

#include <usearch/index_plugins.hpp>
#include <Columns/ColumnArray.h>
#include <Columns/ColumnsNumber.h>
#include <Common/BitHelpers.h>
#include <Common/CurrentThread.h>
#include <Common/setThreadName.h>
#include <Common/formatReadable.h>
#include <Common/getNumberOfCPUCoresToUse.h>
#include <Common/logger_useful.h>
#include <Common/quoteString.h>
#include <Common/threadPoolCallbackRunner.h>
#include <Common/typeid_cast.h>
#include <Core/Field.h>
#include <Core/ServerSettings.h>
#include <Core/Settings.h>
#include <DataTypes/DataTypeArray.h>
#include <IO/ReadHelpers.h>
#include <IO/WriteHelpers.h>
#include <Interpreters/Context.h>
#include <Interpreters/ProcessList.h>
#include <Interpreters/castColumn.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <string_view>

#include <fmt/ranges.h>

namespace ProfileEvents
{
    extern const Event USearchAddCount;
    extern const Event USearchAddVisitedMembers;
    extern const Event USearchAddComputedDistances;
    extern const Event USearchSearchCount;
    extern const Event USearchSearchVisitedMembers;
    extern const Event USearchSearchComputedDistances;
    extern const Event IVFBuildCount;
    extern const Event IVFBuildRows;
    extern const Event IVFBuildComputedDistances;
    extern const Event IVFSearchCount;
    extern const Event IVFSearchProbes;
    extern const Event IVFSearchComputedDistances;
}

namespace DB
{

namespace Setting
{
    extern const SettingsUInt64 hnsw_candidate_list_size_for_search;
    extern const SettingsFloat vector_search_index_fetch_multiplier;
    extern const SettingsUInt64 max_limit_for_vector_search_queries;
    extern const SettingsBool vector_search_with_rescoring;
}

namespace ServerSetting
{
    extern const ServerSettingsUInt64 max_build_vector_similarity_index_thread_pool_size;
}

namespace ErrorCodes
{
    extern const int FORMAT_VERSION_TOO_OLD;
    extern const int ILLEGAL_COLUMN;
    extern const int INCORRECT_DATA;
    extern const int INCORRECT_NUMBER_OF_COLUMNS;
    extern const int INCORRECT_QUERY;
    extern const int INVALID_SETTING_VALUE;
    extern const int LOGICAL_ERROR;
    extern const int NOT_IMPLEMENTED;
    extern const int TOO_LARGE_ARRAY_SIZE;
}

namespace
{

const std::unordered_map<String, VectorSimilarityIndexMethod> methodNameToMethod = {
    {"hnsw", VectorSimilarityIndexMethod::HNSW},
    {"ivf", VectorSimilarityIndexMethod::IVF}};
const std::set<String> methods = {"hnsw", "ivf"};

/// Maps from user-facing name to internal name
const std::unordered_map<String, unum::usearch::metric_kind_t> distanceFunctionToMetricKind = {
    {"L2Distance", unum::usearch::metric_kind_t::l2sq_k},
    {"cosineDistance", unum::usearch::metric_kind_t::cos_k},
    {"dotProduct", unum::usearch::metric_kind_t::ip_k}};

/// Maps from user-facing name to internal name
const std::unordered_map<String, unum::usearch::scalar_kind_t> quantizationToScalarKind = {
    {"f64", unum::usearch::scalar_kind_t::f64_k},
    {"f32", unum::usearch::scalar_kind_t::f32_k},
    {"f16", unum::usearch::scalar_kind_t::f16_k},
    {"bf16", unum::usearch::scalar_kind_t::bf16_k},
    {"i8", unum::usearch::scalar_kind_t::i8_k},
    {"b1", unum::usearch::scalar_kind_t::b1x8_k}};
/// Usearch provides more quantizations but ^^ above ones seem the only ones comprehensively supported across all distance functions.

constexpr size_t DEFAULT_IVF_TRAINING_ITERATIONS = 8;
constexpr size_t MAX_IVF_TRAINING_ROWS = 65536;
constexpr size_t MAX_IVF_PARTITIONS = 4096;

template<typename T>
concept is_set = std::same_as<T, std::set<typename T::key_type, typename T::key_compare, typename T::allocator_type>>;

template<typename T>
concept is_unordered_map = std::same_as<T, std::unordered_map<typename T::key_type, typename T::mapped_type, typename T::hasher, typename T::key_equal, typename T::allocator_type>>;

template <typename T>
String joinByComma(const T & t)
{
    if constexpr (is_set<T>)
    {
        return fmt::format("{}", fmt::join(t, ", "));
    }
    else if constexpr (is_unordered_map<T>)
    {
        auto keys = std::views::keys(t);
        return fmt::format("{}", fmt::join(keys, ", "));
    }
    std::unreachable();
}

}

namespace
{

Float32 getDefaultVectorSearchScore(unum::usearch::metric_kind_t metric_kind);

}

USearchIndexWithSerialization::USearchIndexWithSerialization(
    size_t dimensions,
    unum::usearch::metric_kind_t metric_kind_,
    unum::usearch::scalar_kind_t scalar_kind,
    UsearchHnswParams usearch_hnsw_params)
    : metric_kind(metric_kind_)
{
    USearchIndex::metric_t metric(dimensions, metric_kind_, scalar_kind);

    unum::usearch::index_dense_config_t config(usearch_hnsw_params.connectivity, usearch_hnsw_params.expansion_add, default_expansion_search);
    config.enable_key_lookups = false; /// we don't do row-to-vector lookups

    auto result = USearchIndex::make(metric, config);
    if (!result)
        throw Exception(ErrorCodes::INCORRECT_DATA, "Could not create vector similarity index. Error: {}", result.error.release());
    swap(result.index);
}

void USearchIndexWithSerialization::serialize(WriteBuffer & ostr) const
{
    auto callback = [&ostr](void * from, size_t n)
    {
        /// USearch may call callback from noexcept function
        try
        {
            ostr.write(reinterpret_cast<const char *>(from), n);
            return true;
        }
        catch (...)
        {
            tryLogCurrentException("VectorSimilarityIndex", "An error while serializing USearch index");
            return false;
        }
    };

    if (auto result = Base::save_to_stream(callback); !result)
        throw Exception(ErrorCodes::INCORRECT_DATA, "Could not save vector similarity index. Error: {}", result.error.release());
}

void USearchIndexWithSerialization::deserialize(ReadBuffer & istr)
{
    auto callback = [&istr](void * from, size_t n)
    {
        /// USearch may call callback from noexcept function
        try
        {
            istr.readStrict(reinterpret_cast<char *>(from), n);
            return true;
        }
        catch (...)
        {
            tryLogCurrentException("VectorSimilarityIndex", "An error while deserializing USearch index");
            return false;
        }
    };

    if (auto result = Base::load_from_stream(callback); !result)
        /// See the comment in MergeTreeIndexGranuleVectorSimilarity::deserializeBinary why we throw here
        throw Exception(ErrorCodes::INCORRECT_DATA, "Could not load vector similarity index. Please drop the index and create it again. Error: {}", result.error.release());

    /// USearch pre-allocates internal data structures for at most N threads. This makes the implicit assumption that the caller (this
    /// class) uses at most this number of threads. The problem here is that there is no such guarantee in ClickHouse because of potential
    /// oversubscription. Therefore, set N as 2 * the available cores - that should be pretty safe. In the unlikely case there are still
    /// more threads at runtime than this limit, we patched usearch to return an error.
    try_reserve(unum::usearch::index_limits_t(limits().members, 2 * getNumberOfCPUCoresToUse()));
}

USearchIndexWithSerialization::Statistics USearchIndexWithSerialization::getStatistics() const
{
    USearchIndex::stats_t global_stats = Base::stats();

    Statistics statistics = {
        .max_level = max_level(),
        .connectivity = connectivity(),
        .size = size(),
        .capacity = capacity(),
        .memory_usage = memory_usage(),
        .bytes_per_vector = bytes_per_vector(),
        .scalar_words = scalar_words(),
        .nodes = global_stats.nodes,
        .edges = global_stats.edges,
        .max_edges = global_stats.max_edges,
        .level_stats = {}};

    for (size_t i = 0; i < statistics.max_level; ++i)
        statistics.level_stats.push_back(Base::stats(i));

    return statistics;
}

String USearchIndexWithSerialization::Statistics::toString() const
{
    return fmt::format("max_level = {}, connectivity = {}, size = {}, capacity = {}, memory_usage = {}, bytes_per_vector = {}, scalar_words = {}, nodes = {}, edges = {}, max_edges = {}",
            max_level, connectivity, size, capacity, ReadableSize(memory_usage), bytes_per_vector, scalar_words, nodes, edges, max_edges);

}

size_t USearchIndexWithSerialization::memoryUsageBytes() const
{
    /// Memory consumption is extremely high, asked in Discord: https://discord.com/channels/1063947616615923875/1064496121520590878/1309266814299144223
    return Base::memory_usage();
}

NearestNeighbours USearchIndexWithSerialization::search(
    const std::vector<Float64> & reference_vector,
    size_t limit,
    bool return_distances,
    size_t expansion_search) const
{
    /// We want to run the search with the user-provided value for setting hnsw_candidate_list_size_for_search (aka. expansion_search).
    /// The way to do this in USearch is to call index_dense_gt::change_expansion_search. Unfortunately, this introduces a need to
    /// synchronize index access, see https://github.com/unum-cloud/usearch/issues/500. As a workaround, we extended USearch' search method
    /// to accept a custom expansion_add setting. The config value is only used on the fly, i.e. not persisted in the index.
    auto search_result = Base::search(reference_vector.data(), limit, USearchIndex::any_thread(), false, expansion_search);
    if (!search_result)
        throw Exception(ErrorCodes::INCORRECT_DATA, "Could not search in vector similarity index. Error: {}", search_result.error.release());

    NearestNeighbours result;
    result.default_distance = getDefaultVectorSearchScore(metric_kind);
    result.rows.resize(search_result.size());
    if (return_distances)
    {
        result.distances = std::vector<float>(search_result.size());
        search_result.dump_to(result.rows.data(), result.distances.value().data());
    }
    else
    {
        search_result.dump_to(result.rows.data());
    }

    ProfileEvents::increment(ProfileEvents::USearchSearchCount);
    ProfileEvents::increment(ProfileEvents::USearchSearchVisitedMembers, search_result.visited_members);
    ProfileEvents::increment(ProfileEvents::USearchSearchComputedDistances, search_result.computed_distances);

    return result;
}

namespace
{

Float32 clampInnerProductScoreToFloat32(Float64 score)
{
    if (std::isnan(score))
        return std::numeric_limits<Float32>::lowest();
    if (score > std::numeric_limits<Float32>::max())
        return std::numeric_limits<Float32>::max();
    if (score < std::numeric_limits<Float32>::lowest())
        return std::numeric_limits<Float32>::lowest();
    return static_cast<Float32>(score);
}

Float32 clampL2DistanceScoreToFloat32(Float64 score)
{
    if (std::isnan(score))
        return std::numeric_limits<Float32>::max();
    if (score <= 0.0)
        return 0.0f;
    if (score > std::numeric_limits<Float32>::max())
        return std::numeric_limits<Float32>::max();
    return static_cast<Float32>(score);
}

Float32 clampCosineDistanceScoreToFloat32(Float64 score)
{
    if (std::isnan(score))
        return std::numeric_limits<Float32>::max();
    if (score <= 0.0)
        return 0.0f;
    if (score >= 2.0)
        return 2.0f;
    return static_cast<Float32>(score);
}

Float64 squaredNorm(const Float32 * vector, size_t dimensions)
{
    Float64 norm_squared = 0.0;
    for (size_t dimension = 0; dimension != dimensions; ++dimension)
    {
        const Float64 value = vector[dimension];
        norm_squared += value * value;
    }

    return norm_squared;
}

void normalizeVectorForCosine(Float32 * vector, size_t dimensions, int error_code, std::string_view context)
{
    const Float64 norm_squared = squaredNorm(vector, dimensions);
    if (!std::isfinite(norm_squared) || norm_squared == 0.0)
        throw Exception(error_code, "Vector for IVF vector similarity index ({}) must have non-zero finite magnitude", context);

    const Float64 inverse_norm = 1.0 / std::sqrt(norm_squared);
    for (size_t dimension = 0; dimension != dimensions; ++dimension)
    {
        const Float64 normalized_value = static_cast<Float64>(vector[dimension]) * inverse_norm;
        if (!std::isfinite(normalized_value) || std::abs(normalized_value) > std::numeric_limits<Float32>::max())
            throw Exception(error_code, "Normalized vector value for IVF vector similarity index ({}) is outside Float32 range", context);
        vector[dimension] = static_cast<Float32>(normalized_value);
    }
}

void normalizeVectorsForCosine(std::vector<Float32> & vectors, size_t row_count, size_t dimensions)
{
    for (size_t row = 0; row != row_count; ++row)
        normalizeVectorForCosine(&vectors[row * dimensions], dimensions, ErrorCodes::INCORRECT_DATA, "indexed vector");
}

Float32 scoreVectorAgainstCentroid(
    const Float32 * vector,
    const Float32 * centroid,
    Float64 vector_norm_squared,
    size_t dimensions,
    unum::usearch::metric_kind_t metric_kind)
{
    if (metric_kind == unum::usearch::metric_kind_t::ip_k)
    {
        Float64 dot = 0.0;
        for (size_t dimension = 0; dimension != dimensions; ++dimension)
            dot += static_cast<Float64>(vector[dimension]) * static_cast<Float64>(centroid[dimension]);

        return clampInnerProductScoreToFloat32(dot);
    }

    if (metric_kind == unum::usearch::metric_kind_t::cos_k)
    {
        Float64 dot = 0.0;
        Float64 centroid_norm_squared = 0.0;
        for (size_t dimension = 0; dimension != dimensions; ++dimension)
        {
            const Float64 vector_value = vector[dimension];
            const Float64 centroid_value = centroid[dimension];
            dot += vector_value * centroid_value;
            centroid_norm_squared += centroid_value * centroid_value;
        }

        if (vector_norm_squared == 0.0 || centroid_norm_squared == 0.0)
            return std::numeric_limits<Float32>::max();
        return clampCosineDistanceScoreToFloat32(1.0 - dot / std::sqrt(vector_norm_squared * centroid_norm_squared));
    }

    Float64 l2_squared = 0.0;
    for (size_t dimension = 0; dimension != dimensions; ++dimension)
    {
        const Float64 diff = static_cast<Float64>(vector[dimension]) - static_cast<Float64>(centroid[dimension]);
        l2_squared += diff * diff;
    }

    return clampL2DistanceScoreToFloat32(l2_squared);
}

struct PartitionScoringContext
{
    Float64 base_score = 0.0;
    Float64 base_norm_squared = 0.0;
    std::vector<Float64> linear_terms;
    std::vector<Float64> norm_linear_terms;
    std::vector<Float64> quadratic_terms;
};

PartitionScoringContext createPartitionScoringContext(size_t dimensions, unum::usearch::metric_kind_t metric_kind)
{
    PartitionScoringContext context;
    context.linear_terms.resize(dimensions);
    if (metric_kind != unum::usearch::metric_kind_t::ip_k)
        context.quadratic_terms.resize(dimensions);
    if (metric_kind == unum::usearch::metric_kind_t::cos_k)
        context.norm_linear_terms.resize(dimensions);
    return context;
}

void preparePartitionScoringContext(
    PartitionScoringContext & context,
    const Float32 * centroid,
    const Float32 * scales,
    const Float32 * reference_vector,
    size_t dimensions,
    unum::usearch::metric_kind_t metric_kind)
{
    if (metric_kind == unum::usearch::metric_kind_t::ip_k)
    {
        context.base_score = 0.0;
        for (size_t dimension = 0; dimension != dimensions; ++dimension)
        {
            const Float64 reference_value = reference_vector[dimension];
            context.base_score += static_cast<Float64>(centroid[dimension]) * reference_value;
            context.linear_terms[dimension] = static_cast<Float64>(scales[dimension]) * reference_value;
        }

        return;
    }

    if (metric_kind == unum::usearch::metric_kind_t::cos_k)
    {
        context.base_score = 0.0;
        context.base_norm_squared = 0.0;
        for (size_t dimension = 0; dimension != dimensions; ++dimension)
        {
            const Float64 centroid_value = centroid[dimension];
            const Float64 scale = scales[dimension];
            context.base_score += centroid_value * static_cast<Float64>(reference_vector[dimension]);
            context.base_norm_squared += centroid_value * centroid_value;
            context.linear_terms[dimension] = scale * static_cast<Float64>(reference_vector[dimension]);
            context.norm_linear_terms[dimension] = 2.0 * centroid_value * scale;
            context.quadratic_terms[dimension] = scale * scale;
        }

        return;
    }

    context.base_score = 0.0;
    for (size_t dimension = 0; dimension != dimensions; ++dimension)
    {
        const Float64 diff = static_cast<Float64>(centroid[dimension]) - static_cast<Float64>(reference_vector[dimension]);
        const Float64 scale = scales[dimension];
        context.base_score += diff * diff;
        context.linear_terms[dimension] = 2.0 * diff * scale;
        context.quadratic_terms[dimension] = scale * scale;
    }
}

Float32 scoreApproximateVectorAgainstReference(
    const Int8 * quantized_vector,
    const PartitionScoringContext & context,
    Float64 reference_norm_squared,
    size_t dimensions,
    unum::usearch::metric_kind_t metric_kind)
{
    Float64 score = context.base_score;
    if (metric_kind == unum::usearch::metric_kind_t::ip_k)
    {
        for (size_t dimension = 0; dimension != dimensions; ++dimension)
            score += context.linear_terms[dimension] * static_cast<Float64>(quantized_vector[dimension]);

        return clampInnerProductScoreToFloat32(score);
    }

    if (metric_kind == unum::usearch::metric_kind_t::cos_k)
    {
        Float64 approximate_norm_squared = context.base_norm_squared;
        for (size_t dimension = 0; dimension != dimensions; ++dimension)
        {
            const Float64 quantized_value = quantized_vector[dimension];
            score += context.linear_terms[dimension] * quantized_value;
            approximate_norm_squared += context.norm_linear_terms[dimension] * quantized_value
                + context.quadratic_terms[dimension] * quantized_value * quantized_value;
        }

        return (!std::isfinite(approximate_norm_squared) || approximate_norm_squared <= 0.0 || reference_norm_squared == 0.0)
            ? std::numeric_limits<Float32>::max()
            : clampCosineDistanceScoreToFloat32(1.0 - score / std::sqrt(approximate_norm_squared * reference_norm_squared));
    }

    for (size_t dimension = 0; dimension != dimensions; ++dimension)
    {
        const Float64 quantized_value = quantized_vector[dimension];
        score += context.linear_terms[dimension] * quantized_value
            + context.quadratic_terms[dimension] * quantized_value * quantized_value;
    }

    return clampL2DistanceScoreToFloat32(score);
}

bool isBetterVectorSearchScore(
    Float32 lhs,
    Float32 rhs,
    unum::usearch::metric_kind_t metric_kind)
{
    if (metric_kind == unum::usearch::metric_kind_t::ip_k)
        return lhs > rhs;
    return lhs < rhs;
}

Float32 getDefaultVectorSearchScore(unum::usearch::metric_kind_t metric_kind)
{
    if (metric_kind == unum::usearch::metric_kind_t::ip_k)
        return std::numeric_limits<Float32>::lowest();
    return std::numeric_limits<Float32>::max();
}

bool isWorseVectorSearchScore(
    Float32 lhs,
    Float32 rhs,
    unum::usearch::metric_kind_t metric_kind)
{
    return isBetterVectorSearchScore(rhs, lhs, metric_kind);
}

UInt32 chooseClosestCentroid(
    const Float32 * vector,
    const std::vector<Float32> & centroids,
    size_t partition_count,
    size_t dimensions,
    unum::usearch::metric_kind_t metric_kind)
{
    UInt32 best_partition = 0;
    const Float64 vector_norm_squared = metric_kind == unum::usearch::metric_kind_t::cos_k ? squaredNorm(vector, dimensions) : 0.0;
    Float32 best_score = scoreVectorAgainstCentroid(vector, centroids.data(), vector_norm_squared, dimensions, metric_kind);

    for (UInt32 partition = 1; partition != partition_count; ++partition)
    {
        const Float32 score = scoreVectorAgainstCentroid(vector, &centroids[partition * dimensions], vector_norm_squared, dimensions, metric_kind);
        if (isBetterVectorSearchScore(score, best_score, metric_kind))
        {
            best_score = score;
            best_partition = partition;
        }
    }

    return best_partition;
}

size_t chooseTrainingRow(size_t training_row, size_t training_row_count, size_t row_count)
{
    if (training_row_count == row_count)
        return training_row;
    return (training_row * row_count) / training_row_count;
}

size_t saturatingProduct(size_t lhs, size_t rhs)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
        return std::numeric_limits<size_t>::max();
    return lhs * rhs;
}

template <typename T>
void readVectorBinaryWithExpectedSize(
    std::vector<T> & vector,
    ReadBuffer & buf,
    size_t expected_size,
    std::string_view field_name)
{
    size_t size = 0;
    readVarUInt(size, buf);

    if (size != expected_size)
        throw Exception(
            ErrorCodes::INCORRECT_DATA,
            "Persisted IVF vector similarity index field {} has wrong size. Expected {}, got {}",
            field_name,
            expected_size,
            size);
    if (size > DEFAULT_MAX_STRING_SIZE / sizeof(T))
        throw Exception(
            ErrorCodes::TOO_LARGE_ARRAY_SIZE,
            "Persisted IVF vector similarity index field {} is too large. Maximum bytes: {}",
            field_name,
            DEFAULT_MAX_STRING_SIZE);

    vector.resize(expected_size);
    for (size_t i = 0; i != expected_size; ++i)
        readBinary(vector[i], buf);
}

template <typename T>
Float32 convertVectorElementToFloat32ForPartitionedQuantizedIndex(T value, int error_code, std::string_view context)
{
    Float64 value_float64;
    if constexpr (std::is_same_v<T, BFloat16>)
        value_float64 = static_cast<Float64>(static_cast<Float32>(value));
    else
        value_float64 = static_cast<Float64>(value);
    if (!std::isfinite(value_float64)
        || value_float64 < static_cast<Float64>(std::numeric_limits<Float32>::lowest())
        || value_float64 > static_cast<Float64>(std::numeric_limits<Float32>::max()))
        throw Exception(
            error_code,
            "Vector for IVF vector similarity index ({}) must contain only values representable as Float32",
            context);

    return static_cast<Float32>(value_float64);
}

}

PartitionedQuantizedIndexWithSerialization::PartitionedQuantizedIndexWithSerialization(
    size_t dimensions_arg,
    unum::usearch::metric_kind_t metric_kind_,
    PartitionedQuantizedParams params_)
    : dimensions_(dimensions_arg)
    , metric_kind(metric_kind_)
    , params(params_)
{
}

size_t PartitionedQuantizedIndexWithSerialization::choosePartitions(size_t rows) const
{
    if (rows == 0)
        return 0;

    if (params.partitions != 0)
        return std::min(params.partitions, rows);

    const auto automatic_partitions = static_cast<size_t>(std::sqrt(static_cast<double>(rows)));
    return std::clamp<size_t>(automatic_partitions, 1, std::min(rows, MAX_IVF_PARTITIONS));
}

size_t PartitionedQuantizedIndexWithSerialization::chooseProbes(size_t partitions_) const
{
    if (partitions_ == 0)
        return 0;

    if (params.probes != 0)
        return std::min(params.probes, partitions_);

    const auto automatic_probes = static_cast<size_t>(std::ceil(2.0 * std::sqrt(static_cast<double>(partitions_))));
    return std::clamp<size_t>(automatic_probes, 1, partitions_);
}

void PartitionedQuantizedIndexWithSerialization::build(std::vector<Float32> && vectors)
{
    if (dimensions_ == 0)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Vector dimensions must be greater than zero");
    if (vectors.size() % dimensions_ != 0)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Vector buffer size must be a multiple of dimensions");

    row_count = vectors.size() / dimensions_;
    partition_count = choosePartitions(row_count);
    probe_count = chooseProbes(partition_count);

    if (row_count == 0)
        return;

    if (metric_kind == unum::usearch::metric_kind_t::cos_k)
        normalizeVectorsForCosine(vectors, row_count, dimensions_);

    ProfileEvents::increment(ProfileEvents::IVFBuildCount);
    ProfileEvents::increment(ProfileEvents::IVFBuildRows, row_count);

    const size_t total_row_count = row_count;
    const size_t training_row_count = std::min(total_row_count, MAX_IVF_TRAINING_ROWS);

    auto check_if_query_was_killed = []
    {
        if (auto query_context = CurrentThread::tryGetQueryContext())
            if (auto query_status = query_context->getProcessListElementSafe())
                query_status->throwIfKilled();
    };

    centroids.assign(partition_count * dimensions_, 0.0f);

    auto initialize_centroid_from_training_row = [&](size_t partition, size_t training_row)
    {
        const size_t source_row = chooseTrainingRow(training_row, training_row_count, total_row_count);
        std::copy_n(&vectors[source_row * dimensions_], dimensions_, &centroids[partition * dimensions_]);
    };

    auto score_training_row_against_centroid = [&](size_t training_row, size_t partition)
    {
        const size_t row = chooseTrainingRow(training_row, training_row_count, total_row_count);
        const Float32 * vector = &vectors[row * dimensions_];
        const Float64 vector_norm_squared = metric_kind == unum::usearch::metric_kind_t::cos_k ? squaredNorm(vector, dimensions_) : 0.0;
        return scoreVectorAgainstCentroid(vector, &centroids[partition * dimensions_], vector_norm_squared, dimensions_, metric_kind);
    };

    /// Deterministic farthest-point seeding is a small build-time cost compared to the Lloyd iterations below,
    /// but it avoids making partition quality depend too much on the physical row order inside a part.
    ProfileEvents::increment(ProfileEvents::IVFBuildComputedDistances, saturatingProduct(training_row_count, partition_count));
    initialize_centroid_from_training_row(0, 0);

    std::vector<UInt8> selected_training_rows(training_row_count, 0);
    selected_training_rows[0] = 1;
    std::vector<Float32> closest_centroid_scores(training_row_count);

    auto update_closest_centroid_scores = [&](size_t partition, bool initialize)
    {
        static constexpr size_t min_parallel_training_rows = 8192;
        static constexpr size_t training_rows_per_task = 2048;

        auto update_range = [&](size_t begin, size_t end)
        {
            for (size_t training_row = begin; training_row != end; ++training_row)
            {
                if ((training_row & 1023) == 0)
                    check_if_query_was_killed();

                const Float32 score = score_training_row_against_centroid(training_row, partition);
                if (initialize || isBetterVectorSearchScore(score, closest_centroid_scores[training_row], metric_kind))
                    closest_centroid_scores[training_row] = score;
            }
        };

        if (training_row_count < min_parallel_training_rows)
        {
            update_range(0, training_row_count);
            return;
        }

        auto & thread_pool = Context::getGlobalContextInstance()->getBuildVectorSimilarityIndexThreadPool();
        ThreadPoolCallbackRunnerLocal<void> runner(thread_pool, ThreadName::MERGETREE_VECTOR_SIM_INDEX);
        for (size_t begin = 0; begin < training_row_count; begin += training_rows_per_task)
        {
            const size_t end = std::min(begin + training_rows_per_task, training_row_count);
            runner.enqueueAndKeepTrack([&, begin, end] { update_range(begin, end); });
        }
        runner.waitForAllToFinishAndRethrowFirstError();
    };

    update_closest_centroid_scores(0, true);

    for (size_t partition = 1; partition != partition_count; ++partition)
    {
        if ((partition & 63) == 0)
            check_if_query_was_killed();

        size_t selected_training_row = training_row_count;
        for (size_t training_row = 0; training_row != training_row_count; ++training_row)
        {
            if (selected_training_rows[training_row])
                continue;

            if (selected_training_row == training_row_count
                || isWorseVectorSearchScore(
                    closest_centroid_scores[training_row],
                    closest_centroid_scores[selected_training_row],
                    metric_kind))
                selected_training_row = training_row;
        }

        if (selected_training_row == training_row_count)
            selected_training_row = 0;

        selected_training_rows[selected_training_row] = 1;
        initialize_centroid_from_training_row(partition, selected_training_row);
        update_closest_centroid_scores(partition, false);
    }

    auto assign_vectors_to_centroids = [&](size_t assignment_count, const auto & get_row, std::vector<UInt32> & assignments)
    {
        static constexpr size_t min_parallel_assignments = 8192;
        static constexpr size_t assignments_per_task = 2048;

        ProfileEvents::increment(ProfileEvents::IVFBuildComputedDistances, saturatingProduct(assignment_count, partition_count));

        if (assignment_count < min_parallel_assignments)
        {
            for (size_t assignment = 0; assignment != assignment_count; ++assignment)
            {
                if ((assignment & 1023) == 0)
                    check_if_query_was_killed();

                const size_t row = get_row(assignment);
                const Float32 * vector = &vectors[row * dimensions_];
                assignments[assignment] = chooseClosestCentroid(vector, centroids, partition_count, dimensions_, metric_kind);
            }
            return;
        }

        auto & thread_pool = Context::getGlobalContextInstance()->getBuildVectorSimilarityIndexThreadPool();
        ThreadPoolCallbackRunnerLocal<void> runner(thread_pool, ThreadName::MERGETREE_VECTOR_SIM_INDEX);
        for (size_t begin = 0; begin < assignment_count; begin += assignments_per_task)
        {
            const size_t end = std::min(begin + assignments_per_task, assignment_count);
            runner.enqueueAndKeepTrack(
                [&, begin, end]
                {
                    for (size_t assignment = begin; assignment != end; ++assignment)
                    {
                        if ((assignment & 1023) == 0)
                            check_if_query_was_killed();

                        const size_t row = get_row(assignment);
                        const Float32 * vector = &vectors[row * dimensions_];
                        assignments[assignment] = chooseClosestCentroid(vector, centroids, partition_count, dimensions_, metric_kind);
                    }
                });
        }
        runner.waitForAllToFinishAndRethrowFirstError();
    };

    std::vector<UInt32> training_assignments(training_row_count, 0);
    for (size_t iteration = 0; iteration != DEFAULT_IVF_TRAINING_ITERATIONS; ++iteration)
    {
        assign_vectors_to_centroids(
            training_row_count,
            [training_row_count, total_row_count](size_t training_row) { return chooseTrainingRow(training_row, training_row_count, total_row_count); },
            training_assignments);

        std::vector<Float64> centroid_sums(partition_count * dimensions_, 0.0);
        std::vector<Float32> next_centroids(partition_count * dimensions_, 0.0f);
        std::vector<UInt32> partition_sizes(partition_count, 0);

        for (size_t training_row = 0; training_row != training_row_count; ++training_row)
        {
            const size_t row = chooseTrainingRow(training_row, training_row_count, row_count);
            const UInt32 partition = training_assignments[training_row];
            ++partition_sizes[partition];

            const Float32 * vector = &vectors[row * dimensions_];
            Float64 * centroid_sum = &centroid_sums[partition * dimensions_];
            for (size_t dimension = 0; dimension != dimensions_; ++dimension)
                centroid_sum[dimension] += static_cast<Float64>(vector[dimension]);
        }

        for (size_t partition = 0; partition != partition_count; ++partition)
        {
            Float32 * centroid = &next_centroids[partition * dimensions_];
            if (partition_sizes[partition] == 0)
            {
                std::copy_n(&centroids[partition * dimensions_], dimensions_, centroid);
                continue;
            }

            const Float64 scale = 1.0 / static_cast<Float64>(partition_sizes[partition]);
            const Float64 * centroid_sum = &centroid_sums[partition * dimensions_];
            for (size_t dimension = 0; dimension != dimensions_; ++dimension)
            {
                const Float64 centroid_value = centroid_sum[dimension] * scale;
                if (!std::isfinite(centroid_value) || std::abs(centroid_value) > std::numeric_limits<Float32>::max())
                    throw Exception(ErrorCodes::INCORRECT_DATA, "Centroid value for IVF vector similarity index is outside Float32 range");
                centroid[dimension] = static_cast<Float32>(centroid_value);
            }

            if (metric_kind == unum::usearch::metric_kind_t::cos_k)
            {
                const Float64 centroid_norm_squared = squaredNorm(centroid, dimensions_);
                if (centroid_norm_squared == 0.0)
                    std::copy_n(&centroids[partition * dimensions_], dimensions_, centroid);
                else
                    normalizeVectorForCosine(centroid, dimensions_, ErrorCodes::INCORRECT_DATA, "centroid");
            }
        }

        centroids = std::move(next_centroids);
    }

    std::vector<UInt32> assignments(row_count, 0);
    assign_vectors_to_centroids(row_count, [](size_t row) { return row; }, assignments);

    std::vector<UInt32> partition_sizes(partition_count, 0);
    partition_dimension_scales.assign(partition_count * dimensions_, 0.0f);

    for (size_t row = 0; row != row_count; ++row)
    {
        const UInt32 partition = assignments[row];
        ++partition_sizes[partition];

        const Float32 * vector = &vectors[row * dimensions_];
        const Float32 * centroid = &centroids[partition * dimensions_];
        Float32 * scales = &partition_dimension_scales[partition * dimensions_];
        for (size_t dimension = 0; dimension != dimensions_; ++dimension)
        {
            const Float64 residual_scale =
                std::abs(static_cast<Float64>(vector[dimension]) - static_cast<Float64>(centroid[dimension])) / 127.0;
            if (!std::isfinite(residual_scale) || residual_scale > std::numeric_limits<Float32>::max())
                throw Exception(ErrorCodes::INCORRECT_DATA, "Residual scale for IVF vector similarity index is outside Float32 range");
            scales[dimension] = std::max(scales[dimension], static_cast<Float32>(residual_scale));
        }
    }

    for (auto & scale : partition_dimension_scales)
        scale = scale == 0.0f ? 1.0f : scale;

    partition_offsets.assign(partition_count + 1, 0);
    for (size_t partition = 0; partition != partition_count; ++partition)
        partition_offsets[partition + 1] = partition_offsets[partition] + partition_sizes[partition];

    row_ids.resize(row_count);
    quantized_vectors.resize(row_count * dimensions_);

    std::vector<UInt32> write_positions = partition_offsets;
    for (size_t row = 0; row != row_count; ++row)
    {
        const UInt32 partition = assignments[row];
        const UInt32 position = write_positions[partition]++;
        row_ids[position] = static_cast<UInt32>(row);

        const Float32 * vector = &vectors[row * dimensions_];
        const Float32 * centroid = &centroids[partition * dimensions_];
        const Float32 * scales = &partition_dimension_scales[partition * dimensions_];
        Int8 * quantized_vector = &quantized_vectors[position * dimensions_];

        for (size_t dimension = 0; dimension != dimensions_; ++dimension)
        {
            const auto scaled_residual =
                (static_cast<Float64>(vector[dimension]) - static_cast<Float64>(centroid[dimension]))
                / static_cast<Float64>(scales[dimension]);
            if (!std::isfinite(scaled_residual))
                throw Exception(ErrorCodes::INCORRECT_DATA, "Scaled residual for IVF vector similarity index is not finite");
            const auto quantized = std::lround(scaled_residual);
            quantized_vector[dimension] = static_cast<Int8>(std::clamp<long>(quantized, -127, 127));
        }
    }
}

void PartitionedQuantizedIndexWithSerialization::serialize(WriteBuffer & ostr) const
{
    writeIntBinary(static_cast<UInt64>(dimensions_), ostr);
    writeIntBinary(static_cast<UInt64>(row_count), ostr);
    writeIntBinary(static_cast<UInt64>(partition_count), ostr);
    writeIntBinary(static_cast<UInt64>(probe_count), ostr);
    writeVectorBinary(centroids, ostr);
    writeVectorBinary(partition_dimension_scales, ostr);
    writeVectorBinary(partition_offsets, ostr);
    writeVectorBinary(row_ids, ostr);
    writeVectorBinary(quantized_vectors, ostr);
}

void PartitionedQuantizedIndexWithSerialization::deserialize(ReadBuffer & istr)
{
    UInt64 persisted_dimensions;
    UInt64 persisted_row_count;
    UInt64 persisted_partition_count;
    UInt64 persisted_probe_count;

    readIntBinary(persisted_dimensions, istr);
    readIntBinary(persisted_row_count, istr);
    readIntBinary(persisted_partition_count, istr);
    readIntBinary(persisted_probe_count, istr);

    if (persisted_dimensions != dimensions_)
        throw Exception(
            ErrorCodes::INCORRECT_DATA,
            "Vector similarity index dimensions mismatch. Index definition expects {}, persisted index has {}",
            dimensions_, persisted_dimensions);

    if ((persisted_partition_count != 0 && persisted_dimensions > std::numeric_limits<size_t>::max() / persisted_partition_count)
        || (persisted_row_count != 0 && persisted_dimensions > std::numeric_limits<size_t>::max() / persisted_row_count)
        || (persisted_row_count == 0 && (persisted_partition_count != 0 || persisted_probe_count != 0))
        || (persisted_row_count != 0
            && (persisted_partition_count == 0
                || persisted_partition_count > persisted_row_count
                || persisted_partition_count > MAX_IVF_PARTITIONS
                || persisted_probe_count == 0
                || persisted_probe_count > persisted_partition_count))
        || persisted_row_count > std::numeric_limits<UInt32>::max())
        throw Exception(ErrorCodes::INCORRECT_DATA, "Persisted IVF vector similarity index has inconsistent dimensions or counts");

    row_count = persisted_row_count;
    partition_count = persisted_partition_count;
    probe_count = persisted_probe_count;

    const size_t centroid_values = partition_count * dimensions_;
    const size_t quantized_values = row_count * dimensions_;

    readVectorBinaryWithExpectedSize(centroids, istr, centroid_values, "centroids");
    readVectorBinaryWithExpectedSize(partition_dimension_scales, istr, centroid_values, "partition_dimension_scales");
    readVectorBinaryWithExpectedSize(partition_offsets, istr, partition_count + 1, "partition_offsets");
    readVectorBinaryWithExpectedSize(row_ids, istr, row_count, "row_ids");
    readVectorBinaryWithExpectedSize(quantized_vectors, istr, quantized_values, "quantized_vectors");

    if ((row_count == 0 && (partition_count != 0 || probe_count != 0))
        || (row_count != 0 && (partition_count == 0 || probe_count == 0 || probe_count > partition_count))
        || partition_offsets.front() != 0
        || partition_offsets.back() != row_count)
        throw Exception(ErrorCodes::INCORRECT_DATA, "Persisted IVF vector similarity index has inconsistent partition metadata");

    for (size_t partition = 0; partition != partition_count; ++partition)
    {
        if (partition_offsets[partition] > partition_offsets[partition + 1])
            throw Exception(ErrorCodes::INCORRECT_DATA, "Persisted IVF vector similarity index has unsorted partition offsets");
    }

    for (const Float32 centroid : centroids)
    {
        if (!std::isfinite(centroid))
            throw Exception(ErrorCodes::INCORRECT_DATA, "Persisted IVF vector similarity index has non-finite centroid values");
    }

    for (const Float32 scale : partition_dimension_scales)
    {
        if (!std::isfinite(scale) || scale <= 0.0f)
            throw Exception(ErrorCodes::INCORRECT_DATA, "Persisted IVF vector similarity index has invalid residual scales");
    }

    for (const Int8 quantized_value : quantized_vectors)
    {
        if (quantized_value < -127)
            throw Exception(ErrorCodes::INCORRECT_DATA, "Persisted IVF vector similarity index has quantized residual value outside supported range");
    }

    std::vector<UInt8> row_id_seen(row_count, 0);
    for (const UInt32 row_id : row_ids)
    {
        if (row_id >= row_count)
            throw Exception(ErrorCodes::INCORRECT_DATA, "Persisted IVF vector similarity index has row id {} outside row count {}", row_id, row_count);
        if (row_id_seen[row_id])
            throw Exception(ErrorCodes::INCORRECT_DATA, "Persisted IVF vector similarity index has duplicate row id {}", row_id);
        row_id_seen[row_id] = 1;
    }
}

NearestNeighbours PartitionedQuantizedIndexWithSerialization::search(
    const std::vector<Float64> & reference_vector,
    size_t limit,
    bool return_distances,
    size_t /*expansion_search*/) const
{
    if (limit == 0 || row_count == 0)
        return {};

    std::vector<Float32> reference_vector_float(dimensions_);
    for (size_t dimension = 0; dimension != dimensions_; ++dimension)
        reference_vector_float[dimension] = convertVectorElementToFloat32ForPartitionedQuantizedIndex(
            reference_vector[dimension],
            ErrorCodes::INCORRECT_QUERY,
            "reference vector in the SELECT query");

    Float64 reference_norm_squared = 0.0;
    if (metric_kind == unum::usearch::metric_kind_t::cos_k)
        reference_norm_squared = squaredNorm(reference_vector_float.data(), dimensions_);

    struct PartitionScore
    {
        Float32 score;
        UInt32 partition;
    };

    std::vector<PartitionScore> partition_scores;
    partition_scores.reserve(partition_count);
    for (UInt32 partition = 0; partition != partition_count; ++partition)
    {
        partition_scores.push_back(
            {.score = scoreVectorAgainstCentroid(
                 reference_vector_float.data(),
                 &centroids[partition * dimensions_],
                 reference_norm_squared,
                 dimensions_,
                 metric_kind),
             .partition = partition});
    }

    std::sort(
        partition_scores.begin(),
        partition_scores.end(),
        [this](const PartitionScore & lhs, const PartitionScore & rhs)
        {
            if (isBetterVectorSearchScore(lhs.score, rhs.score, metric_kind))
                return true;
            if (isBetterVectorSearchScore(rhs.score, lhs.score, metric_kind))
                return false;
            return lhs.partition < rhs.partition;
        });

    struct Candidate
    {
        Float32 score;
        UInt64 row;
    };

    const auto better = [this](const Candidate & lhs, const Candidate & rhs)
    {
        if (isBetterVectorSearchScore(lhs.score, rhs.score, metric_kind))
            return true;
        if (isBetterVectorSearchScore(rhs.score, lhs.score, metric_kind))
            return false;
        return lhs.row < rhs.row;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(std::min(limit, row_count));
    const size_t target_candidate_count = std::min(limit, row_count);
    size_t probed_partitions = 0;
    size_t computed_distances = partition_count;
    auto scoring_context = createPartitionScoringContext(dimensions_, metric_kind);
    for (size_t probe = 0; probe != partition_scores.size(); ++probe)
    {
        if (probe >= probe_count && candidates.size() >= target_candidate_count)
            break;

        const UInt32 partition = partition_scores[probe].partition;
        const Float32 * centroid = &centroids[partition * dimensions_];
        const Float32 * scales = &partition_dimension_scales[partition * dimensions_];

        const UInt32 begin = partition_offsets[partition];
        const UInt32 end = partition_offsets[partition + 1];
        ++probed_partitions;
        computed_distances += end - begin;

        preparePartitionScoringContext(
            scoring_context,
            centroid,
            scales,
            reference_vector_float.data(),
            dimensions_,
            metric_kind);

        for (UInt32 position = begin; position != end; ++position)
        {
            const Int8 * quantized_vector = &quantized_vectors[position * dimensions_];
            const Float32 score = scoreApproximateVectorAgainstReference(
                quantized_vector,
                scoring_context,
                reference_norm_squared,
                dimensions_,
                metric_kind);

            Candidate candidate{.score = score, .row = row_ids[position]};
            if (candidates.size() < limit)
            {
                candidates.push_back(candidate);
                std::push_heap(candidates.begin(), candidates.end(), better);
            }
            else if (better(candidate, candidates.front()))
            {
                std::pop_heap(candidates.begin(), candidates.end(), better);
                candidates.back() = candidate;
                std::push_heap(candidates.begin(), candidates.end(), better);
            }
        }
    }

    std::sort_heap(candidates.begin(), candidates.end(), better);

    NearestNeighbours result;
    result.default_distance = getDefaultVectorSearchScore(metric_kind);
    result.rows.reserve(candidates.size());
    if (return_distances)
        result.distances = std::vector<Float32>();
    if (result.distances)
        result.distances->reserve(candidates.size());

    for (const auto & candidate : candidates)
    {
        result.rows.push_back(candidate.row);
        if (result.distances)
            result.distances->push_back(candidate.score);
    }

    ProfileEvents::increment(ProfileEvents::IVFSearchCount);
    ProfileEvents::increment(ProfileEvents::IVFSearchProbes, probed_partitions);
    ProfileEvents::increment(ProfileEvents::IVFSearchComputedDistances, computed_distances);

    return result;
}

size_t PartitionedQuantizedIndexWithSerialization::memoryUsageBytes() const
{
    return sizeof(*this)
        + centroids.capacity() * sizeof(Float32)
        + partition_dimension_scales.capacity() * sizeof(Float32)
        + partition_offsets.capacity() * sizeof(UInt32)
        + row_ids.capacity() * sizeof(UInt32)
        + quantized_vectors.capacity() * sizeof(Int8);
}

String PartitionedQuantizedIndexWithSerialization::statisticsString() const
{
    return fmt::format(
        "method = ivf, partitions = {}, probes = {}, scale = per_dimension, size = {}, memory_usage = {}, bytes_per_vector = {}",
        partition_count,
        probe_count,
        row_count,
        ReadableSize(memoryUsageBytes()),
        row_count == 0 ? 0 : memoryUsageBytes() / row_count);
}

MergeTreeIndexGranuleVectorSimilarity::MergeTreeIndexGranuleVectorSimilarity(
    const String & index_name_,
    VectorSimilarityIndexParams params_)
    : MergeTreeIndexGranuleVectorSimilarity(index_name_, params_, nullptr)
{
}

MergeTreeIndexGranuleVectorSimilarity::MergeTreeIndexGranuleVectorSimilarity(
    const String & index_name_,
    VectorSimilarityIndexParams params_,
    VectorSimilarityIndexWithSerializationPtr index_)
    : index_name(index_name_)
    , params(params_)
    , index(std::move(index_))
{
}

void MergeTreeIndexGranuleVectorSimilarity::serializeBinary(WriteBuffer & ostr) const
{
    LOG_TRACE(logger, "Start writing vector similarity index");

    if (empty())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Attempt to write empty vector similarity index {}", backQuote(index_name));

    writeIntBinary(FILE_FORMAT_VERSION, ostr);
    writeIntBinary(static_cast<UInt8>(params.method), ostr);
    writeIntBinary(static_cast<UInt64>(params.metric_kind), ostr);
    writeIntBinary(static_cast<UInt64>(params.scalar_kind), ostr);

    /// Number of dimensions is required in the index constructor,
    /// so it must be written and read separately from the other part
    writeIntBinary(static_cast<UInt64>(index->dimensions()), ostr);

    index->serialize(ostr);

    LOG_TRACE(logger, "Wrote vector similarity index: {}", index->statisticsString());
}

void MergeTreeIndexGranuleVectorSimilarity::deserializeBinary(ReadBuffer & istr, MergeTreeIndexVersion /*version*/)
{
    LOG_TRACE(logger, "Start loading vector similarity index");

    UInt64 file_version;
    readIntBinary(file_version, istr);
    if (file_version != 1 && file_version != 2 && file_version != FILE_FORMAT_VERSION)
        throw Exception(
            ErrorCodes::FORMAT_VERSION_TOO_OLD,
            "Vector similarity index could not be loaded because its version is too old (current version: {}, persisted version: {}). Please drop the index and create it again.",
            FILE_FORMAT_VERSION, file_version);
        /// More fancy error handling would be: Set a flag on the index that it failed to load. During usage return all granules, i.e.
        /// behave as if the index does not exist. Since format changes are expected to happen only rarely and it is "only" an index, keep it simple for now.

    VectorSimilarityIndexMethod persisted_method = VectorSimilarityIndexMethod::HNSW;
    if (file_version >= 2)
    {
        UInt8 method;
        readIntBinary(method, istr);
        persisted_method = static_cast<VectorSimilarityIndexMethod>(method);
    }

    if (persisted_method != params.method)
        throw Exception(
            ErrorCodes::INCORRECT_DATA,
            "Vector similarity index method mismatch. Index definition expects {}, persisted index has {}",
            static_cast<UInt32>(params.method),
            static_cast<UInt32>(persisted_method));

    if (file_version == FILE_FORMAT_VERSION)
    {
        UInt64 persisted_metric_kind;
        UInt64 persisted_scalar_kind;
        readIntBinary(persisted_metric_kind, istr);
        readIntBinary(persisted_scalar_kind, istr);

        if (persisted_metric_kind != static_cast<UInt64>(params.metric_kind))
            throw Exception(
                ErrorCodes::INCORRECT_DATA,
                "Vector similarity index metric mismatch. Index definition expects {}, persisted index has {}",
                static_cast<UInt64>(params.metric_kind),
                persisted_metric_kind);

        if (persisted_scalar_kind != static_cast<UInt64>(params.scalar_kind))
            throw Exception(
                ErrorCodes::INCORRECT_DATA,
                "Vector similarity index scalar kind mismatch. Index definition expects {}, persisted index has {}",
                static_cast<UInt64>(params.scalar_kind),
                persisted_scalar_kind);
    }

    UInt64 dimensions;
    readIntBinary(dimensions, istr);

    if (params.method == VectorSimilarityIndexMethod::HNSW)
        index = std::make_shared<USearchIndexWithSerialization>(dimensions, params.metric_kind, params.scalar_kind, params.hnsw);
    else
        index = std::make_shared<PartitionedQuantizedIndexWithSerialization>(dimensions, params.metric_kind, params.ivf);

    index->deserialize(istr);

    LOG_TRACE(logger, "Loaded vector similarity index: {}", index->statisticsString());
}

MergeTreeIndexAggregatorVectorSimilarity::MergeTreeIndexAggregatorVectorSimilarity(
    const String & index_name_,
    const Block & index_sample_block_,
    UInt64 dimensions_,
    VectorSimilarityIndexParams params_)
    : index_name(index_name_)
    , index_sample_block(index_sample_block_)
    , dimensions(dimensions_)
    , params(params_)
{
}

MergeTreeIndexGranulePtr MergeTreeIndexAggregatorVectorSimilarity::getGranuleAndReset()
{
    VectorSimilarityIndexWithSerializationPtr index_for_granule;
    if (params.method == VectorSimilarityIndexMethod::HNSW)
    {
        index_for_granule = index;
        index = nullptr;
    }
    else
    {
        auto ivf_index = std::make_shared<PartitionedQuantizedIndexWithSerialization>(dimensions, params.metric_kind, params.ivf);
        ivf_index->build(std::move(pending_vectors));
        pending_vectors.clear();
        index_for_granule = std::move(ivf_index);
    }

    auto granule = std::make_shared<MergeTreeIndexGranuleVectorSimilarity>(index_name, params, std::move(index_for_granule));
    index = nullptr;
    return granule;
}

namespace
{

/// Check inputs to prevent undefined behavior further down in USearch and to reject vectors that make the distance undefined.
/// - No vector element is +inf, -inf or nan.
/// - For distance functions that cannot handle zero magnitude vectors, additionally require a non-zero magnitude.
template <typename T>
void checkVectorIsSane(
    const T * vector,
    size_t dimension,
    unum::usearch::scalar_kind_t scalar_kind,
    int error_code,
    std::string_view context,
    bool reject_zero_magnitude_for_i8 = true)
{
    double magnitude_squared = 0.0;
    for (size_t i = 0; i != dimension; ++i)
    {
        T casted = static_cast<T>(vector[i]);
        if constexpr (std::is_same_v<T, BFloat16>)
        {
            if (!casted.isFinite())
                throw Exception(error_code,
                    "Vector for vector similarity index ({}) must not contain non-finite values (NaN or Inf)", context);
        }
        else
        {
            if (!std::isfinite(casted))
                throw Exception(error_code,
                    "Vector for vector similarity index ({}) must not contain non-finite values (NaN or Inf)", context);
        }

        if (scalar_kind == unum::usearch::scalar_kind_t::i8_k)
        {
            double v = static_cast<double>(vector[i]);
            magnitude_squared += v * v;
        }
    }

    if (reject_zero_magnitude_for_i8 && scalar_kind == unum::usearch::scalar_kind_t::i8_k && magnitude_squared == 0.0)
        throw Exception(error_code,
            "Zero-magnitude vectors for vector similarity index ({}) are not supported with `i8` quantization", context);
}

size_t calculateVectorIndexFetchLimit(size_t query_limit, float index_fetch_multiplier, size_t max_limit)
{
    /// The multiplier is only meant to fetch extra candidates for post-filtering or rescoring.
    /// It must not shrink the request below the query LIMIT, otherwise the index path can return too few rows.
    if (query_limit >= max_limit)
        return max_limit;

    const double multiplied_limit_float = static_cast<double>(query_limit) * static_cast<double>(index_fetch_multiplier);
    if (multiplied_limit_float >= static_cast<double>(max_limit))
        return max_limit;

    return std::max(query_limit, static_cast<size_t>(multiplied_limit_float));
}

template <typename Column>
void updateImpl(const ColumnArray * column_array, const ColumnArray::Offsets & column_array_offsets, USearchIndexWithSerializationPtr & index, size_t dimensions, [[maybe_unused]] unum::usearch::scalar_kind_t scalar_kind, size_t rows)
{
    const auto & column_array_data = column_array->getData();
    const auto & column_array_data_float = typeid_cast<const Column &>(column_array_data);
    const auto & column_array_data_float_data = column_array_data_float.getData();

    /// Check all sizes are the same
    for (size_t row = 0; row < rows - 1; ++row)
        if (column_array_offsets[row + 1] - column_array_offsets[row] != dimensions)
            throw Exception(ErrorCodes::INCORRECT_DATA, "All arrays in column with vector similarity index must have equal length");

    /// Reserving space is mandatory
    size_t max_thread_pool_size = Context::getGlobalContextInstance()->getServerSettings()[ServerSetting::max_build_vector_similarity_index_thread_pool_size];
    if (max_thread_pool_size == 0)
        max_thread_pool_size = getNumberOfCPUCoresToUse();
    unum::usearch::index_limits_t limits(roundUpToPowerOfTwoOrZero(index->size() + rows), max_thread_pool_size);
    index->reserve(limits);

    /// Vector index creation is slooooow. Add the new rows in parallel. The threadpool is global to avoid oversubscription when multiple
    /// indexes are build simultaneously (e.g. multiple merges run at the same time).
    auto & thread_pool = Context::getGlobalContextInstance()->getBuildVectorSimilarityIndexThreadPool();

    /// The lambda must be declared before the runner so that during stack unwinding
    /// the runner is destroyed first (waits for all tasks) and the lambda is destroyed second.
    auto add_vector_to_index = [&](USearchIndex::vector_key_t key, size_t row)
    {
        /// Check if the query has been cancelled. USearch internally does not check for cancellation,
        /// and a single `add` call can take a very long time under sanitizers. Without this check, KILL QUERY
        /// cannot stop the index building. The check is cheap (reads an atomic flag).
        if (auto query_context = CurrentThread::tryGetQueryContext())
            if (auto query_status = query_context->getProcessListElementSafe())
                query_status->throwIfKilled();

        const size_t vector_start = row == 0 ? 0 : column_array_offsets[row - 1];
        const typename Column::ValueType & value = column_array_data_float_data[vector_start];

        checkVectorIsSane(&value, dimensions, scalar_kind, ErrorCodes::INCORRECT_DATA, "indexed vector");

        unum::usearch::index_dense_t::add_result_t result;

        /// Note: add is thread-safe
        if constexpr (std::is_same_v<Column, ColumnBFloat16>)
        {
            /// bf16 was standardized with C++23 but libcxx does not support it yet.
            /// As a result, ClickHouse and usearch each emulate bf16 and we need to implement some ugly special handling for bf16 below.
            result = index->add(key, reinterpret_cast<const unum::usearch::bf16_bits_t *>(&value.raw()));
        }
        else
        {
            static_assert(std::is_same_v<Column, ColumnFloat32> || std::is_same_v<Column, ColumnFloat64>);
            result = index->add(key, &value);
        }

        if (!result)
            throw Exception(ErrorCodes::INCORRECT_DATA, "Could not add data to vector similarity index. Error: {}", result.error.release());

        ProfileEvents::increment(ProfileEvents::USearchAddCount);
        ProfileEvents::increment(ProfileEvents::USearchAddVisitedMembers, result.visited_members);
        ProfileEvents::increment(ProfileEvents::USearchAddComputedDistances, result.computed_distances);
    };


    size_t index_size = index->size();
    ThreadPoolCallbackRunnerLocal<void> runner(thread_pool, ThreadName::MERGETREE_VECTOR_SIM_INDEX);
    for (size_t row = 0; row < rows; ++row)
    {
        auto key = static_cast<USearchIndex::vector_key_t>(index_size + row);
        /// Passing add_vector_to_index by reference is safe because it outlives the runner
        runner.enqueueAndKeepTrack([&add_vector_to_index, key, row] { add_vector_to_index(key, row); });
    }

    runner.waitForAllToFinishAndRethrowFirstError();
}

template <typename Column>
void appendToPartitionedQuantizedIndex(
    const ColumnArray * column_array,
    const ColumnArray::Offsets & column_array_offsets,
    std::vector<Float32> & vectors,
    size_t dimensions,
    [[maybe_unused]] unum::usearch::scalar_kind_t scalar_kind,
    size_t rows,
    bool reject_zero_magnitude)
{
    const auto & column_array_data = column_array->getData();
    const auto & column_array_data_float = typeid_cast<const Column &>(column_array_data);
    const auto & column_array_data_float_data = column_array_data_float.getData();

    if (dimensions != 0 && rows > (std::numeric_limits<size_t>::max() - vectors.size()) / dimensions)
        throw Exception(ErrorCodes::INCORRECT_DATA, "Size of IVF vector similarity index would exceed the maximum vector buffer size");

    vectors.reserve(vectors.size() + rows * dimensions);

    for (size_t row = 0; row != rows; ++row)
    {
        const size_t vector_start = row == 0 ? 0 : column_array_offsets[row - 1];
        const size_t vector_end = column_array_offsets[row];
        if (vector_end - vector_start != dimensions)
            throw Exception(ErrorCodes::INCORRECT_DATA, "All arrays in column with vector similarity index must have equal length");

        const typename Column::ValueType & value = column_array_data_float_data[vector_start];
        checkVectorIsSane(&value, dimensions, scalar_kind, ErrorCodes::INCORRECT_DATA, "indexed vector", reject_zero_magnitude);

        for (size_t dimension = 0; dimension != dimensions; ++dimension)
            vectors.push_back(convertVectorElementToFloat32ForPartitionedQuantizedIndex(
                column_array_data_float_data[vector_start + dimension],
                ErrorCodes::INCORRECT_DATA,
                "indexed vector"));
    }
}

}

void MergeTreeIndexAggregatorVectorSimilarity::update(const Block & block, size_t * pos, size_t limit)
{
    if (*pos >= block.rows())
        throw Exception(
            ErrorCodes::LOGICAL_ERROR,
            "The provided position is not less than the number of block rows. Position: {}, Block rows: {}.",
            *pos, block.rows());

    size_t rows_read = std::min(limit, block.rows() - *pos);

    if (rows_read == 0)
        return;

    if (rows_read > std::numeric_limits<UInt32>::max())
        throw Exception(ErrorCodes::INCORRECT_DATA, "Index granularity is too big: more than {} rows per index granule.", std::numeric_limits<UInt32>::max());

    if (index_sample_block.columns() > 1)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected that index is build over a single column");

    const auto & index_column_name = index_sample_block.getByPosition(0).name;

    const auto & index_column = block.getByName(index_column_name).column;
    ColumnPtr column_cut = index_column->cut(*pos, rows_read);

    const auto * column_array = typeid_cast<const ColumnArray *>(column_cut.get());
    if (!column_array)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected Array(Float32|Float64|BFloat16) column");

    if (column_array->empty())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Array is unexpectedly empty");

    const size_t rows = column_array->size();

    const auto & column_array_offsets = column_array->getOffsets();
    const size_t dimensions_inserted = column_array_offsets[0];

    if (dimensions != dimensions_inserted)
        throw Exception(ErrorCodes::INCORRECT_DATA, "Array values in column with vector similarity index have {} elements, expects {} elements", dimensions_inserted, dimensions);

    if (params.method == VectorSimilarityIndexMethod::HNSW && !index)
        index = std::make_shared<USearchIndexWithSerialization>(dimensions, params.metric_kind, params.scalar_kind, params.hnsw);

    /// We use Usearch's index_dense_t as index type which supports only 4 bio entries according to https://github.com/unum-cloud/usearch/tree/main/cpp
    const size_t index_size = params.method == VectorSimilarityIndexMethod::HNSW ? index->size() : pending_vectors.size() / dimensions;
    if (index_size + rows > std::numeric_limits<UInt32>::max())
        throw Exception(ErrorCodes::INCORRECT_DATA, "Size of vector similarity index would exceed 4 billion entries");

    const auto * data_type_array = typeid_cast<const DataTypeArray *>(block.getByName(index_column_name).type.get());
    if (!data_type_array)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected data type Array(Float32|Float64|BFloat16)");

    const TypeIndex nested_type_index = data_type_array->getNestedType()->getTypeId();
    WhichDataType which(nested_type_index);
    if (which.isFloat32())
    {
        if (params.method == VectorSimilarityIndexMethod::HNSW)
            updateImpl<ColumnFloat32>(column_array, column_array_offsets, index, dimensions, params.scalar_kind, rows);
        else
            appendToPartitionedQuantizedIndex<ColumnFloat32>(
                column_array,
                column_array_offsets,
                pending_vectors,
                dimensions,
                params.scalar_kind,
                rows,
                params.metric_kind == unum::usearch::metric_kind_t::cos_k);
    }
    else if (which.isFloat64())
    {
        if (params.method == VectorSimilarityIndexMethod::HNSW)
            updateImpl<ColumnFloat64>(column_array, column_array_offsets, index, dimensions, params.scalar_kind, rows);
        else
            appendToPartitionedQuantizedIndex<ColumnFloat64>(
                column_array,
                column_array_offsets,
                pending_vectors,
                dimensions,
                params.scalar_kind,
                rows,
                params.metric_kind == unum::usearch::metric_kind_t::cos_k);
    }
    else if (which.isBFloat16())
    {
        if (params.method == VectorSimilarityIndexMethod::HNSW)
            updateImpl<ColumnBFloat16>(column_array, column_array_offsets, index, dimensions, params.scalar_kind, rows);
        else
            appendToPartitionedQuantizedIndex<ColumnBFloat16>(
                column_array,
                column_array_offsets,
                pending_vectors,
                dimensions,
                params.scalar_kind,
                rows,
                params.metric_kind == unum::usearch::metric_kind_t::cos_k);
    }
    else
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected data type Array(Float*)");


    *pos += rows_read;
}

MergeTreeIndexConditionVectorSimilarity::MergeTreeIndexConditionVectorSimilarity(
    const std::optional<VectorSearchParameters> & parameters_,
    const String & index_column_,
    VectorSimilarityIndexParams params_,
    ContextPtr context)
    : parameters(parameters_)
    , index_column(index_column_)
    , params(params_)
    , expansion_search(context->getSettingsRef()[Setting::hnsw_candidate_list_size_for_search])
    , index_fetch_multiplier(context->getSettingsRef()[Setting::vector_search_index_fetch_multiplier])
    , max_limit(context->getSettingsRef()[Setting::max_limit_for_vector_search_queries])
    , is_rescoring(context->getSettingsRef()[Setting::vector_search_with_rescoring])
{
    static constexpr auto MAX_INDEX_FETCH_MULTIPLIER = 1000.0;

    if (params.method == VectorSimilarityIndexMethod::HNSW && expansion_search == 0)
        throw Exception(ErrorCodes::INVALID_SETTING_VALUE, "Setting 'hnsw_candidate_list_size_for_search' must not be 0");

    if (!std::isfinite(index_fetch_multiplier)
        || index_fetch_multiplier <= 0.0 || index_fetch_multiplier > MAX_INDEX_FETCH_MULTIPLIER
        || (parameters && !std::isfinite(index_fetch_multiplier * static_cast<double>(parameters->limit))))
            throw Exception(ErrorCodes::INVALID_SETTING_VALUE, "Setting 'vector_search_index_fetch_multiplier' must be greater than 0.0 and less than {}", MAX_INDEX_FETCH_MULTIPLIER);
}

bool MergeTreeIndexConditionVectorSimilarity::mayBeTrueOnGranule(MergeTreeIndexGranulePtr, const UpdatePartialDisjunctionResultFn & /*update_partial_disjunction_result_fn*/) const
{
    throw Exception(ErrorCodes::LOGICAL_ERROR, "mayBeTrueOnGranule is not supported for vector similarity indexes");
}

bool MergeTreeIndexConditionVectorSimilarity::alwaysUnknownOrTrue() const
{
    if (!parameters)
        return true;

    /// The vector similarity index was build on a specific column.
    /// It can only be used if the ORDER BY clause in the SELECT query is against the same column.
    if (parameters->column != index_column)
        return true;

    /// The vector similarity index was build for a specific distance function.
    /// It can only be used if the ORDER BY clause in the SELECT query uses the same distance function.
    if ((parameters->distance_function == "L2Distance" && params.metric_kind != unum::usearch::metric_kind_t::l2sq_k)
        || (parameters->distance_function == "cosineDistance" && params.metric_kind != unum::usearch::metric_kind_t::cos_k && params.metric_kind != unum::usearch::metric_kind_t::hamming_k)
        || (parameters->distance_function == "dotProduct" && params.metric_kind != unum::usearch::metric_kind_t::ip_k))
            return true;

    return false;
}

NearestNeighbours MergeTreeIndexConditionVectorSimilarity::calculateApproximateNearestNeighbors(MergeTreeIndexGranulePtr granule_) const
{
    if (!parameters)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected vector_search_parameters to be set");

    const auto granule = std::dynamic_pointer_cast<MergeTreeIndexGranuleVectorSimilarity>(granule_);
    if (granule == nullptr)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Granule has the wrong type");

    const VectorSimilarityIndexWithSerializationPtr index = granule->index;

    if (parameters->reference_vector.size() != index->dimensions())
        throw Exception(ErrorCodes::INCORRECT_QUERY, "The dimension of the reference vector in the query ({}) does not match the dimension in the index ({})",
            parameters->reference_vector.size(), index->dimensions());

    checkVectorIsSane(
        parameters->reference_vector.data(), parameters->reference_vector.size(),
        granule->params.scalar_kind,
        ErrorCodes::INCORRECT_QUERY,
        "reference vector in the SELECT query",
        params.method == VectorSimilarityIndexMethod::HNSW || params.metric_kind == unum::usearch::metric_kind_t::cos_k);

    size_t limit = parameters->limit;
    if (parameters->additional_filters_present || is_rescoring)
        /// Additional filters mean post-filtering which means that matches may be removed. To compensate, allow to fetch more rows by a factor.
        /// Similarly, if rescoring is on, fetch more neighbours from the index and pass them for the final re-ranking by ORDER BY ... LIMIT.
        limit = calculateVectorIndexFetchLimit(limit, index_fetch_multiplier, max_limit);

    return index->search(parameters->reference_vector, limit, parameters->return_distances, expansion_search);
}

MergeTreeIndexVectorSimilarity::MergeTreeIndexVectorSimilarity(
    const IndexDescription & index_,
    UInt64 dimensions_,
    VectorSimilarityIndexParams params_)
    : IMergeTreeIndex(index_)
    , dimensions(dimensions_)
    , params(params_)
{
}

MergeTreeIndexGranulePtr MergeTreeIndexVectorSimilarity::createIndexGranule() const
{
    return std::make_shared<MergeTreeIndexGranuleVectorSimilarity>(index.name, params);
}

MergeTreeIndexAggregatorPtr MergeTreeIndexVectorSimilarity::createIndexAggregator() const
{
    return std::make_shared<MergeTreeIndexAggregatorVectorSimilarity>(index.name, index.sample_block, dimensions, params);
}

MergeTreeIndexConditionPtr MergeTreeIndexVectorSimilarity::createIndexCondition(const ActionsDAG::Node * /*predicate*/, ContextPtr /*context*/) const
{
    throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Function not supported for vector similarity index");
}

MergeTreeIndexConditionPtr MergeTreeIndexVectorSimilarity::createIndexCondition(const ActionsDAG::Node * /*predicate*/, ContextPtr context, const std::optional<VectorSearchParameters> & parameters) const
{
    const String & index_column = index.column_names[0];
    return std::make_shared<MergeTreeIndexConditionVectorSimilarity>(parameters, index_column, params, context);
}

MergeTreeIndexPtr vectorSimilarityIndexCreator(const IndexDescription & index)
{
    FieldVector args = getFieldsFromIndexArgumentsAST(index.arguments);
    UInt64 dimensions = args[2].safeGet<UInt64>();
    const auto method = methodNameToMethod.at(args[0].safeGet<String>());

    /// Default parameters:
    VectorSimilarityIndexParams params;
    params.method = method;
    params.metric_kind = distanceFunctionToMetricKind.at(args[1].safeGet<String>());
    params.scalar_kind = method == VectorSimilarityIndexMethod::HNSW
        ? unum::usearch::scalar_kind_t::bf16_k
        : unum::usearch::scalar_kind_t::i8_k;

    /// Optional parameters:
    const bool has_six_args = (args.size() == 6);
    if (has_six_args)
    {
        params.scalar_kind = quantizationToScalarKind.at(args[3].safeGet<String>());

        if (method == VectorSimilarityIndexMethod::HNSW)
        {
            params.hnsw = {.connectivity  = args[4].safeGet<UInt64>(),
                           .expansion_add = args[5].safeGet<UInt64>()};
        }
        else
        {
            params.ivf = {.partitions = args[4].safeGet<UInt64>(),
                          .probes = args[5].safeGet<UInt64>()};
        }

        /// Special handling for binary quantization:
        if (params.scalar_kind == unum::usearch::scalar_kind_t::b1x8_k)
            params.metric_kind = unum::usearch::metric_kind_t::hamming_k;
    }

    return std::make_shared<MergeTreeIndexVectorSimilarity>(index, dimensions, params);
}

void vectorSimilarityIndexValidator(const IndexDescription & index, bool /* attach */)
{
    FieldVector args = getFieldsFromIndexArgumentsAST(index.arguments);
    const bool has_three_args = (args.size() == 3);
    const bool has_six_args = (args.size() == 6);

    /// Check number and type of arguments
    if (!has_three_args && !has_six_args)
        throw Exception(ErrorCodes::INCORRECT_QUERY, "Vector similarity index must have three or six arguments");
    if (args[0].getType() != Field::Types::String)
        throw Exception(ErrorCodes::INCORRECT_QUERY, "First argument of vector similarity index (method) must be of type String");
    if (args[1].getType() != Field::Types::String)
        throw Exception(ErrorCodes::INCORRECT_QUERY, "Second argument of vector similarity index (metric) must be of type String");
    if (args[2].getType() != Field::Types::UInt64)
        throw Exception(ErrorCodes::INCORRECT_QUERY, "Third argument of vector similarity index (dimensions) must be of type UInt64");
    if (has_six_args)
    {
        if (args[3].getType() != Field::Types::String)
            throw Exception(ErrorCodes::INCORRECT_QUERY, "Fourth argument of vector similarity index (quantization) must be of type String");
        if (args[4].getType() != Field::Types::UInt64)
            throw Exception(ErrorCodes::INCORRECT_QUERY, "Fifth argument of vector similarity index must be of type UInt64");
        if (args[5].getType() != Field::Types::UInt64)
            throw Exception(ErrorCodes::INCORRECT_QUERY, "Sixth argument of vector similarity index must be of type UInt64");
    }

    /// Check that passed arguments are supported
    if (!methods.contains(args[0].safeGet<String>()))
        throw Exception(ErrorCodes::INCORRECT_DATA, "First argument (method) of vector similarity index is not supported. Supported methods are: {}", joinByComma(methods));
    if (!distanceFunctionToMetricKind.contains(args[1].safeGet<String>()))
        throw Exception(ErrorCodes::INCORRECT_DATA, "Second argument (distance function) of vector similarity index is not supported. Supported distance function are: {}", joinByComma(distanceFunctionToMetricKind));
    if (args[2].safeGet<UInt64>() == 0)
        throw Exception(ErrorCodes::INCORRECT_DATA, "Third argument (dimensions) of vector similarity index must be > 0");

    const auto method = methodNameToMethod.at(args[0].safeGet<String>());
    if (has_six_args)
    {
        if (!quantizationToScalarKind.contains(args[3].safeGet<String>()))
            throw Exception(ErrorCodes::INCORRECT_DATA, "Fourth argument (quantization) of vector similarity index is not supported. Supported quantizations are: {}", joinByComma(quantizationToScalarKind));

        const auto quantization = quantizationToScalarKind.at(args[3].safeGet<String>());
        if (method == VectorSimilarityIndexMethod::IVF)
        {
            if (quantization != unum::usearch::scalar_kind_t::i8_k)
                throw Exception(ErrorCodes::INCORRECT_DATA, "IVF vector similarity index currently supports only 'i8' quantization");

            const UInt64 partitions = args[4].safeGet<UInt64>();
            const UInt64 probes = args[5].safeGet<UInt64>();
            if (partitions > MAX_IVF_PARTITIONS)
                throw Exception(
                    ErrorCodes::INCORRECT_DATA,
                    "IVF vector similarity index partition count must be less than or equal to {}, or 0 for automatic selection",
                    MAX_IVF_PARTITIONS);
            if (probes > MAX_IVF_PARTITIONS)
                throw Exception(
                    ErrorCodes::INCORRECT_DATA,
                    "IVF vector similarity index probe count must be less than or equal to {}, or 0 for automatic selection",
                    MAX_IVF_PARTITIONS);
            if (partitions != 0 && probes > partitions)
                throw Exception(ErrorCodes::INCORRECT_DATA, "IVF vector similarity index probe count must be less than or equal to partition count, or 0 for automatic selection");
        }
        /// More checks for binary quantization
        else if (quantization == unum::usearch::scalar_kind_t::b1x8_k)
        {
            if (distanceFunctionToMetricKind.at(args[1].safeGet<String>()) != unum::usearch::metric_kind_t::cos_k)
                throw Exception(ErrorCodes::INCORRECT_DATA, "Binary quantization in vector similarity index can only be used with the cosine distance as distance function");
            if (args[2].safeGet<UInt64>() % 8 != 0)
                throw Exception(ErrorCodes::INCORRECT_DATA, "Binary quantization in vector similarity index requires that the dimension is a multiple of 8");
        }

        if (method == VectorSimilarityIndexMethod::HNSW)
        {
            /// Call Usearch's own parameter validation method for HNSW-specific parameters
            UInt64 connectivity = args[4].safeGet<UInt64>();
            UInt64 expansion_add = args[5].safeGet<UInt64>();
            UInt64 expansion_search = default_expansion_search;
            unum::usearch::index_dense_config_t config(connectivity, expansion_add, expansion_search);
            if (auto error = config.validate(); error)
                throw Exception(ErrorCodes::INCORRECT_DATA, "Invalid parameters passed to vector similarity index. Error: {}", error.release());
        }
    }
    else if (method == VectorSimilarityIndexMethod::IVF)
    {
        /// Three-argument IVF uses i8 quantization with automatic partition and probe counts.
    }

    /// Check that the index is created on a single column
    if (index.column_names.size() != 1 || index.data_types.size() != 1)
        throw Exception(ErrorCodes::INCORRECT_NUMBER_OF_COLUMNS, "Vector similarity index must be created on a single column");

    /// Check that the data type is Array(Float32|Float64|BFloat16)
    DataTypePtr data_type = index.sample_block.getDataTypes()[0];
    const auto * data_type_array = typeid_cast<const DataTypeArray *>(data_type.get());
    if (!data_type_array)
        throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Vector similarity index can only be created on columns of type Array(Float32|Float64|BFloat16)");
    TypeIndex nested_type_index = data_type_array->getNestedType()->getTypeId();
    WhichDataType which(nested_type_index);
    if (!which.isNativeFloat() && !which.isBFloat16())
        throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Vector similarity index can only be created on columns of type Array(Float32|Float64|BFloat16)");
}

}

#endif
