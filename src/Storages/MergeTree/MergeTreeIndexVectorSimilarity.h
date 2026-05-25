#pragma once

#include "config.h"

#if USE_USEARCH

#include <Storages/MergeTree/MergeTreeIndices.h>
#include <Common/Logger.h>

/// Include immintrin. Otherwise `simsimd` fails to build: `unknown type name '__bfloat16'`
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include <usearch/index_dense.hpp>

namespace DB
{

/// Defaults for HNSW parameters. Instead of using the default parameters provided by USearch (default_connectivity(),
/// default_expansion_add(), default_expansion_search()), we experimentally came up with our own default parameters. They provide better
/// trade-offs with regards to index construction time, search precision and queries-per-second (speed).
static constexpr size_t default_connectivity = 32;
static constexpr size_t default_expansion_add = 128;
static constexpr size_t default_expansion_search = 256;

/// Parameters for HNSW index construction.
struct UsearchHnswParams
{
    size_t connectivity = default_connectivity;
    size_t expansion_add = default_expansion_add;
};

/// Parameters for the bulk-built partitioned quantized index.
struct PartitionedQuantizedParams
{
    /// 0 means choose automatically from the number of rows in the granule.
    size_t partitions = 0;
    /// 0 means choose automatically from the number of partitions.
    size_t probes = 0;
};

enum class VectorSimilarityIndexMethod : uint8_t
{
    HNSW = 1,
    IVF = 2,
};

struct VectorSimilarityIndexParams
{
    VectorSimilarityIndexMethod method = VectorSimilarityIndexMethod::HNSW;
    unum::usearch::metric_kind_t metric_kind = unum::usearch::metric_kind_t::l2sq_k;
    unum::usearch::scalar_kind_t scalar_kind = unum::usearch::scalar_kind_t::bf16_k;
    UsearchHnswParams hnsw;
    PartitionedQuantizedParams ivf;
};

class IVectorSimilarityIndexWithSerialization
{
public:
    virtual ~IVectorSimilarityIndexWithSerialization() = default;

    virtual void serialize(WriteBuffer & ostr) const = 0;
    virtual void deserialize(ReadBuffer & istr) = 0;
    virtual NearestNeighbours search(const std::vector<Float64> & reference_vector, size_t limit, bool return_distances, size_t expansion_search) const = 0;

    virtual size_t dimensions() const = 0;
    virtual size_t size() const = 0;
    virtual size_t memoryUsageBytes() const = 0;
    virtual String statisticsString() const = 0;
};

using VectorSimilarityIndexWithSerializationPtr = std::shared_ptr<IVectorSimilarityIndexWithSerialization>;

using USearchIndex = unum::usearch::index_dense_t;

class USearchIndexWithSerialization : public USearchIndex, public IVectorSimilarityIndexWithSerialization
{
    using Base = USearchIndex;

public:
    USearchIndexWithSerialization(
        size_t dimensions,
        unum::usearch::metric_kind_t metric_kind,
        unum::usearch::scalar_kind_t scalar_kind,
        UsearchHnswParams usearch_hnsw_params);

    void serialize(WriteBuffer & ostr) const override;
    void deserialize(ReadBuffer & istr) override;
    NearestNeighbours search(const std::vector<Float64> & reference_vector, size_t limit, bool return_distances, size_t expansion_search) const override;

    struct Statistics
    {
        size_t max_level;
        size_t connectivity;
        size_t size;                /// number of indexed vectors
        size_t capacity;            /// reserved number of indexed vectors
        size_t memory_usage;        /// byte size (not exact)
        size_t bytes_per_vector;
        size_t scalar_words;
        size_t nodes;
        size_t edges;
        size_t max_edges;

        std::vector<USearchIndex::stats_t> level_stats; /// for debugging, excluded from getStatistics()

        String toString() const;
    };

    Statistics getStatistics() const;

    size_t dimensions() const override { return Base::dimensions(); }
    size_t size() const override { return Base::size(); }
    size_t memoryUsageBytes() const override;
    String statisticsString() const override { return getStatistics().toString(); }

private:
    unum::usearch::metric_kind_t metric_kind;
};

using USearchIndexWithSerializationPtr = std::shared_ptr<USearchIndexWithSerialization>;

class PartitionedQuantizedIndexWithSerialization final : public IVectorSimilarityIndexWithSerialization
{
public:
    PartitionedQuantizedIndexWithSerialization(
        size_t dimensions_arg,
        unum::usearch::metric_kind_t metric_kind_,
        PartitionedQuantizedParams params_);

    void build(std::vector<Float32> && vectors_);

    void serialize(WriteBuffer & ostr) const override;
    void deserialize(ReadBuffer & istr) override;
    NearestNeighbours search(const std::vector<Float64> & reference_vector, size_t limit, bool return_distances, size_t expansion_search) const override;

    size_t dimensions() const override { return dimensions_; }
    size_t size() const override { return row_count; }
    size_t memoryUsageBytes() const override;
    String statisticsString() const override;

private:
    size_t choosePartitions(size_t rows) const;
    size_t chooseProbes(size_t partitions_) const;

    size_t dimensions_;
    unum::usearch::metric_kind_t metric_kind;
    PartitionedQuantizedParams params;

    size_t row_count = 0;
    size_t partition_count = 0;
    size_t probe_count = 0;
    std::vector<Float32> centroids;
    std::vector<Float32> partition_dimension_scales;
    std::vector<UInt32> partition_offsets;
    std::vector<UInt32> row_ids;
    std::vector<Int8> quantized_vectors;
};


struct MergeTreeIndexGranuleVectorSimilarity final : public IMergeTreeIndexGranule
{
    MergeTreeIndexGranuleVectorSimilarity(
        const String & index_name_,
        VectorSimilarityIndexParams params_);

    MergeTreeIndexGranuleVectorSimilarity(
        const String & index_name_,
        VectorSimilarityIndexParams params_,
        VectorSimilarityIndexWithSerializationPtr index_);

    ~MergeTreeIndexGranuleVectorSimilarity() override = default;

    void serializeBinary(WriteBuffer & ostr) const override;
    void deserializeBinary(ReadBuffer & istr, MergeTreeIndexVersion version) override;

    bool empty() const override { return !index || index->size() == 0; }

    size_t memoryUsageBytes() const override { return index ? index->memoryUsageBytes() : 0; }

    const String index_name;
    const VectorSimilarityIndexParams params;
    VectorSimilarityIndexWithSerializationPtr index;

    LoggerPtr logger = getLogger("VectorSimilarityIndex");

private:
    /// The version of the persistence format of USearch index. Increment whenever you change the format.
    /// Note: USearch prefixes the serialized data with its own version header. We can't rely on that because 1. the index in ClickHouse
    /// is (at least in theory) agnostic of specific vector search libraries, and 2. additional data (e.g. the number of dimensions)
    /// outside USearch exists which we should version separately.
    static constexpr UInt64 FILE_FORMAT_VERSION = 3;
};


struct MergeTreeIndexAggregatorVectorSimilarity final : IMergeTreeIndexAggregator
{
    MergeTreeIndexAggregatorVectorSimilarity(
        const String & index_name_,
        const Block & index_sample_block,
        UInt64 dimensions_,
        VectorSimilarityIndexParams params_);

    ~MergeTreeIndexAggregatorVectorSimilarity() override = default;

    bool empty() const override
    {
        if (params.method == VectorSimilarityIndexMethod::HNSW)
            return !index || index->size() == 0;
        return pending_vectors.empty();
    }
    MergeTreeIndexGranulePtr getGranuleAndReset() override;
    void update(const Block & block, size_t * pos, size_t limit) override;

    const String index_name;
    const Block index_sample_block;
    const UInt64 dimensions;
    const VectorSimilarityIndexParams params;
    USearchIndexWithSerializationPtr index;
    std::vector<Float32> pending_vectors;
};


class MergeTreeIndexConditionVectorSimilarity final : public IMergeTreeIndexCondition
{
public:
    explicit MergeTreeIndexConditionVectorSimilarity(
        const std::optional<VectorSearchParameters> & parameters_,
        const String & index_column_,
        VectorSimilarityIndexParams params_,
        ContextPtr context);

    ~MergeTreeIndexConditionVectorSimilarity() override = default;

    bool alwaysUnknownOrTrue() const override;
    bool mayBeTrueOnGranule(MergeTreeIndexGranulePtr granule, const UpdatePartialDisjunctionResultFn & update_partial_disjunction_result_fn) const override;
    NearestNeighbours calculateApproximateNearestNeighbors(MergeTreeIndexGranulePtr granule) const override;
    std::string getDescription() const override { return ""; }

private:
    std::optional<VectorSearchParameters> parameters;
    const String index_column;
    const VectorSimilarityIndexParams params;
    const size_t expansion_search;
    const float index_fetch_multiplier;
    const size_t max_limit;
    const bool is_rescoring;
};


class MergeTreeIndexVectorSimilarity : public IMergeTreeIndex
{
public:
    MergeTreeIndexVectorSimilarity(
        const IndexDescription & index_,
        UInt64 dimensions_,
        VectorSimilarityIndexParams params_);

    ~MergeTreeIndexVectorSimilarity() override = default;

    MergeTreeIndexGranulePtr createIndexGranule() const override;
    MergeTreeIndexAggregatorPtr createIndexAggregator() const override;
    MergeTreeIndexConditionPtr createIndexCondition(const ActionsDAG::Node * predicate, ContextPtr context) const override;
    MergeTreeIndexConditionPtr createIndexCondition(const ActionsDAG::Node * predicate, ContextPtr context, const std::optional<VectorSearchParameters> & parameters) const override;
    bool isVectorSimilarityIndex() const override { return true; }

private:
    const UInt64 dimensions;
    const VectorSimilarityIndexParams params;
};

}

#endif
