export module infinity_core:base_memindex;

import :chunk_index_meta;
import :memindex_tracer;
import internal_types;
import std.compat;

export namespace infinity {

class BaseMemIndex {
public:
    virtual ~BaseMemIndex() = default;
    virtual RowID GetBeginRowID() const = 0;
    virtual const ChunkIndexMetaInfo GetChunkIndexMetaInfo() const = 0;

protected:
    virtual MemIndexTracerInfo GetInfo() const = 0;
    void IncreaseMemoryUsageBase(size_t) {}
};

} // namespace infinity
