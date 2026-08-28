export module infinity_core:sparse_util;

import :infinity_type;

export namespace infinity {

template <typename DataType, typename IndexType>
struct SparseVecRef {
    SparseVecRef(i64 nnz, const IndexType *indices, const DataType *data) : nnz_(nnz), indices_(indices), data_(data) {}

    i64 nnz_;
    const IndexType *indices_;
    const DataType *data_;
};

} // namespace infinity
