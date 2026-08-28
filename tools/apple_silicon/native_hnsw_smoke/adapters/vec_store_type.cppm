export module infinity_core:vec_store_type;

import :dist_func_l2;
import :plain_vec_store;

export namespace infinity {

template <typename DataT, typename CompressT, bool LSG = false>
class LVQCosVecStoreType;

template <typename DataT, typename CompressT, bool LSG = false>
class LVQL2VecStoreType;

template <typename DataT, typename CompressT, bool LSG = false>
class LVQIPVecStoreType;

template <typename DataT, bool LSG = false>
class RabitqCosVecStoreType;

template <typename DataT, bool LSG = false>
class RabitqL2VecStoreType;

template <typename DataT, bool LSG = false>
class RabitqIPVecStoreType;

template <typename DataT>
class NativePlainStoreBase {
public:
    using DataType = DataT;
    using CompressType = void;

    template <bool OwnMem>
    using Meta = PlainVecStoreMeta<DataType>;

    template <bool OwnMem>
    using Inner = PlainVecStoreInner<DataType, OwnMem>;

    using QueryVecType = const DataType *;
    using StoreType = typename Meta<true>::StoreType;
    using QueryType = typename Meta<true>::QueryType;
    using Distance = PlainL2Dist<DataType>;

    static constexpr bool IsPlainDense = true;
    static constexpr bool HasOptimize = false;
};

template <typename DataT, bool LSG = false>
class PlainCosVecStoreType : public NativePlainStoreBase<DataT> {
public:
    using DataType = DataT;
    static constexpr bool IsPlainDense = !LSG;

    template <typename CompressT>
    static constexpr LVQCosVecStoreType<DataType, CompressT, LSG> ToLVQ() {
        return {};
    }

    static constexpr RabitqCosVecStoreType<DataType, LSG> ToRabitq() { return {}; }
};

template <typename DataT, bool LSG = false>
class PlainIPVecStoreType : public NativePlainStoreBase<DataT> {
public:
    using DataType = DataT;
    static constexpr bool IsPlainDense = !LSG;

    template <typename CompressT>
    static constexpr LVQIPVecStoreType<DataType, CompressT, LSG> ToLVQ() {
        return {};
    }

    static constexpr RabitqIPVecStoreType<DataType, LSG> ToRabitq() { return {}; }
};

template <typename DataT, bool LSG = false>
class PlainL2VecStoreType : public NativePlainStoreBase<DataT> {
public:
    using DataType = DataT;
    static constexpr bool IsPlainDense = !LSG;

    template <typename CompressT>
    static constexpr LVQL2VecStoreType<DataType, CompressT, LSG> ToLVQ() {
        return {};
    }

    static constexpr RabitqL2VecStoreType<DataType, LSG> ToRabitq() { return {}; }
};

template <typename DataT, typename CompressT, bool LSG>
class LVQCosVecStoreType : public NativePlainStoreBase<DataT> {
public:
    using DataType = DataT;
    using CompressType = CompressT;
    static constexpr bool IsPlainDense = false;
    static constexpr bool HasOptimize = false;

    template <typename>
    static constexpr LVQCosVecStoreType ToLVQ() {
        return {};
    }

    static constexpr RabitqCosVecStoreType<DataType, LSG> ToRabitq() { return {}; }
};

template <typename DataT, typename CompressT, bool LSG>
class LVQL2VecStoreType : public NativePlainStoreBase<DataT> {
public:
    using DataType = DataT;
    using CompressType = CompressT;
    static constexpr bool IsPlainDense = false;
    static constexpr bool HasOptimize = false;

    template <typename>
    static constexpr LVQL2VecStoreType ToLVQ() {
        return {};
    }

    static constexpr RabitqL2VecStoreType<DataType, LSG> ToRabitq() { return {}; }
};

template <typename DataT, typename CompressT, bool LSG>
class LVQIPVecStoreType : public NativePlainStoreBase<DataT> {
public:
    using DataType = DataT;
    using CompressType = CompressT;
    static constexpr bool IsPlainDense = false;
    static constexpr bool HasOptimize = false;

    template <typename>
    static constexpr LVQIPVecStoreType ToLVQ() {
        return {};
    }

    static constexpr RabitqIPVecStoreType<DataType, LSG> ToRabitq() { return {}; }
};

template <typename DataT, bool LSG>
class RabitqCosVecStoreType : public NativePlainStoreBase<DataT> {
public:
    static constexpr bool IsPlainDense = false;

    template <typename>
    static constexpr RabitqCosVecStoreType ToLVQ() {
        return {};
    }
    static constexpr RabitqCosVecStoreType ToRabitq() { return {}; }
};

template <typename DataT, bool LSG>
class RabitqL2VecStoreType : public NativePlainStoreBase<DataT> {
public:
    static constexpr bool IsPlainDense = false;

    template <typename>
    static constexpr RabitqL2VecStoreType ToLVQ() {
        return {};
    }
    static constexpr RabitqL2VecStoreType ToRabitq() { return {}; }
};

template <typename DataT, bool LSG>
class RabitqIPVecStoreType : public NativePlainStoreBase<DataT> {
public:
    static constexpr bool IsPlainDense = false;

    template <typename>
    static constexpr RabitqIPVecStoreType ToLVQ() {
        return {};
    }
    static constexpr RabitqIPVecStoreType ToRabitq() { return {}; }
};

} // namespace infinity
