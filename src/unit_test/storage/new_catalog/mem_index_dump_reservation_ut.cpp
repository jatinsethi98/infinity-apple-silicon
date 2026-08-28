// Copyright(C) 2026 InfiniFlow, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

module;

#include "unit_test/gtest_expand.h"

module infinity_core:ut.mem_index_dump_reservation;

import :ut.base_test;
import :column_vector;
import :data_block;
import :db_meta;
import :defer_op;
import :hnsw_handler;
import :index_hnsw;
import :infinity_context;
import :mem_index;
import :memindex_tracer;
import :new_catalog;
import :new_txn;
import :new_txn_manager;
import :segment_index_meta;
import :table_def;
import :table_index_meta;
import :table_meta;
import :value;

import std;
import third_party;

import column_def;
import data_type;
import embedding_info;
import extra_ddl_info;
import logical_type;
import statement_common;

namespace infinity {

class InspectableMemIndexTracer final : public MemIndexTracer {
public:
    InspectableMemIndexTracer() : MemIndexTracer(1) {}

    void AddProposal(const std::shared_ptr<MemIndex> &mem_index, size_t dump_size) {
        std::lock_guard lock(mtx_);
        proposed_dump_[mem_index] = dump_size;
        proposed_dump_size_ += dump_size;
    }

    std::pair<size_t, size_t> ProposalState() const {
        std::lock_guard lock(mtx_);
        return {proposed_dump_.size(), proposed_dump_size_};
    }

    void TriggerDump(std::shared_ptr<DumpMemIndexTask>) override {}

protected:
    std::vector<std::shared_ptr<MemIndexDetail>> GetAllMemIndexes(NewTxn *) override { return {}; }
};

TEST(MemIndexDumpReservationTest, updater_retries_without_holding_catalog_lock) {
    constexpr auto kBlockedCheck = std::chrono::milliseconds(100);
    constexpr auto kDeadlockGuard = std::chrono::seconds(5);

    NewCatalog catalog(nullptr);
    const std::string key = "db/table/index/segment/mem_index";
    auto original = catalog.GetMemIndex(key, true);
    ASSERT_NE(original, nullptr);

    auto reserved = catalog.TryReserveMemIndexForDump(key, original);
    ASSERT_EQ(reserved, original);
    EXPECT_TRUE(original->IsDumping());

    std::promise<void> updater_started;
    auto updater_started_future = updater_started.get_future();
    auto updater = std::async(std::launch::async, [&] {
        updater_started.set_value();
        return catalog.GetMemIndex(key, true);
    });
    ASSERT_EQ(updater_started_future.wait_for(kDeadlockGuard), std::future_status::ready);
    EXPECT_EQ(updater.wait_for(kBlockedCheck), std::future_status::timeout);

    auto independent_lookup = std::async(std::launch::async, [&] { return catalog.GetMemIndex("independent", false); });
    EXPECT_EQ(independent_lookup.wait_for(kDeadlockGuard), std::future_status::ready);

    auto wrong_identity = std::make_shared<MemIndex>();
    EXPECT_FALSE(catalog.PopReservedMemIndex(key, wrong_identity));
    EXPECT_EQ(catalog.GetMemIndex(key, false), original);

    original->UpdateEnd();
    original->WaitUpdate();
    EXPECT_EQ(updater.wait_for(kBlockedCheck), std::future_status::timeout);

    EXPECT_TRUE(catalog.PopReservedMemIndex(key, original));
    original->SetIsDumping(false);

    ASSERT_EQ(updater.wait_for(kDeadlockGuard), std::future_status::ready);
    auto replacement = updater.get();
    ASSERT_NE(replacement, nullptr);
    EXPECT_NE(replacement, original);
    EXPECT_EQ(catalog.GetMemIndex(key, false), replacement);
    replacement->UpdateEnd();
}

TEST(MemIndexDumpReservationTest, rejected_dump_retains_identity_and_releases_proposal) {
    NewCatalog catalog(nullptr);
    InspectableMemIndexTracer tracer;
    const std::string key = "db/table/index/segment/rejected_mem_index";

    auto original = catalog.GetMemIndex(key, false);
    tracer.AddProposal(original, 64);
    const std::pair<size_t, size_t> proposed_state{1, 64};
    ASSERT_EQ(tracer.ProposalState(), proposed_state);

    auto reserved = catalog.TryReserveMemIndexForDump(key, original);
    ASSERT_EQ(reserved, original);

    reserved->SetIsDumping(false);
    tracer.DumpDone(reserved);

    const std::pair<size_t, size_t> cleaned_state{0, 0};
    EXPECT_EQ(tracer.ProposalState(), cleaned_state);
    EXPECT_EQ(catalog.GetMemIndex(key, false), original);

    auto update_lease = catalog.GetMemIndex(key, true);
    ASSERT_EQ(update_lease, original);
    update_lease->UpdateEnd();
}

class PoisonedHnswDumpTest : public BaseTestParamStr {};

INSTANTIATE_TEST_SUITE_P(PoisonedHnswDump, PoisonedHnswDumpTest, ::testing::Values(BaseTestParamStr::NEW_CONFIG_PATH));

TEST_P(PoisonedHnswDumpTest, rejected_dump_retains_catalog_identity_without_chunk_metadata) {
    NewTxnManager *txn_manager = InfinityContext::instance().storage()->new_txn_manager();
    const std::string db_name = "poisoned_hnsw_dump_db";
    const std::string table_name = "poisoned_hnsw_dump_table";
    const std::string index_name = "poisoned_hnsw_dump_index";

    auto embedding_info = std::make_shared<EmbeddingInfo>(EmbeddingDataType::kElemFloat, 4);
    auto column_def =
        std::make_shared<ColumnDef>(0, std::make_shared<DataType>(LogicalType::kEmbedding, embedding_info), "embedding", std::set<ConstraintType>());
    auto table_def = TableDef::Make(std::make_shared<std::string>(db_name),
                                    std::make_shared<std::string>(table_name),
                                    std::make_shared<std::string>(),
                                    {column_def});

    std::vector<InitParameter *> index_parameters;
    index_parameters.push_back(new InitParameter("metric", "l2"));
    DeferFn free_parameters([&] {
        for (auto *parameter : index_parameters) {
            delete parameter;
        }
    });
    auto index_def = IndexHnsw::Make(std::make_shared<std::string>(index_name),
                                     std::make_shared<std::string>(),
                                     "poisoned_hnsw_dump",
                                     {column_def->name()},
                                     index_parameters);

    {
        auto *txn = txn_manager->BeginTxn(std::make_unique<std::string>("create db"), TransactionType::kCreateDB);
        ASSERT_TRUE(txn->CreateDatabase(db_name, ConflictType::kError, std::make_shared<std::string>()).ok());
        ASSERT_TRUE(txn_manager->CommitTxn(txn).ok());
    }
    {
        auto *txn = txn_manager->BeginTxn(std::make_unique<std::string>("create table"), TransactionType::kCreateTable);
        ASSERT_TRUE(txn->CreateTable(db_name, table_def, ConflictType::kError).ok());
        ASSERT_TRUE(txn_manager->CommitTxn(txn).ok());
    }
    {
        auto *txn = txn_manager->BeginTxn(std::make_unique<std::string>("create index"), TransactionType::kCreateIndex);
        ASSERT_TRUE(txn->CreateIndex(db_name, table_name, index_def, ConflictType::kError).ok());
        ASSERT_TRUE(txn_manager->CommitTxn(txn).ok());
    }
    {
        auto block = std::make_shared<DataBlock>();
        auto column = ColumnVector::Make(column_def->type());
        column->Initialize();
        for (size_t row = 0; row < 64; ++row) {
            column->AppendValue(Value::MakeEmbedding(std::vector<float>{
                static_cast<float>(row), static_cast<float>(row + 1), static_cast<float>(row + 2), static_cast<float>(row + 3)}));
        }
        block->InsertVector(column, 0);
        block->Finalize();

        auto *txn = txn_manager->BeginTxn(std::make_unique<std::string>("append"), TransactionType::kAppend);
        ASSERT_TRUE(txn->Append(db_name, table_name, block).ok());
        ASSERT_TRUE(txn_manager->CommitTxn(txn).ok());
    }

    std::shared_ptr<MemIndex> original_mem_index;
    {
        auto *txn = txn_manager->BeginTxn(std::make_unique<std::string>("poison hnsw"), TransactionType::kRead);
        std::shared_ptr<DBMeta> db_meta;
        std::shared_ptr<TableMeta> table_meta;
        std::shared_ptr<TableIndexMeta> table_index_meta;
        std::string table_key;
        std::string index_key;
        ASSERT_TRUE(
            txn->GetTableIndexMeta(db_name, table_name, index_name, db_meta, table_meta, table_index_meta, &table_key, &index_key).ok());

        SegmentIndexMeta segment_index_meta(0, *table_index_meta);
        original_mem_index = segment_index_meta.GetMemIndex();
        ASSERT_NE(original_mem_index, nullptr);
        auto hnsw_index = original_mem_index->GetHnswIndex();
        ASSERT_NE(hnsw_index, nullptr);
        ASSERT_NE(hnsw_index->get(), nullptr);
        hnsw_index->get()->MarkBuildFailed();
        ASSERT_TRUE(hnsw_index->IsBuildFailed());

        auto [chunk_ids, chunk_status] = segment_index_meta.GetChunkIDs1();
        ASSERT_TRUE(chunk_status.ok());
        ASSERT_TRUE(chunk_ids->empty());
        ASSERT_TRUE(txn_manager->CommitTxn(txn).ok());
    }

    {
        auto *txn = txn_manager->BeginTxn(std::make_unique<std::string>("reject poisoned dump"), TransactionType::kDumpMemIndex);
        const Status status = txn->DumpMemIndex(db_name, table_name, index_name, 0);
        EXPECT_EQ(status.code(), ErrorCode::kInvalidMemIndex);
        ASSERT_TRUE(txn_manager->RollBackTxn(txn).ok());
    }

    {
        auto *txn = txn_manager->BeginTxn(std::make_unique<std::string>("verify rejected dump"), TransactionType::kRead);
        std::shared_ptr<DBMeta> db_meta;
        std::shared_ptr<TableMeta> table_meta;
        std::shared_ptr<TableIndexMeta> table_index_meta;
        std::string table_key;
        std::string index_key;
        ASSERT_TRUE(
            txn->GetTableIndexMeta(db_name, table_name, index_name, db_meta, table_meta, table_index_meta, &table_key, &index_key).ok());

        SegmentIndexMeta segment_index_meta(0, *table_index_meta);
        auto retained_mem_index = segment_index_meta.GetMemIndex();
        EXPECT_EQ(retained_mem_index, original_mem_index);
        EXPECT_FALSE(retained_mem_index->IsDumping());
        ASSERT_NE(retained_mem_index->GetHnswIndex(), nullptr);
        EXPECT_TRUE(retained_mem_index->GetHnswIndex()->IsBuildFailed());

        auto [chunk_ids, chunk_status] = segment_index_meta.GetChunkIDs1();
        ASSERT_TRUE(chunk_status.ok());
        EXPECT_TRUE(chunk_ids->empty());
        ASSERT_TRUE(txn_manager->CommitTxn(txn).ok());
    }
}

} // namespace infinity
