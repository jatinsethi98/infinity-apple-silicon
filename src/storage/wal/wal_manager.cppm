// Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.
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

export module infinity_core:wal_manager;

import :blocking_queue;
import :log_file;
import :options;

namespace infinity {
// class FlushOptionType;

export class Storage;
class BGTaskProcessor;
class NewTxn;
class CheckpointTaskBase;
class ForceCheckpointTask;
class BottomExecutor;

struct WalEntry;

export enum class StorageMode {
    kUnInitialized,
    kAdmin,
    kReadable,
    kWritable,
};

export std::string ToString(StorageMode storage_mode) {
    switch (storage_mode) {
        case StorageMode::kUnInitialized: {
            return "Uninitialized";
        }
        case StorageMode::kAdmin: {
            return "Admin";
        }
        case StorageMode::kReadable: {
            return "Readable";
        }
        case StorageMode::kWritable: {
            return "Writable";
        }
    }
}

export struct ReplayWalOptions {
    bool on_startup_;
    bool is_replay_;
    bool sync_from_leader_;
};

export class WalManager {
public:
    WalManager(Storage *storage, std::string wal_dir, u64 wal_size_threshold, FlushOptionType flush_option);
    WalManager(Storage *storage, std::string wal_dir, std::string data_dir, u64 wal_size_threshold, FlushOptionType flush_option);

    ~WalManager();

    void Start();

    void Stop();

    void SubmitTxn(std::vector<NewTxn *> &txn_batch);

    // Flush is scheduled regularly. It collects a batch of transactions, sync
    // wal and do parallel committing. Each sync cost ~1s. Each checkpoint cost
    // ~10s. So it's necessary to sync for a batch of transactions, and to
    // checkpoint for a batch of sync.
    void NewFlush();

    void FlushLogByReplication(const std::vector<std::string> &synced_logs, bool on_startup);

    bool SetCheckpointing();
    bool UnsetCheckpoint();
    bool IsCheckpointing() const;

    void SwapWalFile(TxnTimeStamp max_commit_ts, bool error_if_duplicate);

    // Flush ofs_ and, per the configured durability level, push the WAL to the device.
    // Must be called before any transaction whose bytes are in this file is acknowledged,
    // and before the file is closed or swapped away.
    void SyncWal();

    // Open / close sync_fd_ alongside ofs_ for the current wal_path_.
    void OpenSyncFd();
    void CloseSyncFd();

    // fsync the WAL directory itself. Syncing a file does not make its directory entry
    // durable, so without this a power loss can lose a freshly created or freshly renamed
    // WAL file even though its data was synced.
    void SyncWalDirectory();

    // Number of completed WAL syncs. For tests: durability at the acknowledgement point
    // is asserted by observing that a sync happened, not by killing the process, which
    // the page cache makes indistinguishable.
    u64 SyncCount() const { return sync_count_.load(std::memory_order_relaxed); }

    std::string GetWalFilename() const;

    std::tuple<TransactionID, TxnTimeStamp, TxnTimeStamp> GetReplayEntries(StorageMode targe_storage_mode,
                                                                           std::vector<std::shared_ptr<WalEntry>> &replay_entries);

    std::tuple<TransactionID, TxnTimeStamp> ReplayWalEntries(const std::vector<std::shared_ptr<WalEntry>> &replay_entries);

    std::vector<std::shared_ptr<WalEntry>> CollectWalEntries() const;

    TxnTimeStamp LastCheckpointTS() const;
    void SetLastCheckpointTS(TxnTimeStamp new_last_ckp_ts);

    std::vector<std::shared_ptr<std::string>> GetDiffWalEntryString(TxnTimeStamp timestamp) const;
    void UpdateCommitState(TxnTimeStamp commit_ts, i64 wal_size);

public:
    std::tuple<TxnTimeStamp, i64> GetCommitState();
    void SetLastCkpWalSize(i64 wal_size);

private:
    i64 GetLastCkpWalSize();

public:
    u64 cfg_wal_size_threshold_{};

    const std::string &wal_dir() const { return wal_dir_; }
    const std::string &data_path() const { return data_path_; }

private:
    // Concurrent writing WAL is disallowed. So put all WAL writing into a queue
    // and do serial writing.
    std::string wal_dir_{};
    std::string wal_path_{};
    std::string data_path_{};

    Storage *storage_{};

    // WalManager state
    std::atomic<bool> running_{};
    std::thread new_flush_thread_{};

    // TxnManager and Flush thread access following members
    BlockingQueue<NewTxn *> new_wait_flush_{"WalManager"};

    // Only Flush thread access following members
    std::ofstream ofs_{};
    // A second descriptor on the SAME file as ofs_, used only to sync it. ofs_.flush()
    // moves bytes from the stream buffer into the page cache; only fsync/F_FULLFSYNC on a
    // descriptor gets them onto the device, and std::ofstream exposes no descriptor.
    // fsync acts on the file, not on the descriptor that wrote it, so syncing through a
    // separate fd for the same path is correct as long as ofs_ has been flushed first.
    // -1 when no WAL file is open.
    int sync_fd_{-1};
    FlushOptionType flush_option_{FlushOptionType::kNoSync};
    // Counts completed WAL syncs. Durability cannot be proven by killing the process --
    // the page cache serves the bytes back either way -- so tests assert on this instead.
    std::atomic<u64> sync_count_{0};
    std::unique_ptr<BottomExecutor> bottom_executor_{nullptr};

    // Flush and Checkpoint threads access following members
    mutable std::mutex mutex2_{};
    TxnTimeStamp max_commit_ts_{};
    TxnTimeStamp last_swap_wal_ts_{};
    i64 wal_size_{};
    i64 last_ckp_wal_size_{};
    std::atomic<bool> checkpoint_in_progress_{false};

    // Only Checkpoint/Cleanup thread access following members
    mutable std::mutex last_ckp_ts_mutex_{};
    TxnTimeStamp last_ckp_ts_{};
};

} // namespace infinity
