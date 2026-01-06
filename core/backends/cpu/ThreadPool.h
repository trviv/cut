#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace cut {

/**
 * A simple thread pool for parallel task execution.
 * Tasks are submitted to a queue and executed by worker threads.
 */
class ThreadPool {
public:
  /**
   * Constructs a thread pool with the specified number of worker threads.
   * @param numThreads Number of worker threads (0 = hardware_concurrency).
   */
  explicit ThreadPool(size_t numThreads = 0);

  /**
   * Destructor. Stops all worker threads and waits for them to finish.
   */
  ~ThreadPool();

  /// Deleted copy constructor.
  ThreadPool(const ThreadPool &) = delete;

  /// Deleted copy assignment.
  ThreadPool &operator=(const ThreadPool &) = delete;

  /**
   * Submits a task to the thread pool for execution.
   * @param task The function to execute.
   */
  void submit(std::function<void()> task);

  /**
   * Waits for all submitted tasks to complete.
   * Blocks until the task queue is empty and all tasks have finished.
   */
  void waitAll();

  /**
   * Returns the number of worker threads.
   */
  size_t numThreads() const { return workers_.size(); }

private:
  /// Worker thread function.
  void workerLoop();

  std::vector<std::thread> workers_;          ///< Worker threads.
  std::queue<std::function<void()>> tasks_;   ///< Task queue.
  std::mutex mutex_;                          ///< Protects task queue.
  std::condition_variable taskCondition_;     ///< Signals new tasks.
  std::condition_variable completeCondition_; ///< Signals task completion.
  std::atomic<size_t> activeTasks_{0};        ///< Number of tasks in flight.
  bool stop_{false};                          ///< Signals shutdown.
};

} // namespace cut
