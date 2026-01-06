#include "ThreadPool.h"

namespace cut {

ThreadPool::ThreadPool(size_t numThreads) {
  if (numThreads == 0) {
    numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) {
      numThreads = 1; // Fallback if hardware_concurrency returns 0
    }
  }

  workers_.reserve(numThreads);
  for (size_t i = 0; i < numThreads; ++i) {
    workers_.emplace_back([this] { workerLoop(); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  taskCondition_.notify_all();

  for (auto &worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

void ThreadPool::submit(std::function<void()> task) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.push(std::move(task));
    ++activeTasks_;
  }
  taskCondition_.notify_one();
}

void ThreadPool::waitAll() {
  std::unique_lock<std::mutex> lock(mutex_);
  completeCondition_.wait(lock, [this] { return activeTasks_ == 0; });
}

void ThreadPool::workerLoop() {
  while (true) {
    std::function<void()> task;

    {
      std::unique_lock<std::mutex> lock(mutex_);
      taskCondition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

      if (stop_ && tasks_.empty()) {
        return;
      }

      task = std::move(tasks_.front());
      tasks_.pop();
    }

    // Execute task outside the lock
    task();

    // Signal completion
    {
      std::lock_guard<std::mutex> lock(mutex_);
      --activeTasks_;
    }
    completeCondition_.notify_all();
  }
}

} // namespace cut
