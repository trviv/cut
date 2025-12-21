#include <ComputeContainer.h>

#include <cstddef>
#include <string_view>

namespace cut {

// ComputeContainer implementation

ComputeContainer::ComputeContainer(std::string_view name) : name_(name) {}

ComputeContainer::~ComputeContainer() {
  refCounts_.clear();
  freeHandles_.clear();
}

size_t ComputeContainer::allocateSlot() {
  size_t index;
  if (!freeHandles_.empty()) {
    index = freeHandles_.back();
    freeHandles_.pop_back();
    refCounts_[index] = 0;
  } else {
    index = refCounts_.size();
    refCounts_.emplace_back(0);
  }
  return index;
}

void ComputeContainer::verify(const ComputeHandle &handle) const {
  if (handle.container_ != this) {
    throw std::runtime_error("Trying to get data for handle which does not "
                             "belong to the container");
  }
}

void ComputeContainer::addRef(const ComputeHandle &ref) {
  refCounts_[ref.id_]++;
}

void ComputeContainer::remRef(ComputeHandle &ref) {
  refCounts_[ref.id_]--;
  if (refCounts_[ref.id_] == 0) {
    destroy(ref);
    freeHandles_.push_back(ref.id_);
    ref.container_ = nullptr;
  }
}

} // namespace cut
