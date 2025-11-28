#include <ComputeContainer.h>

namespace cut {

ComputeContainer::ComputeContainer(uint32_t type) : type_(type) {}

ComputeContainer::~ComputeContainer() {
  if (objects_.size() != freeHandles_.size()) {
    logErr("Trying to destroy container before all objects in it have "
           "been deallocated.");
  }
  objects_.clear();
  refCounts_.clear();
  freeHandles_.clear();
}

const ComputeContainer::HandleData &
ComputeContainer::data(const ComputeHandle &handle) const {
  if (!handle) {
    throw std::runtime_error("Trying to get data for an empty handle");
  }
  verify(handle);

  return objects_[handle.id_];
}

ComputeHandle ComputeContainer::createHandle(const HandleData &data) {
  size_t index;
  if (!freeHandles_.empty()) {
    index = freeHandles_.back();
    freeHandles_.pop_back();
    objects_[index] = data;
    refCounts_[index] = 0;
  } else {
    index = objects_.size();
    objects_.emplace_back(data);
    refCounts_.emplace_back(0);
  }

  return ComputeHandle(this, index);
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
    destroy(ref.id_);
    freeHandles_.push_back(ref.id_);
    ref.container_ = nullptr;
  }
}

} // namespace cut
