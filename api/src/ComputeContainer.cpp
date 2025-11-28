#include <ComputeContainer.h>

namespace cut {

// ComputeContainer implementation

ComputeContainer::ComputeContainer(uint32_t type) : type_(type) {}

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

ComputeHandle ComputeContainer::createHandleFromSlot(size_t index) {
  return ComputeHandle(this, index);
}

void ComputeContainer::verify(const ComputeHandle &handle) const {
  if (handle.container_ != this) {
    throw std::runtime_error("Trying to get data for handle which does not "
                             "belong to the container");
  }
}

void ComputeContainer::markSlotFree(size_t index) {
  freeHandles_.push_back(index);
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

// ComputeDataContainer implementation

ComputeDataContainer::ComputeDataContainer(uint32_t type)
    : ComputeContainer(type) {}

ComputeDataContainer::~ComputeDataContainer() {
  if (objects_.size() != freeSlotCount()) {
    logErr("Trying to destroy container before all objects in it have "
           "been deallocated.");
  }
  objects_.clear();
}

const ComputeDataContainer::HandleData &
ComputeDataContainer::data(const ComputeHandle &handle) const {
  if (!handle) {
    throw std::runtime_error("Trying to get data for an empty handle");
  }
  verify(handle);

  return objects_[handle.id_];
}

ComputeHandle ComputeDataContainer::createHandle(const HandleData &data) {
  size_t index = allocateSlot();

  if (index < objects_.size()) {
    objects_[index] = data;
  } else {
    objects_.emplace_back(data);
  }

  return createHandleFromSlot(index);
}

} // namespace cut
