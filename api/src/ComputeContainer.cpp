#include <ComputeContainer.h>

namespace cut {

ComputeContainer::ComputeContainer(uint32_t type) : type_(type) {}

ComputeContainer::~ComputeContainer() {
  if (objects_.size() != freeHandles_.size()) {
    logErr("Trying to destroy container before all objects in it have "
           "been deallocated.");
  }
  objects_.clear();
  freeHandles_.clear();
}

const ComputeContainer::HandleData &
ComputeContainer::data(const ComputeHandle &handle) const {
  if (!handle) {
    throw std::runtime_error("Trying to get data for an empty handle");
  }
  verify(handle);

  return objects_[handle.id_].data;
}

ComputeHandle ComputeContainer::createHandle(const HandleData &data) {
  size_t index;
  if (!freeHandles_.empty()) {
    index = freeHandles_.back();
    freeHandles_.pop_back();
    objects_[index] = HandleStruct(data);
  } else {
    index = objects_.size();
    objects_.emplace_back(HandleStruct(data));
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
  objects_[ref.id_].refCount++;
}

void ComputeContainer::remRef(ComputeHandle &ref) {
  auto &objectRef = objects_[ref.id_];
  objectRef.refCount--;
  if (objectRef.refCount == 0) {
    destroy(objectRef.data);
    freeHandles_.push_back(ref.id_);
    ref.container_ = nullptr;
  }
}

} // namespace cut
