#include <ComputeHandle.h>

namespace cut {

ComputeHandle::ComputeHandle() : container_(nullptr), id_(0) {}

ComputeHandle::ComputeHandle(ComputeContainer *container, size_t handleId)
    : container_(container), id_(handleId) {
  container->addRef(*this);
}

ComputeHandle::~ComputeHandle() {
  reset();
}

ComputeHandle::ComputeHandle(ComputeHandle &&ref) {
  container_ = ref.container_;
  id_ = ref.id_;

  ref.container_ = nullptr;
  ref.id_ = 0;
}

ComputeHandle::ComputeHandle(const ComputeHandle &ref) {
  container_ = ref.container_;
  id_ = ref.id_;

  if (ref) {
    container_->addRef(ref);
  }
}

ComputeHandle::operator bool() const {
  return container_ != nullptr;
}

void ComputeHandle::operator=(const ComputeHandle &ref) {
  if (ref) {
    ref.container_->addRef(ref);
  }

  if (*this) {
    container_->remRef(*this);
  }

  container_ = ref.container_;
  id_ = ref.id_;
}

void ComputeHandle::reset() {
  if (*this) {
    container_->remRef(*this);

    container_ = nullptr;
    id_ = 0;
  }
}

ComputeContainer::~ComputeContainer() {
  if (objects_.size() != freeHandles_.size()) {
    logErr("Trying to destroy container before all objects in it have "
           "been deallocated.");
  }
  objects_.clear();
  freeHandles_.clear();
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

const ComputeContainer::HandleData &
ComputeContainer::data(const ComputeHandle &handle) const {
  if (!handle) {
    throw std::runtime_error("Trying to get data for an empty handle");
  }
  verify(handle);

  return objects_[handle.id_].data;
}

void ComputeContainer::verify(const ComputeHandle &handle) const {
  if (handle.container_ != this) {
    throw std::runtime_error("Trying to get data for handle which does not "
                             "belong to the container");
  }
}

void ComputeContainer::addRef(const ComputeHandle &ref) {
  // Add object reference
  objects_[ref.id_].refCount++;
}

void ComputeContainer::remRef(ComputeHandle &ref) {
  // Reduce object reference
  auto &objectRef = objects_[ref.id_];
  objectRef.refCount--;
  // If object references reached zero then object can be deallocated
  if (objectRef.refCount == 0) {
    // Deallocate the object based on API specific implementation
    destroy(objectRef.data);
    // Add object handle to free handle list
    freeHandles_.push_back(ref.id_);
    ref.container_ = nullptr;
  }
}

} // namespace cut
