#include <ComputeHandle.h>

namespace cut {

ComputeHandle::ComputeHandle() : container_(nullptr), id_(0) {}

ComputeHandle::ComputeHandle(ComputeContainer<> *container, size_t handleId)
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

// ComputeContainer implementation

template <typename DataType>
ComputeContainer<DataType>::ComputeContainer(uint32_t type) : type_(type) {}

template <typename DataType>
ComputeContainer<DataType>::~ComputeContainer() {
  if (objects_.size() != freeHandles_.size()) {
    logErr("Trying to destroy container before all objects in it have "
           "been deallocated.");
  }
  objects_.clear();
  freeHandles_.clear();
}

template <typename DataType>
const typename ComputeContainer<DataType>::HandleData &
ComputeContainer<DataType>::data(const ComputeHandle &handle) const {
  if (!handle) {
    throw std::runtime_error("Trying to get data for an empty handle");
  }
  verify(handle);

  return objects_[handle.id_].data;
}

template <typename DataType>
ComputeHandle ComputeContainer<DataType>::createHandle(const HandleData &data) {
  size_t index;
  if (!freeHandles_.empty()) {
    index = freeHandles_.back();
    freeHandles_.pop_back();
    objects_[index] = HandleStruct(data);
  } else {
    index = objects_.size();
    objects_.emplace_back(HandleStruct(data));
  }

  return ComputeHandle(reinterpret_cast<ComputeContainer<> *>(this), index);
}

template <typename DataType>
void ComputeContainer<DataType>::verify(const ComputeHandle &handle) const {
  if (handle.container_ != reinterpret_cast<const ComputeContainer<> *>(this)) {
    throw std::runtime_error("Trying to get data for handle which does not "
                             "belong to the container");
  }
}

template <typename DataType>
void ComputeContainer<DataType>::addRef(const ComputeHandle &ref) {
  objects_[ref.id_].refCount++;
}

template <typename DataType>
void ComputeContainer<DataType>::remRef(ComputeHandle &ref) {
  auto &objectRef = objects_[ref.id_];
  objectRef.refCount--;
  if (objectRef.refCount == 0) {
    destroy(objectRef.data);
    freeHandles_.push_back(ref.id_);
    ref.container_ = nullptr;
  }
}

// Explicit template instantiation
template class ComputeContainer<void *>;

} // namespace cut
