#include <ComputeContainer.h>

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

bool ComputeHandle::operator==(const ComputeHandle &other) const {
  return container_ == other.container_ && id_ == other.id_;
}

bool ComputeHandle::operator!=(const ComputeHandle &other) const {
  return !(*this == other);
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

} // namespace cut
