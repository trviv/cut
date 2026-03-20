#include "MemoryPlanner.h"
#include "TensorStore.h"

#include <algorithm>
#include <unordered_set>
#include <vector>

namespace cut {
namespace graph {

MemoryPlanner::MemoryPlanner(TensorStore &store) : store_(&store) {}

size_t MemoryPlanner::plan(Graph &graph) {
  // Step 1: Get topological order and build position map
  auto order = graph.topologicalOrder();
  if (order.empty())
    return 0;

  std::vector<uint32_t> posMap(graph.size(), UINT32_MAX);
  for (uint32_t pos = 0; pos < order.size(); ++pos) {
    posMap[order[pos]] = pos;
  }

  // Step 2: Compute last-use position for each node
  std::vector<uint32_t> lastUse(graph.size(), 0);
  for (uint32_t pos = 0; pos < order.size(); ++pos) {
    uint32_t idx = order[pos];
    auto &gn = graph.nodes()[idx];
    if (!gn.op || gn.isRemoved)
      continue;
    for (uint32_t inputId : gn.inputIds) {
      if (inputId < graph.size()) {
        lastUse[inputId] = std::max(lastUse[inputId], pos);
      }
    }
  }

  // Identify nodes consumed by slice operations. These nodes' buffers must
  // not be arena-aliased because slice creates zero-copy views that reference
  // the parent buffer after graph execution completes.
  std::unordered_set<uint32_t> sliceParents;
  for (uint32_t idx : order) {
    auto &gn = graph.nodes()[idx];
    if (!gn.op || gn.isRemoved)
      continue;
    if (gn.logicalType == LogicalOpType::Slice) {
      for (uint32_t inputId : gn.inputIds) {
        sliceParents.insert(inputId);
      }
    }
  }

  // Step 3: Identify transient nodes and build liveness intervals
  struct LiveInterval {
    uint32_t nodeId;
    uint32_t birth;
    uint32_t death;
    size_t sizeBytes;
  };
  std::vector<LiveInterval> intervals;
  size_t originalTotal = 0;

  for (uint32_t idx : order) {
    auto &gn = graph.nodes()[idx];
    if (!gn.op || gn.isRemoved)
      continue;
    if (gn.isInput || gn.isOutput)
      continue;
    // Skip nodes whose buffers are referenced by slice views
    if (sliceParents.count(idx))
      continue;

    size_t sizeBytes = store_->getTensor(gn.op->output()).size();
    if (sizeBytes == 0)
      continue;

    intervals.push_back(
        {idx, posMap[idx], std::max(posMap[idx], lastUse[idx]), sizeBytes});
    originalTotal += sizeBytes;
  }

  if (intervals.empty())
    return 0;

  // Step 4: Sort by size descending (first-fit decreasing)
  std::sort(intervals.begin(), intervals.end(),
            [](const LiveInterval &a, const LiveInterval &b) {
              return a.sizeBytes > b.sizeBytes;
            });

  // Step 5: Assign offsets
  const size_t alignment = store_->offsetAlignment();
  const size_t alignMask = ~(alignment - 1);

  struct Allocation {
    uint32_t nodeId;
    size_t offset;
    size_t size;
    uint32_t birth;
    uint32_t death;
  };
  std::vector<Allocation> allocations;
  allocations.reserve(intervals.size());
  size_t arenaSize = 0;

  for (auto &interval : intervals) {
    // Collect occupied regions that overlap in time with this interval
    std::vector<std::pair<size_t, size_t>> occupied;
    for (auto &alloc : allocations) {
      bool overlaps =
          (alloc.birth <= interval.death) && (interval.birth <= alloc.death);
      if (overlaps) {
        occupied.push_back({alloc.offset, alloc.offset + alloc.size});
      }
    }
    std::sort(occupied.begin(), occupied.end());

    // Find lowest aligned offset that fits in a gap
    size_t candidateOffset = 0;
    for (auto &[start, end] : occupied) {
      if (candidateOffset + interval.sizeBytes <= start) {
        break; // fits in this gap
      }
      size_t alignedEnd = (end + alignment - 1) & alignMask;
      if (alignedEnd > candidateOffset) {
        candidateOffset = alignedEnd;
      }
    }

    allocations.push_back({interval.nodeId, candidateOffset, interval.sizeBytes,
                           interval.birth, interval.death});
    arenaSize = std::max(arenaSize, candidateOffset + interval.sizeBytes);
  }

  // Step 6: Skip if no savings
  if (arenaSize == 0 || arenaSize >= originalTotal)
    return 0;

  // Step 7: Create arena buffer
  // Use Float32 (4 bytes/element). Round up element count to cover arenaSize.
  // calculateAlignedSize rounds innermost dim to multiple of 4, so the actual
  // buffer may be slightly larger than arenaSize — that's fine.
  uint32_t arenaElements =
      static_cast<uint32_t>((arenaSize + sizeof(float) - 1) / sizeof(float));
  Tensor arena = store_->createTensorEmpty({arenaElements}, DataType::Float32);

  // Step 8: Create views and rebind outputs
  for (auto &alloc : allocations) {
    auto &gn = graph.nodes()[alloc.nodeId];
    const auto &shape = gn.outputShape;
    auto dtype = gn.outputDtype;
    Tensor view = store_->createTensorView(arena, alloc.offset, shape, dtype);
    gn.op->rebindOutput(view);
  }

  return originalTotal - arenaSize;
}

} // namespace graph
} // namespace cut
