/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

#include "DynamicCVPipeline/Common/Utils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "merge-compute-block";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;

// ============================================================================
// Data Structures
// ============================================================================

/// Represents a ComputeBlock: a group of ops sharing the same block_id
struct ComputeBlock {
  int id;                        // block_id value
  CVPipeline::CoreType coreType; // CUBE_ONLY / VECTOR_ONLY
  SmallVector<Operation *> ops;  // all ops in the group (in IR order)
};

// ============================================================================
// Sub-function Implementations
// ============================================================================

/// Collect the body Blocks of the innermost scf::ForOp that contain
/// linalg::MatmulOp in ModuleOp (deduplicated).
static void collectInnermostMatmulLoopBlocks(ModuleOp module,
                                             SmallVectorImpl<Block *> &blocks) {
  DenseSet<Block *> seen;
  module->walk([&](linalg::MatmulOp matmul) {
    Operation *parent = matmul->getParentOp();
    while (parent) {
      if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
        Block *body = forOp.getBody();
        if (seen.insert(body).second) {
          blocks.push_back(body);
        }
        break;
      }
      parent = parent->getParentOp();
    }
  });
}

/// Check whether a ComputeBlock has any tensor-typed result.
static bool hasTensorResult(const ComputeBlock &blk) {
  for (Operation *op : blk.ops) {
    for (Value result : op->getResults()) {
      if (isa<TensorType>(result.getType())) {
        return true;
      }
    }
  }
  return false;
}

/// Step 1: Group ops and build block-level SSA dependency graph (enhanced).
/// computeBlocks: output, block_id → ComputeBlock
/// succs/preds: output, block_id → successor/predecessor block_id list
/// blockEdges: output, srcId → dstId → operands (cross-block SSA edges)
static void groupAndBuildGraph(
    Block *block, DenseMap<int, ComputeBlock> &computeBlocks,
    DenseMap<int, SmallVector<int>> &succs,
    DenseMap<int, SmallVector<int>> &preds,
    DenseMap<int, DenseMap<int, SmallVector<Value>>> &blockEdges) {
  // Walk all ops in the body block and its nested regions, group by
  // block_id
  block->walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::IsTerminator>()) {
      return;
    }
    auto optId = CVPipeline::getOpBlockId(op);
    if (!optId.has_value()) {
      return;
    }
    int bid = *optId;
    if (!computeBlocks.contains(bid)) {
        // find new block_id, create new ComputeBlock
      computeBlocks[bid] = {bid, CVPipeline::getOpCoreType(op), {}};
    }
    computeBlocks[bid].ops.push_back(op);
  });

  if (computeBlocks.empty()) {
    return;
  }

  // Build SSA dependency graph between ComputeBlocks, recording
  // inter-block operand edges
  DenseSet<std::pair<int, int>> seenEdges;
  for (auto &kv : computeBlocks) {
    int curId = kv.first;
    for (Operation *op : kv.second.ops) {
      for (Value operand : op->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if (!defOp) {
          continue; // block argument
        }
        Operation *ancestor = CVPipeline::getAncestorInBlock(defOp, block);
        if (!ancestor) {
          continue; // not in this block
        }
        auto ancIdOpt = CVPipeline::getOpBlockId(ancestor);
        if (!ancIdOpt.has_value()) {
          continue;
        }
        int ancId = *ancIdOpt;
        if (ancId == curId) {
          continue; // same ComputeBlock internal edge
        }

        // Record the operand that triggers this cross-block edge
        blockEdges[ancId][curId].push_back(operand);

        if (!seenEdges.insert({ancId, curId}).second) {
          continue;
        }
        succs[ancId].push_back(curId);
        preds[curId].push_back(ancId);
      }
    }
  }
}

/// Find the first CUBE predecessor of a given block.
/// Returns -1 if not found.
static int findCubePred(int blockId,
                        const DenseMap<int, ComputeBlock> &computeBlocks,
                        const DenseMap<int, SmallVector<int>> &preds) {
  auto it = preds.find(blockId);
  if (it == preds.end()) {
    return -1;
  }
  for (int p : it->second) {
    if (computeBlocks.lookup(p).coreType == CVPipeline::CoreType::CUBE_ONLY) {
      return p;
    }
  }
  return -1;
}


/// Helper to add scalar/index-typed SSA defining ops of \p op to the worklist
/// (recursively processes new additions in the same call).
static void collectScalarDeps(Operation *op,
                               SmallPtrSet<Operation *, 16> &collected,
                               SmallVectorImpl<Operation *> &worklist) {
  for (Value operand : op->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp) {
      continue;
    }
    if (collected.insert(defOp).second) {
      worklist.push_back(defOp);
      collectScalarDeps(defOp, collected, worklist);
    }
  }
}

/// Recursively collect all execution predecessors of an op using
/// MemoryDependenceGraph::getExecBefore, and also collect scalar/index-typed
/// SSA dependencies for each collected op. Collects the transitive closure of
/// both memory execution predecessors and scalar data dependencies.
static void collectExecPreds(Operation *op,
                             const CVPipeline::MemoryDependenceGraph &memGraph,
                             SmallPtrSet<Operation *, 16> &collected) {
  SmallVector<Operation *, 16> worklist;

  for (Operation *pred : memGraph.getExecBefore(op)) {
    if (collected.insert(pred).second) {
      worklist.push_back(pred);
      collectScalarDeps(pred, collected, worklist);
    }
  }
  while (!worklist.empty()) {
    Operation *cur = worklist.pop_back_val();
    for (Operation *pred : memGraph.getExecBefore(cur)) {
      if (collected.insert(pred).second) {
        worklist.push_back(pred);
        collectScalarDeps(pred, collected, worklist);
      }
    }
    // Trace operand dependencies for ops nested inside cur's regions
    // (e.g., linalg.fill inside scf.if), so that their operand-defining
    // ops are also collected and cloned.
    for (auto &region : cur->getRegions()) {
      for (auto &blk : region) {
        for (auto &innerOp : blk) {
          collectScalarDeps(&innerOp, collected, worklist);
        }
      }
    }
  }
}

/// Check if Cube depends on CubePre via bufferization::ToTensorOp defined in
/// CubePre. If so, returns the to_tensor ops in CubePre that Cube depends on.
static SmallVector<Operation *>
findToTensorDeps(int cubePreId, int cubeId,
                 const DenseMap<int, ComputeBlock> &computeBlocks,
                 const DenseMap<int, DenseMap<int, SmallVector<Value>>>
                     &blockEdges) {
  SmallVector<Operation *> result;
  auto cubePreIt = computeBlocks.find(cubePreId);
  if (cubePreIt == computeBlocks.end()) {
    return result;
  }
  DenseSet<Operation *> cubePreOpsSet(cubePreIt->second.ops.begin(),
                                       cubePreIt->second.ops.end());

  auto edgeIt = blockEdges.find(cubePreId);
  if (edgeIt == blockEdges.end()) {
    return result;
  }
  auto dstIt = edgeIt->second.find(cubeId);
  if (dstIt == edgeIt->second.end()) {
    return result;
  }

  for (Value operand : dstIt->second) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp) {
      continue;
    }
    if (isa<bufferization::ToTensorOp>(defOp) && cubePreOpsSet.contains(defOp)) {
      result.push_back(defOp);
    }
  }
  return result;
}

/// Clone ops from CubePre into Cube at the front of Cube's ops.
/// Selected ops must be in CubePre and are in `toClone`.
static void cloneOpCrossCubeDep(int cubePreId, int cubeId,
                             const SmallPtrSet<Operation *, 16> &toClone,
                             const DenseMap<int, ComputeBlock> &computeBlocks,
                             CVPipeline::ComputeBlockIdManager &bm) {
  const auto &cubePreBlock = computeBlocks.at(cubePreId);
  const auto &cubeBlock = computeBlocks.at(cubeId);

  if (cubeBlock.ops.empty()) {
    return;
  }

  Operation *insertBefore = cubeBlock.ops.front();
  OpBuilder builder(insertBefore);
  IRMapping mapper;
  for (Operation *op : cubePreBlock.ops) {
    if (!toClone.contains(op)) {
      continue;
    }
    Operation *cloned = builder.clone(*op, mapper);
    bm.updateBlockId(cloned, cubeId);
    // Update block_id for all ops nested inside regions (e.g. scf.if body)
    for (auto &region : cloned->getRegions()) {
      for (auto &block : region) {
        block.walk([&](Operation *innerOp) {
          bm.updateBlockId(innerOp, cubeId);
        });
      }
    }
  }

  // Remap Cube's original ops' operands: replace references to old CubePre
  // values with the corresponding cloned values now in Cube.
  for (Operation *op : cubeBlock.ops) {
    if (toClone.contains(op)) {
      continue;
    }
    for (auto &operand : op->getOpOperands()) {
      if (Value mapped = mapper.lookupOrNull(operand.get())) {
        operand.set(mapped);
      }
    }
  }
}

/// Step 2: Collect VECTOR_ONLY candidates that have tensor results.
static void collectVectorCandidates(
    const DenseMap<int, ComputeBlock> &computeBlocks,
    DenseSet<int> &vecCandidateSet) {
  for (auto &kv : computeBlocks) {
    if (kv.second.coreType != CVPipeline::CoreType::VECTOR_ONLY)
      continue;
    if (!hasTensorResult(kv.second))
      continue;
    vecCandidateSet.insert(kv.first);
  }
}

/// Step 3: Find a pair of adjacent VECTOR blocks (predV → succV).
/// Returns true if a pair is found.
static bool findAdjacentVectorPair(
    const DenseSet<int> &vecCandidateSet,
    const DenseMap<int, SmallVector<int>> &succs, int &predVId, int &succVId) {
  for (int candId : vecCandidateSet) {
    auto it = succs.find(candId);
    if (it == succs.end())
      continue;
    for (int succId : it->second) {
      if (vecCandidateSet.contains(succId)) {
        predVId = candId;
        succVId = succId;
        return true;
      }
    }
  }
  return false;
}

/// Step 4: Try to directly merge succV into predV.
/// Returns true if merge succeeded.
static bool tryDirectMerge(ArrayRef<Operation *> opsToMerge,
                           const CVPipeline::MemoryDependenceGraph &memGraph,
                           int predVId, int succVId,
                           CVPipeline::ComputeBlockIdManager &bm) {
  if (willCreateCycle(opsToMerge, memGraph, predVId, bm))
    return false;
  LOG_DEBUG("Successfully direct merge: " << succVId << " -> " << predVId);

  for (Operation *op : opsToMerge) 
    bm.updateBlockId(op, predVId);
  return true;
}

/// Step 5e: Collect ops in CubePre that need to be cloned to Cube to break
/// the cycle. Returns the set of ops to clone (empty if none).
static SmallPtrSet<Operation *, 16> collectOpsToBreakCycle(
    int cubePreId, const DenseMap<int, ComputeBlock> &computeBlocks,
    const SmallPtrSet<Operation *, 16> &allPredecessors) {
  auto cbIt = computeBlocks.find(cubePreId);

  SmallPtrSet<Operation *, 16> opsToClone;
  for (Operation *op : allPredecessors) {
    if (llvm::is_contained(cbIt->second.ops, op))
      opsToClone.insert(op);
  }
  return opsToClone;
}

/// Step 5: Try cross-Cube clone to break the cycle, then merge.
/// Returns true if merge succeeded after clone.
static bool tryCrossCubeCloneMerge(
    Block *block, const DenseMap<int, ComputeBlock> &computeBlocks,
    const DenseMap<int, SmallVector<int>> &preds,
    const DenseMap<int, DenseMap<int, SmallVector<Value>>> &blockEdges,
    int predVId, int succVId, ArrayRef<Operation *> opsToMerge,
    const CVPipeline::MemoryDependenceGraph &memGraph,
    CVPipeline::ComputeBlockIdManager &bm) {
  // 5a. Find succV's CUBE predecessor (Cube)
  int cubeId = findCubePred(succVId, computeBlocks, preds);
  if (cubeId == -1) {
    LOG_DEBUG("MergeComputeBlock: succV " << succVId
              << " has no CUBE predecessor, skipping cross-Cube clone");
    return false;
  }

  // 5b. Find Cube's CUBE predecessor (CubePre)
  int cubePreId = findCubePred(cubeId, computeBlocks, preds);
  if (cubePreId == -1) {
    LOG_DEBUG("MergeComputeBlock: Cube " << cubeId
              << " has no CUBE predecessor, skipping cross-Cube clone");
    return false;
  }

  // 5c. Check if Cube depends on CubePre via to_tensor
  SmallVector<Operation *> toTensorOps =
      findToTensorDeps(cubePreId, cubeId, computeBlocks, blockEdges);
  if (toTensorOps.empty()) {
    LOG_DEBUG("MergeComputeBlock: Cube(" << cubeId
              << ") depends on CubePre(" << cubePreId
              << ") but not via to_tensor, skipping");
    return false;
  }

  // 5d. Trace back all transitive execution predecessors
  SmallPtrSet<Operation *, 16> allPredecessors;
  for (Operation *toTensor : toTensorOps) {
    allPredecessors.insert(toTensor);
    collectExecPreds(toTensor, memGraph, allPredecessors);
  }

  // 5e. Filter to ops in CubePre and clone to Cube
  SmallPtrSet<Operation *, 16> opsToClone = 
      collectOpsToBreakCycle(cubePreId, computeBlocks, allPredecessors);

  for (auto op : opsToClone)
    LOG_DEBUG("Cloning op: " << *op);
  cloneOpCrossCubeDep(cubePreId, cubeId, opsToClone, computeBlocks, bm);

  // 5f. Re-check cycle after cloning
  if (willCreateCycle(opsToMerge, memGraph, predVId, bm)) {
    LOG_DEBUG("MergeComputeBlock: still creates cycle after cross-Cube clone for VECTOR "
             << predVId << " -> " << succVId);
    return false;
  }

  LOG_DEBUG("MergeComputeBlock: Successfully merge after cross-Cube clone: "
            << succVId << " -> " << predVId);
  for (Operation *op : opsToMerge) 
    bm.updateBlockId(op, predVId);
  return true;
}

/// Core merge logic for one scf::ForOp body Block.
static void tryMergeInBlock(Block *block, CVPipeline::ComputeBlockIdManager &bm,
                            const CVPipeline::MemoryDependenceGraph &memGraph) {
  while (true) {
    // Step 1: Group and build enhanced dependency graph
    /// computeBlocks: block_id → ComputeBlock
    DenseMap<int, ComputeBlock> computeBlocks;
    /// succs/preds: block_id → successor/predecessor block_id list
    DenseMap<int, SmallVector<int>> succs;
    DenseMap<int, SmallVector<int>> preds;
    /// blockEdges: srcId → dstId → operands (cross-block SSA edges)
    DenseMap<int, DenseMap<int, SmallVector<Value>>> blockEdges;
    groupAndBuildGraph(block, computeBlocks, succs, preds, blockEdges);
    if (computeBlocks.empty())
      return;

    // Step 2: Collect VECTOR candidates
    DenseSet<int> vecCandidateSet;
    collectVectorCandidates(computeBlocks, vecCandidateSet);
    if (vecCandidateSet.size() < 2) {
      LOG_DEBUG("MergeComputeBlock: vecCandidates.size() < 2, skipping");
      return;
    }

    // Step 3: Find a pair of adjacent VECTOR blocks
    int predVId = -1, succVId = -1;
    if (!findAdjacentVectorPair(vecCandidateSet, succs, predVId, succVId)) {
      LOG_DEBUG("MergeComputeBlock: No adjacent VECTOR pair found, skipping");
      return;
    }

    LOG_DEBUG("Found VECTOR pair: predV=" << predVId << ", succV=" << succVId);

    auto succIt = computeBlocks.find(succVId);
    if (succIt == computeBlocks.end())
      return;
    SmallVector<Operation *> opsToMerge = succIt->second.ops;

    // Step 4: Try direct merge
    if (tryDirectMerge(opsToMerge, memGraph, predVId, succVId, bm))
      continue;

    // Step 5: Try cross-Cube clone merge
    if (!tryCrossCubeCloneMerge(block, computeBlocks, preds, blockEdges,
                                predVId, succVId, opsToMerge, memGraph, bm))
      return;
    // continue to try next pair
  }
}

// ============================================================================
// Pass Definition
// ============================================================================

namespace {

class MergeComputeBlockPass
    : public PassWrapper<MergeComputeBlockPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeComputeBlockPass)

  MergeComputeBlockPass() = default;

  StringRef getArgument() const override { return "merge-compute-block"; }

  StringRef getDescription() const override {
    return "Merge adjacent vector compute blocks between/around CUBE blocks";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    LOG_DEBUG("Before: " << *module);

    SmallVector<Block *> blocksToProcess;
    collectInnermostMatmulLoopBlocks(module, blocksToProcess);

    auto &aa = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memGraph(module, aa);
    CVPipeline::ComputeBlockIdManager bm(module);
    for (Block *block : blocksToProcess) {
      tryMergeInBlock(block, bm, memGraph);
    }

    LOG_DEBUG("After: " << *module);
  }
};

} // namespace

// ============================================================================
// Pass Registration
// ============================================================================

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createMergeComputeBlockPass() {
  return std::make_unique<MergeComputeBlockPass>();
}

void registerMergeComputeBlockPass() {
  PassRegistration<MergeComputeBlockPass> reg;
}

} // namespace triton
} // namespace mlir