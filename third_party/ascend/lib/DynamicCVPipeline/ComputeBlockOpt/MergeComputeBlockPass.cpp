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
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;

// ============================================================================
// Data Structures
// ============================================================================

/// Represents a ComputeBlock: a group of ops sharing the same block_id
struct ComputeBlock {
    int id;                            // block_id value
    CVPipeline::CoreType coreType;     // CUBE_ONLY / VECTOR_ONLY
    SmallVector<Operation *> ops;      // all ops in the group (in IR order)
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Trace upward from value along its definition chain (through view-like ops), collecting ops within C2
static void traceUpwardFromValue(Value value,
                                 const DenseMap<int, ComputeBlock> &computeBlocks,
                                 int c2Id, SmallPtrSet<Operation *, 16> &collected)
{
    auto c2It = computeBlocks.find(c2Id);
    if (c2It == computeBlocks.end()) {
        return;
    }
    DenseSet<Operation *> c2OpsSet(c2It->second.ops.begin(), c2It->second.ops.end());

    Value cur = value;
    while (Operation *defOp = cur.getDefiningOp()) {
        if (!c2OpsSet.contains(defOp)) {
            break;  // not in C2, stop
        }
        if (!collected.insert(defOp).second) {
            break;  // already collected
        }

        if (auto viewLike = dyn_cast<ViewLikeOpInterface>(defOp)) {
            Value source = viewLike.getViewSource();
            // trace other operands of the view (e.g. scalar ops like offset/size for subview)
            for (Value operand : defOp->getOperands()) {
                if (operand != source) {
                    traceUpwardFromValue(operand, computeBlocks, c2Id, collected);
                }
            }
            cur = source;  // pass through source, continue upward
            continue;
        }

        // non-view op, recursively trace each operand
        for (Value subOperand : defOp->getOperands()) {
            traceUpwardFromValue(subOperand, computeBlocks, c2Id, collected);
        }
        break;
    }
}

// ============================================================================
// Sub-function Implementations
// ============================================================================

/// Collect the body Blocks of the innermost scf::ForOp that contain linalg::MatmulOp in ModuleOp (deduplicated).
static void collectInnermostMatmulLoopBlocks(ModuleOp module, SmallVectorImpl<Block *> &blocks)
{
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

/// Step 1+2: Group ops and build block-level SSA dependency graph.
/// computeBlocks: output, block_id → ComputeBlock
/// succs/preds: output, block_id → successor/predecessor block_id list
static void groupAndBuildGraph(Block *block, DenseMap<int, ComputeBlock> &computeBlocks,
                               DenseMap<int, SmallVector<int>> &succs,
                               DenseMap<int, SmallVector<int>> &preds)
{
    // Step 1: Walk all ops in the body block and its nested regions, group by block_id
    block->walk([&](Operation *op) {
        if (op->hasTrait<OpTrait::IsTerminator>()) {
            return;
        }
        auto optId = CVPipeline::getOpBlockId(op);
        if (!optId.has_value()) {
            return;
        }
        int bid = *optId;
        auto it = computeBlocks.find(bid);
        if (it == computeBlocks.end()) {
            computeBlocks[bid] = {bid, CVPipeline::getOpCoreType(op), {}};
        }
        computeBlocks[bid].ops.push_back(op);
    });

    if (computeBlocks.empty()) {
        return;
    }

    // Step 2: Build SSA dependency graph between ComputeBlocks
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

                if (!seenEdges.insert({ancId, curId}).second) {
                    continue;
                }
                succs[ancId].push_back(curId);
                preds[curId].push_back(ancId);
            }
        }
    }
}

/// Step 3: Find exactly 2 candidate VECTOR blocks, and determine pred / succ.
/// pred: source of the VECTOR → VECTOR edge, succ: destination
/// Returns true if valid pred/succ found, false otherwise
static bool findPredAndSuccVec(const DenseMap<int, ComputeBlock> &computeBlocks,
                            const DenseMap<int, SmallVector<int>> &succs,
                            const DenseMap<int, SmallVector<int>> &preds,
                            int &predVId, int &succVId)
{
    auto hasCUBEPred = [&](int bid) {
        auto it = preds.find(bid);
        if (it == preds.end()) return false;
        for (int p : it->second) {
            if (computeBlocks.lookup(p).coreType == CVPipeline::CoreType::CUBE_ONLY) {
                return true;
            }
        }
        return false;
    };
    auto hasCUBESucc = [&](int bid) {
        auto it = succs.find(bid);
        if (it == succs.end()) return false;
        for (int s : it->second) {
            if (computeBlocks.lookup(s).coreType == CVPipeline::CoreType::CUBE_ONLY) {
                return true;
            }
        }
        return false;
    };
    auto hasTensorResult = [&](const ComputeBlock &blk) {
        for (Operation *op : blk.ops) {
            for (Value result : op->getResults()) {
                if (isa<TensorType>(result.getType())) {
                    return true;
                }
            }
        }
        return false;
    };

    // Collect candidates
    DenseSet<int> candidates;
    for (auto &kv : computeBlocks) {
        if (kv.second.coreType != CVPipeline::CoreType::VECTOR_ONLY) {
            continue;
        }
        if (!hasCUBEPred(kv.first) || !hasCUBESucc(kv.first)) {
            continue;
        }
        if (!hasTensorResult(kv.second)) {
            continue;
        }
        candidates.insert(kv.first);
    }

    if (candidates.size() != 2) {
        LOG_DEBUG("Found " << candidates.size() << " candidate(s), need exactly 2");
        return false;
    }

    // Determine pred / succ: along the VECTOR → VECTOR edge, source is pred
    for (int u : candidates) {
        if (auto it = succs.find(u); it != succs.end()) {
            for (int v : it->second) {
                if (candidates.contains(v)) { // successor v is also a candidate VECTOR block
                    predVId = u;
                    succVId = v;
                    LOG_DEBUG("pred=" << predVId << ", succ=" << succVId);
                    return true;
                }
            }
        }
    }

    LOG_DEBUG("Two candidate VECTOR blocks have no direct edge between them");
    return false;
}

/// Step 4: Determine C2 and C3
/// C2: the direct CUBE successor of pred
/// C3: the direct CUBE successor of C2
/// Returns true if C2/C3 successfully found
static bool findC2AndC3(const DenseMap<int, ComputeBlock> &computeBlocks,
                        const DenseMap<int, SmallVector<int>> &succs,
                        int predVId, int &c2Id, int &c3Id)
{
    // Find C2: predVector's CUBE successor
    c2Id = -1;
    if (auto it = succs.find(predVId); it != succs.end()) {
        for (int s : it->second) {
            if (computeBlocks.lookup(s).coreType == CVPipeline::CoreType::CUBE_ONLY) {
                c2Id = s; // get predVector's cube_succs
                break;
            }
        }
    }
    if (c2Id == -1) {
        LOG_DEBUG("pred " << predVId << " has no CUBE successor (C2)");
        return false;
    }

    // Find C3: C2's CUBE successor
    c3Id = -1;
    if (auto it = succs.find(c2Id); it != succs.end()) {
        for (int s : it->second) {
            if (computeBlocks.lookup(s).coreType == CVPipeline::CoreType::CUBE_ONLY) {
                c3Id = s; // get c2's succs: c3
                break;
            }
        }
    }
    if (c3Id == -1) {
        LOG_DEBUG("C2 " << c2Id << " has no CUBE successor (C3)");
        return false;
    }

    LOG_DEBUG("C2=" << c2Id << ", C3=" << c3Id);
    return true;
}

/// Find the first bufferization::ToTensorOp in C2
static Operation *findToTensorInC2(const ComputeBlock &c2Block) {
    for (Operation *op : c2Block.ops) {
        if (isa<bufferization::ToTensorOp>(op)) {
            return op;
        }
    }
    return nullptr;
}

/// Step 5: Clone ops from C2 that C3 depends on into C3
static void cloneOpsToC3(const DenseMap<int, ComputeBlock> &computeBlocks,
                         int c2Id, int c3Id, Block *block,
                         CVPipeline::ComputeBlockIdManager &bm)
{
    LOG_DEBUG("cloneOpsToC3: c2Id=" << c2Id << ", c3Id=" << c3Id);

    auto c2It = computeBlocks.find(c2Id);
    auto c3It = computeBlocks.find(c3Id);
    if (c2It == computeBlocks.end() || c3It == computeBlocks.end()) {
        LOG_DEBUG("C2 or C3 not found in computeBlocks");
        return;
    }
    const auto &c2Block = c2It->second;
    const auto &c3Block = c3It->second;

    if (c3Block.ops.empty()) {
        LOG_DEBUG("C3 block has no ops, skip");
        return;
    }

    // Step 5.1: Find the first bufferization.to_tensor op that C3 depends on (defined in C2)
    Operation *toTensor = findToTensorInC2(c2Block);
    if (!toTensor) {
        LOG_DEBUG("C3 does not depend on any to_tensor from C2");
        return;
    }

    // Step 5.2: Collect ops that need to be cloned (to_tensor + memref.copy and their upstream)
    SmallPtrSet<Operation *, 16> opsToClone;
    opsToClone.insert(toTensor);

    for (Operation *op : c2Block.ops) {
        if (!isa<memref::CopyOp>(op)) {
            continue;
        }
        opsToClone.insert(op);  // copy itself
        // trace upward from each operand of copy
        for (Value operand : op->getOperands()) {
            traceUpwardFromValue(operand, computeBlocks, c2Id, opsToClone);
        }
    }

    if (opsToClone.size() == 1) {
        LOG_DEBUG("Only to_tensor ops to clone, no memref.copy found, skip");
        return;
    }

    Operation *insertBefore = c3Block.ops.front();
    OpBuilder builder(insertBefore);
    IRMapping mapper;
    for (Operation *op : c2Block.ops) {
        if (!opsToClone.contains(op)) continue;
        Operation *cloned = builder.clone(*op, mapper);
        bm.updateBlockId(cloned, c3Id);
    }

    // Remap C3's original ops' operands: replace references to old C2 values
    // with the corresponding cloned values now in C3.
    for (Operation *op : c3Block.ops) {
        if (opsToClone.contains(op)) {
            continue;
        }
        for (auto &operand : op->getOpOperands()) {
            if (Value mapped = mapper.lookupOrNull(operand.get())) {
                operand.set(mapped);
            }
        }
    }

    LOG_DEBUG("Cloned " << opsToClone.size() << " ops from C2(" << c2Id
                        << ") to C3(" << c3Id << ")");
}

/// Execute the merge pipeline on a Block: group → build graph → find pred/succ → determine C2/C3 → clone ops → merge VECTOR
///   c1 → c2 → c3 → c4
///    ↓    ↑    ↓    ↑
///      v1   →    v2
static void tryMergeInBlock(Block *block, CVPipeline::ComputeBlockIdManager &bm)
{
    // Step 1+2: Group and build dependency graph
    DenseMap<int, ComputeBlock> computeBlocks;
    DenseMap<int, SmallVector<int>> succs;
    DenseMap<int, SmallVector<int>> preds;
    groupAndBuildGraph(block, computeBlocks, succs, preds);

    if (computeBlocks.empty()) {
        return;
    }

    // Step 3: Find two candidate VECTOR blocks and determine pred / succ
    int predVId = -1, succVId = -1;
    if (!findPredAndSuccVec(computeBlocks, succs, preds, predVId, succVId)) {
        return;
    }

    // Step 4: Determine C2, C3
    int c2Id = -1, c3Id = -1;
    if (!findC2AndC3(computeBlocks, succs, predVId, c2Id, c3Id)) {
        return;
    }

    // Step 5: Clone ops from C2 that C3 depends on into C3
    cloneOpsToC3(computeBlocks, c2Id, c3Id, block, bm);

    // Step 6: Merge adjacent VECTOR blocks (rewrite succVId's block_id to predVId)
    auto succIt = computeBlocks.find(succVId);
    if (succIt != computeBlocks.end()) {
        for (Operation *op : succIt->second.ops) {
            bm.updateBlockId(op, predVId);
        }
    }
    LOG_DEBUG("Merged VECTOR blocks: pred=" << predVId << ", succ=" << succVId
              << " -> target=" << predVId);
}

// ============================================================================
// Pass Definition
// ============================================================================

namespace {

class MergeComputeBlockPass : public PassWrapper<MergeComputeBlockPass, OperationPass<ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeComputeBlockPass)

    MergeComputeBlockPass() = default;

    StringRef getArgument() const override { return "merge-compute-block"; }

    StringRef getDescription() const override
    {
        return "Merge adjacent vector compute blocks between CUBE blocks";
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();

        SmallVector<Block *> blocksToProcess;
        collectInnermostMatmulLoopBlocks(module, blocksToProcess);

        CVPipeline::ComputeBlockIdManager bm(module);
        for (Block *block : blocksToProcess) {
            tryMergeInBlock(block, bm);
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

std::unique_ptr<OperationPass<ModuleOp>> createMergeComputeBlockPass()
{
    return std::make_unique<MergeComputeBlockPass>();
}

void registerMergeComputeBlockPass()
{
    PassRegistration<MergeComputeBlockPass> reg;
}

} // namespace triton
} // namespace mlir
