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

#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "unify-store-block";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;

namespace {

/**
 * @brief Check whether a value is "scalar-like".
 *
 * A value is scalar-like if it is:
 *   1. a true scalar (Int/Index/Float);
 *   2. a tensor with empty shape (e.g. tensor<f32>);
 *   3. a splat constant tensor;
 *   4. a single-element tensor (all dims == 1).
 */
static bool isScalarLike(Value value)
{
    Type type = value.getType();
    auto shapedType = dyn_cast<ShapedType>(type);

    // 1. true scalar (int / index / float)
    if (!shapedType) {
        return type.isIntOrIndexOrFloat();
    }

    // 2. tensor with empty shape (e.g. tensor<f32>)
    ArrayRef<int64_t> shape = shapedType.getShape();
    if (shape.empty()) {
        return true;
    }

    // 3. splat constant tensor (all elements identical)
    Attribute attr;
    if (matchPattern(value, m_Constant(&attr))) {
        auto denseAttr = dyn_cast<DenseIntOrFPElementsAttr>(attr);
        return denseAttr && denseAttr.isSplat() && denseAttr.getElementType().isIntOrIndexOrFloat();
    }

    // 4. single-element tensor (all dims == 1)
    return llvm::all_of(shape, [](int64_t dim) { return dim == 1; });
}

static bool isStoreOp(Operation *op)
{
    return isa<bufferization::MaterializeInDestinationOp>(op) || isa<hivm::StoreOp>(op);
}

static Value getStoreSource(Operation *storeOp)
{
    if (auto materialize = dyn_cast<bufferization::MaterializeInDestinationOp>(storeOp)) {
        return materialize.getSource();
    }
    if (auto hivmStore = dyn_cast<hivm::StoreOp>(storeOp)) {
        return hivmStore.getSrc();
    }
    return Value();
}

static Value getStoreDest(Operation *storeOp)
{
    if (auto materialize = dyn_cast<bufferization::MaterializeInDestinationOp>(storeOp)) {
        return materialize.getDest();
    }
    if (auto hivmStore = dyn_cast<hivm::StoreOp>(storeOp)) {
        return hivmStore.getDst();
    }
    return Value();
}

static bool isViewLikeOp(Operation *op)
{
    return isa<ViewLikeOpInterface>(op) || isa<tensor::ExtractSliceOp>(op);
}

static Value getViewSourceValue(Operation *viewOp)
{
    if (auto viewLike = dyn_cast<ViewLikeOpInterface>(viewOp)) {
        return viewLike.getViewSource();
    }
    if (auto extract = dyn_cast<tensor::ExtractSliceOp>(viewOp)) {
        return extract.getSource();
    }
    return Value();
}

static Operation *traceProducerOp(Value startValue, SmallVector<Operation *> &dataViewOps)
{
    SmallPtrSet<Operation *, 16> visited;
    Value cur = startValue;
    while (Operation *defOp = cur.getDefiningOp()) {
        // view-like op (incl. tensor.extract_slice): transparent, pierce through
        if (isViewLikeOp(defOp)) {
            dataViewOps.push_back(defOp);
            cur = getViewSourceValue(defOp);
            continue;
        }
        // scalar-producing op: skip, continue along its first (data) operand
        if (defOp->getNumResults() > 0 && isScalarLike(defOp->getResult(0))) {
            if (defOp->getNumOperands() == 0) {
                return nullptr; // scalar chain ends without a real producer
            }
            cur = defOp->getOperand(0);
            continue;
        }
        // hit: a real compute op; only unifiable if it's VECTOR_ONLY
        if (CVPipeline::getOpCoreType(defOp) != CVPipeline::CoreType::VECTOR_ONLY) {
            LOG_DEBUG("Producer op is not VECTOR_ONLY: " << *defOp);
            return nullptr;
        }
        LOG_DEBUG("Find producer op: " << *defOp);
        return defOp;
    }
    LOG_DEBUG("Could not find producer op!");
    return nullptr; // reached a block argument (iter_arg / gm arg)
}

/**
 * @brief Collect view ops (data chain + dest chain) to unify, filtered by block_id.
 *
 * Only view ops whose current block_id equals storeBlockId are appended to
 * @p ops (per design §5.1 block_id filter rule).
 */
static void collectViewOpsToUnify(SmallVector<Operation *> &ops, Operation *storeOp,
                                  int storeBlockId,
                                  const SmallVector<Operation *> &dataViewOps,
                                  CVPipeline::ComputeBlockIdManager &bm)
{
    SmallPtrSet<Operation *, 16> seen;
    for (Operation *op : ops) {
        seen.insert(op);
    }
    auto addIfMatch = [&](Operation *op) {
        if (bm.getBlockIdByOp(op) == storeBlockId && seen.insert(op).second) {
            ops.push_back(op);
        }
    };

    for (Operation *viewOp : dataViewOps) {
        addIfMatch(viewOp);
    }

    // dest-chain view ops: walk up the dest memref's view chain and unify each
    // view op whose block_id matches storeBlockId.
    Value cur = getStoreDest(storeOp);
    while (Operation *defOp = cur.getDefiningOp()) {
        if (!isViewLikeOp(defOp)) {
            break;
        }
        addIfMatch(defOp);
        cur = getViewSourceValue(defOp);
    }
}

/**
 * @brief Recursively collect scalar-like dependencies, filtered by block_id.
 *
 * For every scalar-like operand of each op in @p ops (and of @p storeOp's
 * control-flow ancestor ops, e.g. scf.for lb/ub/step), append its defining op
 * only if its block_id equals storeBlockId. Non-matching scalar ops are
 * skipped. The defining op itself is collected (no getAncestorInBlock folding),
 * so scalars defined in outer blocks are also unified.
 */
static void collectScalarDeps(SmallVector<Operation *> &ops, int storeBlockId,
                              Operation *storeOp, CVPipeline::ComputeBlockIdManager &bm)
{
    SmallPtrSet<Operation *, 16> seen;
    for (Operation *op : ops) {
        seen.insert(op);
    }

    SmallVector<Operation *> worklist(ops.begin(), ops.end());

    for (Operation *p = storeOp->getParentOp(); p && !isa<ModuleOp>(p);
         p = p->getParentOp()) {
        if (isa<scf::ForOp, scf::IfOp, scf::WhileOp, scf::IndexSwitchOp>(p)) {
            worklist.push_back(p);
        }
    }

    while (!worklist.empty()) {
        Operation *cur = worklist.pop_back_val();
        for (Value operand : cur->getOperands()) {
            // only scalar-like operands are collected; non-scalar operands
            // belong to data/dest chains or external blocks
            if (!isScalarLike(operand)) {
                continue;
            }
            Operation *defOp = operand.getDefiningOp();
            if (!defOp) {
                continue; // block argument, no block_id to change
            }
            // only collect scalar ops whose block_id matches the store's
            if (bm.getBlockIdByOp(defOp) != storeBlockId || !seen.insert(defOp).second) {
                continue;
            }
            ops.push_back(defOp);
            worklist.push_back(defOp); // recurse on its operands
        }
    }
}

/**
 * @brief Assemble the full set of ops whose block_id should be unified.
 *
 * opsToUnify = {store} + viewOps (data + dest, block_id filtered) +
 *              scalarDeps (recursively collected, block_id filtered).
 */
static SmallVector<Operation *> collectOpsToUnify(Operation *storeOp, int storeBlockId,
                                                  const SmallVector<Operation *> &dataViewOps,
                                                  CVPipeline::ComputeBlockIdManager &bm)
{
    SmallVector<Operation *> opsToUnify;
    opsToUnify.push_back(storeOp);

    collectViewOpsToUnify(opsToUnify, storeOp, storeBlockId, dataViewOps, bm);
    collectScalarDeps(opsToUnify, storeBlockId, storeOp, bm);

    for (Operation *op : opsToUnify) {
        LOG_DEBUG("opToUnify: " << *op);
    }
    return opsToUnify;
}

static LogicalResult tryUnifyForStore(Operation *storeOp,
                                      const CVPipeline::MemoryDependenceGraph &memGraph,
                                      CVPipeline::ComputeBlockIdManager &bm)
{
    LOG_DEBUG("Start from storeOp: " << *storeOp << "\n");
    int storeBlockId = bm.getBlockIdByOp(storeOp);
    if (storeBlockId == -1) {
        LOG_DEBUG("storeOp has no block_id, cannot unify! ");
        return failure();
    }

    // Step 1: trace producer op from store.source, skipping views + scalars,
    //         and collect all viewops in the data chain.
    SmallVector<Operation *> dataViewOps;
    Operation *producer = traceProducerOp(getStoreSource(storeOp), dataViewOps);
    if (!producer) {
        return success(); // block argument / fully-scalar chain / non-VECTOR_ONLY -> skip
    }

    int targetBlockId = bm.getBlockIdByOp(producer);
    if (targetBlockId == -1) {
        return success(); // producer has no block_id, cannot unify
    }

    // Step 2: build opsToUnify = {store} + viewOps + scalarDeps.
    SmallVector<Operation *> opsToUnify =
        collectOpsToUnify(storeOp, storeBlockId, dataViewOps, bm);

    // Step 3: cycle detection. On cycle, fail (no fallback per design §6).
    if (CVPipeline::willCreateCycle(opsToUnify, memGraph, targetBlockId, bm)) {
        LOG_DEBUG("Cycle detected, cannot unify storeOp!");
        return failure();
    }

    // Step 4: unify block_id of every op in opsToUnify to the producer's.
    for (Operation *op : opsToUnify) {
        bm.updateBlockId(op, targetBlockId);
    }
    LOG_DEBUG("Successfully unify storeOp: " << *storeOp);
    return success();
}

} // anonymous namespace

class UnifyStoreBlockPass : public PassWrapper<UnifyStoreBlockPass, OperationPass<ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnifyStoreBlockPass)

    UnifyStoreBlockPass() = default;

    StringRef getArgument() const override { return "unify-store-block"; }

    StringRef getDescription() const override
    {
        return "Merge store-semantic operations into the producer vector compute block";
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();
        LOG_DEBUG("Before UnifyStoreBlock: " << *module);

        auto &aa = getAnalysis<AliasAnalysis>();
        CVPipeline::MemoryDependenceGraph memGraph(module, aa);
        auto bm = CVPipeline::ComputeBlockIdManager(module);

        // Step 1: collect all store-semantic ops (MaterializeInDestinationOp,
        //         hivm::StoreOp) in program order.
        SmallVector<Operation *> storeOps;
        module.walk([&](Operation *op) {
            if (isStoreOp(op)) {
                storeOps.push_back(op);
            }
        });

        // Step 2: for each storeOp, try to unify its block_id with the producer's.
        for (Operation *storeOp : storeOps) {
            if (failed(tryUnifyForStore(storeOp, memGraph, bm))) {
                signalPassFailure();
                return;
            }
        }

        LOG_DEBUG("After UnifyStoreBlock: " << *module);
    }
};

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createUnifyStoreBlockPass()
{
    return std::make_unique<UnifyStoreBlockPass>();
}

void registerUnifyStoreBlockPass()
{
    PassRegistration<UnifyStoreBlockPass> reg;
}

} // namespace triton
} // namespace mlir
