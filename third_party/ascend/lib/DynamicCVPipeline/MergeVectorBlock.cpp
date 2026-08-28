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

#include <memory>

#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/MergeVectorBlock.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

using namespace mlir;
using namespace triton;
using namespace CVPipeline;

static constexpr const char *DEBUG_TYPE = "MergeVectorBlock";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

/// Check if a scope is a vector scope.
static bool isVectorScope(scope::ScopeOp scopeOp) {
  auto coreTypeAttr =
      scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
  if (!coreTypeAttr) {
    return false;
  }
  return coreTypeAttr.getTcoretype() == hivm::TCoreType::VECTOR;
}

/// Collect all block_ids that appear in the given block and its nested regions.
static void collectBlockIds(Block *block, llvm::SmallDenseSet<int> &blockIds) {
  block->walk([&](Operation *op) {
    if (auto attr = op->getAttrOfType<IntegerAttr>(kBlockId)) {
      int bid = attr.getInt();
      if (bid > 0) {
        blockIds.insert(bid);
      }
    }
  });
}

static constexpr llvm::StringLiteral kEnableMergeVectorBlockKernels[] = {
    "flex_attention_backward_dkdv_kernel",
    "flex_attention_backward_dkdv_kernel_tasklist",
    "_swa_bwd_dkdv_kernel",
};

void MergeVectorBlockPass::runOnOperation() {
  auto module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  bool shouldRun = false;
  for (auto funcOp : module.getOps<func::FuncOp>()) {
    if (llvm::is_contained(kEnableMergeVectorBlockKernels,
                           funcOp.getSymName())) {
      LOG_DEBUG("Enable MergeVectorBlock for kernel: " << funcOp.getSymName());
      shouldRun = true;
      break;
    }
  }
  if (!shouldRun) {
    return;
  }

  LOG_DEBUG("MergeVectorBlock pass running\n");
  LOG_DEBUG("module: " << module << "\n");

  // Build the block-id manager to read and update block_ids.
  ComputeBlockIdManager bm(module);

  module.walk([&](scope::ScopeOp scopeOp) {
    if (!isVectorScope(scopeOp)) {
      return WalkResult::advance();
    }

    // Collect all main_loop ops inside the vector scope. There may be multiple.
    llvm::SmallVector<Operation *> mainLoops;
    scopeOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isMainLoopOp(op)) {
        mainLoops.push_back(op);
      }
    });

    for (Operation *mainLoop : mainLoops) {
      Block *body = getLoopBodyBlock(mainLoop);
      if (!body) {
        continue;
      }

      LOG_DEBUG("Found main_loop in vector scope, body block: " << body
                                                                << "\n");

      // Collect all block_ids present in the main_loop body.
      llvm::SmallDenseSet<int> blockIdsInBody;
      collectBlockIds(body, blockIdsInBody);

      // Collect candidate block_ids: those that contain ops with the
      // ssbuffer.crossCoreDeps attribute inside the main_loop body.
      llvm::SmallDenseSet<int> candidateBlockIds;
      for (int bid : blockIdsInBody) {
        for (Operation *op : bm.getOpsRefByBlockId(bid)) {
          if (op->getBlock() != body && !body->findAncestorOpInBlock(*op)) {
            continue;
          }
          if (op->hasAttr(kCrossCoreDeps)) {
            candidateBlockIds.insert(bid);
            LOG_DEBUG("Candidate block_id "
                      << bid << " has crossCoreDeps op: " << *op << "\n");
            break;
          }
        }
      }

      if (candidateBlockIds.size() <= 1) {
        LOG_DEBUG("Found " << candidateBlockIds.size()
                           << " candidate blocks in main_loop " << mainLoop
                           << ", nothing to merge.\n");
        continue;
      }

      // Merge all candidate blocks into a single new block_id.
      int newBlockId = bm.getNextId();
      LOG_DEBUG("Merging " << candidateBlockIds.size()
                           << " candidate blocks in main_loop " << mainLoop
                           << " into new block_id " << newBlockId << "\n");

      // Collect ops to update first to avoid iterator invalidation when
      // updateBlockId modifies the internal vector being iterated.
      llvm::SmallVector<Operation *> opsToUpdate;
      for (int bid : candidateBlockIds) {
        for (Operation *op : bm.getOpsRefByBlockId(bid)) {
          if (op->getBlock() != body && !body->findAncestorOpInBlock(*op)) {
            continue;
          }
          opsToUpdate.push_back(op);
        }
      }
      for (Operation *op : opsToUpdate) {
        bm.updateBlockId(op, newBlockId);
      }
    }

    return WalkResult::advance();
  });
  LOG_DEBUG("After merge, module: " << module << "\n");
}

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>> createMergeVectorBlockPass() {
  return std::make_unique<MergeVectorBlockPass>();
}

void registerMergeVectorBlockPasses() {
  registerPass(createMergeVectorBlockPass);
}

} // namespace mlir::triton
