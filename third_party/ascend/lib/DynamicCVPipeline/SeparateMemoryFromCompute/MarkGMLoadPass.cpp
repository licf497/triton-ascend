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

#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/MarkGMLoadPass.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromComputePass.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace triton;

static constexpr const char *DEBUG_TYPE = "MarkGMLoad";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

namespace {

static constexpr int kDefaultVBufferCount = 1;
static constexpr int kDefaultCBufferCount = 1;

struct MarkCandidate {
  memref::CopyOp copyOp;
  memref::AllocOp destAlloc; // dest backing alloc after view-like piercing
  scope::ScopeOp scopeOp;    // nearest enclosing scope, should not be null
  int bufferCount;           // filled in Phase 2
};

// Rule 1: Pierce view-like ops and trace scf.for / scf.while iter_args back to
// their init values. Returns true only when the terminal is a BlockArgument
// owned by a func::FuncOp (i.e. a GM pointer function argument).
static bool traceSourceToFuncArg(Value v) {
  while (true) {
    // 1. Pierce view-like ops.
    while (auto viewLike =
               dyn_cast_or_null<ViewLikeOpInterface>(v.getDefiningOp())) {
      v = viewLike.getViewSource();
    }
    // 2. Check terminal: must be a BlockArgument.
    auto blockArg = dyn_cast<BlockArgument>(v);
    if (!blockArg) {
      return false; // ends at a defining op, not a GM arg
    }
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (isa<func::FuncOp>(parentOp)) {
      return true; // func argument => GM load
    }
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      // iter_arg: trace its init value (skip induction var at index 0).
      v = forOp.getInitArgs()[blockArg.getArgNumber() - 1];
      continue;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
      // before-block arg: trace to the corresponding init operand.
      if (blockArg.getOwner() == whileOp.getBeforeBody()) {
        v = whileOp.getInits()[blockArg.getArgNumber()];
        continue;
      }
      // after-block arg: trace to the corresponding condition operand.
      if (blockArg.getOwner() == whileOp.getAfterBody()) {
        auto condOp =
            cast<scf::ConditionOp>(whileOp.getBeforeBody()->getTerminator());
        v = condOp.getArgs()[blockArg.getArgNumber()];
        continue;
      }
      return false;
    }
    return false; // other BlockArgument kinds
  }
}

// Rule 2: Pierce view-like ops on the dest chain and return the backing
// memref::AllocOp, or null if the chain does not terminate at one.
static memref::AllocOp traceDestToAlloc(Value v) {
  while (auto viewLike =
             dyn_cast_or_null<ViewLikeOpInterface>(v.getDefiningOp())) {
    v = viewLike.getViewSource();
  }
  return dyn_cast_or_null<memref::AllocOp>(v.getDefiningOp());
}

// Rule 3: Resolve multi-buffer count N from the enclosing scope.
// Returns -1 when scopeOp is null or has an unexpected tcore_type,
// which should normally never happen.
static int resolveBufferCount(scope::ScopeOp scopeOp) {
  int buffer_num = -1;
  if (!scopeOp) {
    return buffer_num;
  }
  bool isCube = false;
  bool isVector = false;
  if (failed(triton::getScopeType(scopeOp, isCube, isVector))) {
    return buffer_num;
  }
  if (isVector) {
    buffer_num = kDefaultVBufferCount;
  } else if (isCube) {
    buffer_num = kDefaultCBufferCount;
  }
  LOG_DEBUG("return buffer num = " << buffer_num);
  return buffer_num;
}

// Apply Rules 1 & 2 to a memref::CopyOp and build a MarkCandidate if eligible.
// Returns std::nullopt when any rule fails.
static std::optional<MarkCandidate> collectCandidate(memref::CopyOp copyOp) {
  // Rule 1: source must trace back to a func argument (GM load).
  if (!traceSourceToFuncArg(copyOp.getSource())) {
    return std::nullopt;
  }
  // Rule 2: dest must be created by a memref::AllocOp.
  auto allocOp = traceDestToAlloc(copyOp.getTarget());
  if (!allocOp) {
    LOG_DEBUG("dest not created by memref::AllocOp, skip");
    return std::nullopt;
  }
  auto scopeOp = copyOp->getParentOfType<scope::ScopeOp>();
  return MarkCandidate{copyOp, allocOp, scopeOp};
}

// Phase 3: insert annotation::MarkOp with multi_buffer attr on the dest alloc.
// Returns false when bufferCount <= 1 (skip marking), true otherwise.
static bool markGMLoadCandidate(MarkCandidate &c) {
  if (c.bufferCount <= 1) {
    LOG_DEBUG("bufferCount <= 1, skip marking");
    return false;
  }
  OpBuilder builder(c.destAlloc);
  builder.setInsertionPointAfter(c.destAlloc);
  auto markOp = builder.create<annotation::MarkOp>(c.destAlloc->getLoc(),
                                                   c.destAlloc.getResult());
  markOp->setAttr(hivm::MultiBufferAttr::name,
                  builder.getI32IntegerAttr(c.bufferCount));
  LOG_DEBUG("marked multi_buffer = " << c.bufferCount << " on " << c.destAlloc
                                     << "\n");
  return true;
}

} // namespace

void MarkGMLoadPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<annotation::AnnotationDialect, hivm::HIVMDialect,
                  memref::MemRefDialect, scope::ScopeDialect, scf::SCFDialect,
                  func::FuncDialect>();
}

void MarkGMLoadPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LOG_DEBUG("Enter MarkGMLoad pass");

  // Phase 1: collect candidates (read-only).
  SmallVector<MarkCandidate, 8> candidates;
  module.walk([&](memref::CopyOp copyOp) {
    if (auto candidate = collectCandidate(copyOp)) {
      candidates.push_back(*candidate);
    }
  });

  if (candidates.empty()) {
    LOG_DEBUG("no GM load candidate found");
    return;
  }

  // Phase 2 & 3: resolve buffer count N per candidate and insert
  // annotation::MarkOp (mutation).
  for (auto &c : candidates) {
    c.bufferCount = resolveBufferCount(c.scopeOp);
    if (c.bufferCount < 0) {
      LOG_DEBUG("resolveBufferCount failed for " << c.copyOp << ", fallback");
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return;
    }
    markGMLoadCandidate(c);
  }

  LOG_DEBUG("after MarkGMLoad: " << module);
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createMarkGMLoadPass() {
  return std::make_unique<MarkGMLoadPass>();
}

} // namespace triton
} // namespace mlir
