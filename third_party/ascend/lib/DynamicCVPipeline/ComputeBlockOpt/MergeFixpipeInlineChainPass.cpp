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

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#define DEBUG_TYPE "merge-fixpipe-inline-chain"
#define LOG_DEBUG(msg)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << msg << "\n")

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {

namespace {

/// Represents a matched pattern: producer matmul -> trunc -> consumer matmul
struct FixpipeInlinePattern {
  linalg::MatmulOp producer;
  Operation *opToMove;
};

} // namespace

class MergeFixpipeInlineChainPass
    : public PassWrapper<MergeFixpipeInlineChainPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeFixpipeInlineChainPass)

  MergeFixpipeInlineChainPass() = default;
  void runOnOperation() override;

  llvm::StringRef getArgument() const final {
    return "merge-fixpipe-inline-chain";
  }
  void getDependentDialects(DialectRegistry &registry) const override;

  llvm::StringRef getDescription() const final {
    return "Merge fixpipe inline chain: move trunc between two matmuls into "
           "the producer matmul's CUBE block";
  }

private:
  /// Collect all patterns matching: producer-matmul -> trunc -> consumer-matmul
  SmallVector<FixpipeInlinePattern> collectPatterns(ModuleOp module);

  /// Apply transforms: set truncOp's core_type=CUBE, block_id=producer's
  void applyTransforms(const SmallVector<FixpipeInlinePattern> &patterns,
                       CVPipeline::ComputeBlockIdManager &bm);
};

SmallVector<FixpipeInlinePattern>
MergeFixpipeInlineChainPass::collectPatterns(ModuleOp module) {
  SmallVector<FixpipeInlinePattern> patterns;

  module.walk([&](linalg::MatmulOp producerMatmul) {
    Value producerResult = producerMatmul.getResult(0);
    if (!producerResult.hasOneUse()) {
      return;
    }

    auto *opToMove = *producerResult.getUsers().begin();
    if (!CVPipeline::isValidTrunc(opToMove)) {
      return;
    }

    Value truncResult = opToMove->getResult(0);
    if (!truncResult.hasOneUse()) {
      return;
    }

    auto consumerMatmul =
        dyn_cast<linalg::MatmulOp>(*truncResult.getUsers().begin());
    if (!consumerMatmul) {
      return;
    }

    LOG_DEBUG("Matched pattern: producer=" << producerMatmul
                                           << " trunc=" << *opToMove
                                           << " consumer=" << consumerMatmul);
    patterns.push_back({producerMatmul, opToMove});
  });

  LOG_DEBUG("== Found " << patterns.size() << " fixpipe inline patterns ==\n");
  return patterns;
}

void MergeFixpipeInlineChainPass::applyTransforms(
    const SmallVector<FixpipeInlinePattern> &patterns,
    CVPipeline::ComputeBlockIdManager &bm) {
  for (const auto &pattern : patterns) {
    int producerBlockId = bm.getBlockIdByOp(pattern.producer);
    if (producerBlockId == -1) {
      LOG_DEBUG("Producer matmul has no block_id, skip: " << pattern.producer);
      continue;
    }

    // Set opToMove's core_type to CUBE
    pattern.opToMove->setAttr(
        CVPipeline::kCoreType,
        StringAttr::get(pattern.opToMove->getContext(), "CUBE"));

    // Set opToMove's block_id to match the producer matmul
    bm.updateBlockId(pattern.opToMove, producerBlockId);

    LOG_DEBUG("Merged opToMove into block_id="
              << producerBlockId << " core_type=CUBE: " << *pattern.opToMove);
  }
}

void MergeFixpipeInlineChainPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<hivm::HIVMDialect>();
}

void MergeFixpipeInlineChainPass::runOnOperation() {
  LOG_DEBUG("== MergeFixpipeInlineChain Pass Start ==\n");
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LOG_DEBUG(module);

  auto patterns = collectPatterns(module);
  if (patterns.empty()) {
    LOG_DEBUG("== MergeFixpipeInlineChain Pass Complete (no patterns) ==\n");
    return;
  }

  auto bm = CVPipeline::ComputeBlockIdManager(module);
  applyTransforms(patterns, bm);

  LOG_DEBUG("== MergeFixpipeInlineChain Pass Complete ==\n");
}

std::unique_ptr<OperationPass<ModuleOp>> createMergeFixpipeInlineChainPass() {
  return std::make_unique<MergeFixpipeInlineChainPass>();
}

} // namespace triton
} // namespace mlir
