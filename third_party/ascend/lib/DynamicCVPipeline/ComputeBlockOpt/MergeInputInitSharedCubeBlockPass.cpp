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

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace triton;

static constexpr const char *DEBUG_TYPE = "merge-input-init-shared-cube-block";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

namespace {

// Trace an operand's defining op back through fixpipe quant (trunc) ops to
// find the producing matmul, if any.
static linalg::MatmulOp resolveInputMatmulProducer(Value operand) {
  Operation *defOp = operand.getDefiningOp();
  while (defOp && CVPipeline::getFixpipePreQuantMode(defOp).has_value()) {
    defOp = defOp->getOperand(0).getDefiningOp();
  }
  return dyn_cast_if_present<linalg::MatmulOp>(defOp);
}

// Return the matmul whose result feeds both the consumer's init (directly) and
// one of its inputs (possibly through a trunc), or null if no such matmul.
static linalg::MatmulOp getSharedInputInitProducer(linalg::MatmulOp consumer) {
  auto inits = consumer.getDpsInits();
  if (inits.empty()) {
    return {};
  }
  auto producer =
      dyn_cast_if_present<linalg::MatmulOp>(inits.front().getDefiningOp());
  if (!producer) {
    return {};
  }
  for (Value input : consumer.getDpsInputs()) {
    if (resolveInputMatmulProducer(input).getOperation() ==
        producer.getOperation()) {
      return producer;
    }
  }
  return {};
}

// Merge the ops of consumerBlockId into producerBlockId when it does not
// create a cycle.
static bool tryMergeBlocks(int consumerBlockId, int producerBlockId,
                           const CVPipeline::MemoryDependenceGraph &memGraph,
                           CVPipeline::ComputeBlockIdManager &bm) {
  llvm::SmallVector<Operation *> consumerOps =
      bm.getOpsByBlockId(consumerBlockId);
  if (consumerOps.empty() ||
      CVPipeline::willCreateCycle(consumerOps, memGraph, producerBlockId, bm)) {
    LOG_DEBUG("Skip merging block " << consumerBlockId
                                    << ": would create cycle");
    return false;
  }
  LOG_DEBUG("Merging block " << consumerBlockId << " into block "
                             << producerBlockId);
  for (Operation *op : consumerOps) {
    bm.updateBlockId(op, producerBlockId);
  }
  return true;
}

// When the consumer's init comes from the producer matmul, split the init out
// of the matmul: accumulate into a zero-filled tensor instead, then add the
// original init back explicitly. This mirrors the init split in
// SplitMatmulPattern, but only the init is split (inputs stay unchanged).
static void splitInit(linalg::MatmulOp consumer, int consumerBlockId,
                      CVPipeline::ComputeBlockIdManager &bm) {
  auto inits = consumer.getDpsInits();
  auto inputs = consumer.getDpsInputs();
  if (inits.empty() || inputs.size() < 2) {
    return;
  }
  Value bias = inits.front();
  auto outputType = dyn_cast<RankedTensorType>(bias.getType());
  if (!outputType) {
    return;
  }
  Type elmType = outputType.getElementType();
  if (!isa<FloatType, IntegerType>(elmType)) {
    return;
  }

  Location loc = consumer.getLoc();
  IRRewriter rewriter(consumer.getContext());
  rewriter.setInsertionPoint(consumer);

  auto tag = [&](Operation *op, int blockId, StringRef coreType) {
    op->setAttr(CVPipeline::kCoreType, rewriter.getStringAttr(coreType));
    bm.updateBlockId(op, blockId);
  };

  // Step 1: zero-filled accumulator with the same shape/type as the init.
  llvm::SmallVector<Value> dynamicSizes;
  for (int64_t i = 0; i < outputType.getRank(); ++i) {
    if (outputType.isDynamicDim(i)) {
      dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, bias, i));
    }
  }
  auto emptyOp =
      rewriter.create<tensor::EmptyOp>(loc, outputType, dynamicSizes);
  Value zeroValue;
  if (auto floatType = dyn_cast<FloatType>(elmType)) {
    APFloat zeroAPFloat = APFloat::getZero(floatType.getFloatSemantics());
    zeroValue =
        rewriter.create<arith::ConstantFloatOp>(loc, floatType, zeroAPFloat)
            .getResult();
  } else {
    zeroValue =
        rewriter
            .create<arith::ConstantIntOp>(loc, cast<IntegerType>(elmType), 0)
            .getResult();
  }
  auto fillOp =
      rewriter.create<linalg::FillOp>(loc, zeroValue, emptyOp.getResult());
  tag(zeroValue.getDefiningOp(), consumerBlockId, CVPipeline::kCoreTypeVector);
  tag(emptyOp, consumerBlockId, CVPipeline::kCoreTypeVector);
  tag(fillOp, consumerBlockId, CVPipeline::kCoreTypeVector);

  // Step 2: new matmul accumulating into the zero-filled tensor.
  auto newMatmul = rewriter.create<linalg::MatmulOp>(
      loc, inputs, ValueRange(fillOp.getResult(0)));
  NamedAttrList attrs(consumer->getAttrDictionary());
  constexpr llvm::StringLiteral kShouldRemoveAttrs[] = {
      "operandSegmentSizes", "res_attrs", "arg_attrs"};
  for (auto attr : kShouldRemoveAttrs) {
    attrs.erase(attr);
  }
  newMatmul->setAttrs(attrs);
  tag(newMatmul, consumerBlockId, CVPipeline::kCoreTypeCube);

  // Detach the about-to-be-erased consumer from the block id manager.
  bm.updateBlockId(consumer, -1);
  rewriter.replaceOp(consumer, newMatmul);
  Value newOutValue = newMatmul.getResult(0);

  // Step 3: add(new_matmul_result, original_init) to restore the bias.
  rewriter.setInsertionPointAfter(newMatmul);
  Operation *addOp;
  if (isa<FloatType>(elmType)) {
    addOp =
        rewriter.create<arith::AddFOp>(loc, newOutValue, bias).getOperation();
  } else {
    addOp =
        rewriter.create<arith::AddIOp>(loc, newOutValue, bias).getOperation();
  }
  addOp->setAttr(CVPipeline::kAddFromMatmul, rewriter.getUnitAttr());
  tag(addOp, bm.getNextId(), CVPipeline::kCoreTypeVector);

  newOutValue.replaceUsesWithIf(addOp->getResult(0), [&](OpOperand &operand) {
    return operand.getOwner() != addOp;
  });

  LOG_DEBUG("Split init of matmul in block " << consumerBlockId
                                             << " into a zero-filled "
                                                "accumulator + add-back bias");
}

class MergeInputInitSharedCubeBlockPass
    : public PassWrapper<MergeInputInitSharedCubeBlockPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MergeInputInitSharedCubeBlockPass)

  MergeInputInitSharedCubeBlockPass() = default;

  StringRef getArgument() const override {
    return "merge-input-init-shared-cube-block";
  }

  StringRef getDescription() const override {
    return "Merge cube blocks where a matmul result is used as both input and "
           "init of the next matmul";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (CVPipeline::hasFallbackAttr(module)) {
      return;
    }

    auto &aa = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memGraph(module, aa);
    CVPipeline::ComputeBlockIdManager bm(module);

    // Collect matmuls first; merging rewrites block_id in place.
    llvm::SmallVector<linalg::MatmulOp> matmuls;
    module.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });

    for (linalg::MatmulOp consumer : matmuls) {
      auto producer = getSharedInputInitProducer(consumer);
      if (!producer) {
        continue;
      }

      auto producerBlockId = CVPipeline::getOpBlockId(producer.getOperation());
      auto consumerBlockId = CVPipeline::getOpBlockId(consumer.getOperation());
      if (!producerBlockId || !consumerBlockId ||
          *producerBlockId == *consumerBlockId) {
        continue;
      }

      if (!tryMergeBlocks(*consumerBlockId, *producerBlockId, memGraph, bm)) {
        splitInit(consumer, *consumerBlockId, bm);
      }
    }
  }
};

} // namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createMergeInputInitSharedCubeBlockPass() {
  return std::make_unique<MergeInputInitSharedCubeBlockPass>();
}

} // namespace triton
} // namespace mlir
