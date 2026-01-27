//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "triton-shared/Conversion/StructuredToMemref/Passes.h"
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonPtrToMemref/Passes.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonToStructured/Passes.h"
#include "triton-shared/Conversion/TritonToUnstructured/Passes.h"
#include "triton-shared/Conversion/UnstructuredToMemref/Passes.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONTOLINALG
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"
} // namespace mlir::triton

namespace {

class TritonToLinalgPass
    : public mlir::triton::impl::TritonToLinalgBase<TritonToLinalgPass> {
  using TritonToLinalgBase<TritonToLinalgPass>::TritonToLinalgBase;

public:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    PassManager pm(&getContext(), moduleOp.getOperationName());

    {
      triton::TritonToStructuredOptions options;
      options.enableMakeGatherScatterTensorPtr =
          enableMakeGatherScatterTensorPtr;
      pm.addPass(triton::createTritonToStructured(options));
    }

    // Erase dead code and fold constants created during lowering
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());

    pm.addPass(triton::createTritonToUnstructured());

    {
      triton::TritonArithToLinalgOptions options;
      options.tensorPtrToLinalg = true;
      pm.addPass(triton::createTritonArithToLinalg(options));
    }

    pm.addPass(triton::createStructuredToMemref());
    pm.addPass(triton::createUnstructuredToMemref());
    pm.addPass(triton::createTritonPtrToMemref());
    pm.addPass(triton::createTritonToPtr());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(triton::createReconcilePtrCasts());

    // Now that remove-dead-values fully works with linalg ops, clean up the IR
    // again, particularly unused loop iter-args that were created
    // during triton-to-structured.
    pm.addPass(createRemoveDeadValuesPass());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    if (enableCollapseShape) {
      // Canonicalizer pass will rewrite tensor.expand_shape(linalg.fill) to
      // linalg.fill(tensor.expand_shape) so we need to run it before
      // collapseShape pass
      pm.addPass(triton::createCollapseShape());
    }

    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace
