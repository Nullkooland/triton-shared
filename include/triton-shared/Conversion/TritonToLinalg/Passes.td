//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_LINALG_CONVERSION_PASSES
#define TRITON_TO_LINALG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToLinalg : Pass<"triton-to-linalg", "mlir::ModuleOp"> {
  let summary = "Convert Triton to Linalg dialect";
  let options = [
      Option<"enableMakeGatherScatterTensorPtr", "enable-make-gather-scatter", "bool", /*default*/"true",
             "Enable make_gather_scatter_tptr support">,
      Option<"enableCollapseShape", "enable-collapse-shape", "bool", /*default*/"false",
             "Enable collapse shape pass">,
  ];
  let dependentDialects = [
    "func::FuncDialect",
    "arith::ArithDialect",
    "math::MathDialect",
    "memref::MemRefDialect",
    "bufferization::BufferizationDialect",
    "scf::SCFDialect",
    "cf::ControlFlowDialect",
    "affine::AffineDialect",
    "linalg::LinalgDialect",
    "tensor::TensorDialect",
    "tts::TritonStructuredDialect",
    "ttx::TritonTilingExtDialect",
  ];
}

def ReconcilePtrCasts : Pass<"reconcile-ptr-casts", "mlir::ModuleOp"> {
  let summary = "Convert unrealized_cast op between tt.ptr or ptr.ptr to memref to to_memref or from_memref";
  let dependentDialects = [
    "ptr::PtrDialect",
    "memref::MemRefDialect",
  ];
}

def TritonToPtr : Pass<"triton-to-ptr", "mlir::ModuleOp"> {
  let summary = "Convert Triton ops on pointers to the Ptr dialect";
  let dependentDialects = [
    "ptr::PtrDialect",
    "arith::ArithDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "linalg::LinalgDialect",
  ];
}

def CollapseShape : Pass</*cli-arg*/"collapse-shape", /*Op*/"mlir::ModuleOp"> {
  let summary = "Compress tensor dimensions to improve linalg{broadcast,transpose,fill,reduce} efficiency";
  let dependentDialects = [
    "tensor::TensorDialect",
    "linalg::LinalgDialect",
  ];
}

#endif
