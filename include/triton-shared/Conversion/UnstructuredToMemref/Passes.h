//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef UNSTRUCTURED_TO_MEMREF_CONVERSION_PASSES_H
#define UNSTRUCTURED_TO_MEMREF_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::triton {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/UnstructuredToMemref/Passes.h.inc"

} // namespace mlir::triton

#endif
