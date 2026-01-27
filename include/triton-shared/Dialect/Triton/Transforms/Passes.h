//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SHARED_TRANSFORMS_PASSES_H
#define TRITON_SHARED_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::triton {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "triton-shared/Dialect/Triton/Transforms/Passes.h.inc"

} // namespace mlir::triton

#endif
