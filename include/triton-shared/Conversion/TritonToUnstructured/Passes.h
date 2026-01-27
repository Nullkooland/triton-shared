#ifndef TRITON_TO_UNSTRUCTURED_CONVERSION_PASSES_H
#define TRITON_TO_UNSTRUCTURED_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::triton {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TritonToUnstructured/Passes.h.inc"

} // namespace mlir::triton

#endif
