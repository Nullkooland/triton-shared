#ifndef TRITON_ARITH_TO_LINALG_CONVERSION_PASSES_H
#define TRITON_ARITH_TO_LINALG_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::triton {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"

void populateTritonArithToLinalgCanonicalizationPatterns(
    RewritePatternSet &patterns);

void populateTritonArithToLinalgConversionPatterns(bool pidsToFuncArgs,
                                                   bool addptrToLinalg,
                                                   bool assertToCf,
                                                   RewritePatternSet &patterns);

// Expand the triton pointer ops operating on pointers to linalg
void populateTritonTensorPtrConversionPatterns(RewritePatternSet &patterns);

} // namespace mlir::triton

#endif
