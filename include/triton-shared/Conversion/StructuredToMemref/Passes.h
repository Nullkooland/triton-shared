#ifndef TRITON_STRUCTURED_TO_MEMREF_CONVERSION_PASSES_H
#define TRITON_STRUCTURED_TO_MEMREF_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::triton {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/StructuredToMemref/Passes.h.inc"

void populateStructuredToMemrefConversionPatterns(RewritePatternSet &patterns,
                                                  TypeConverter &typeConverter);

} // namespace mlir::triton

#endif
