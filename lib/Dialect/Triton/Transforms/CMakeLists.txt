#===------------------------------------------------------------------------===#
#
# Copyright (c) Triton Project Contributors.
#
#===------------------------------------------------------------------------===#

add_triton_library(TritonSharedTransforms
  AddLLVMDebugInfo.cpp

  DEPENDS
  TritonSharedTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  TritonIR
)
