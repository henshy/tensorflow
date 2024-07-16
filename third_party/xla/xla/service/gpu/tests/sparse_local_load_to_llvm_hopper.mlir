// RUN: xla-opt %s -split-input-file --sparse-local-load-to-llvm | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase=2, maxPhase=4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 64, 16]}>
#dot_meta_enc = #triton_gpu.sparse_dot_meta<{parent=#mma}>

module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: convert_sparse_local_load
  tt.func @convert_sparse_local_load(%A: tensor<64x32xf16, #blocked>, %B: tensor<64x64xf16, #blocked>, %meta: tensor<64x4xi16, #blocked>) {
    %A_alloc = triton_gpu.local_alloc %A {allocation.offset = 0 : i32} : (tensor<64x32xf16, #blocked>) -> !tt.memdesc<64x32xf16, #shared, #triton_gpu.shared_memory>
    %B_alloc = triton_gpu.local_alloc %B {allocation.offset = 4096 : i32} : (tensor<64x64xf16, #blocked>) -> !tt.memdesc<64x64xf16, #shared, #triton_gpu.shared_memory>
    // CHECK-COUNT-2: llvm.load %[[_:.*]] : !llvm.ptr<3> -> i16
    %meta_alloc = triton_gpu.local_alloc %meta {allocation.offset = 12288 : i32} : (tensor<64x4xi16, #blocked>) -> !tt.memdesc<64x4xi16, #shared, #triton_gpu.shared_memory>
    %meta_reg = triton_gpu.local_load %meta_alloc : !tt.memdesc<64x4xi16, #shared, #triton_gpu.shared_memory> -> tensor<64x4xi16, #dot_meta_enc>
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %D = triton_gpu.sparse_dot %A_alloc, %B_alloc, %acc, %meta_reg : !tt.memdesc<64x32xf16, #shared, #triton_gpu.shared_memory> meta tensor<64x4xi16, #dot_meta_enc> * !tt.memdesc<64x64xf16, #shared, #triton_gpu.shared_memory> -> tensor<64x64xf32, #mma>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase=2, maxPhase=4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 64, 16]}>

module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: skip_pass_if_no_sparse_loads
  tt.func @skip_pass_if_no_sparse_loads(%A: tensor<64x32xf16, #blocked>, %B: tensor<32x64xf16, #blocked>) {
    // CHECK-NOT: llvm
    // CHECK-NOT: barrier
    %A_alloc = triton_gpu.local_alloc %A {allocation.offset = 0 : i32} : (tensor<64x32xf16, #blocked>) -> !tt.memdesc<64x32xf16, #shared, #triton_gpu.shared_memory>
    %B_alloc = triton_gpu.local_alloc %B {allocation.offset = 12288 : i32} : (tensor<32x64xf16, #blocked>) -> !tt.memdesc<32x64xf16, #shared, #triton_gpu.shared_memory>
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %D = triton_nvidia_gpu.warp_group_dot %A_alloc, %B_alloc, %acc : !tt.memdesc<64x32xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<32x64xf16, #shared, #triton_gpu.shared_memory> -> tensor<64x64xf32, #mma>
    tt.return
  }
}
