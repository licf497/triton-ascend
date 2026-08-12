// RUN: triton-opt --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop --split-input-file %s | FileCheck %s

// Test fixpipe nz2nz f32: channel_split = true
// Producer matmul at block_id=3 -> Consumer matmul at block_id=5

// CHECK-LABEL: func.func @test_c2c_fixpipe_nz2nz_f32

// CHECK: memref.alloc() {{.*}} : memref<32x32xf32, #hivm.address_space<cbuf>>

// CHECK: hivm.hir.fixpipe {channel_split = true, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins({{%.*}} : tensor<32x32xf32>) outs({{%.*}} : memref<32x32xf32, #hivm.address_space<cbuf>>)

// CHECK: memref.memory_space_cast {{%.*}} {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf32, #hivm.address_space<cbuf>> to memref<32x32xf32>
// CHECK: bufferization.to_tensor {{%.*}} restrict writable {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf32> to tensor<32x32xf32>

// CHECK-NOT: ssbuffer.main_loop
// CHECK: return

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_c2c_fixpipe_nz2nz_f32(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %a: tensor<32x1xf32>, %b: tensor<1x32xf32>, %c: tensor<32x1xf32>, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i32 = arith.constant {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
    %c1_i32 = arith.constant {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "VECTOR"} 1 : i32
    %cst = arith.constant {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
    %init32x32 = tensor.empty() {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %fill32x32 = linalg.fill {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%init32x32 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %init32x1 = tensor.empty() {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x1xf32>
    %fill32x1 = linalg.fill {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%init32x1 : tensor<32x1xf32>) -> tensor<32x1xf32>
    scf.for %arg6 = %c0_i32 to %arg5 step %c1_i32  : i32 {
      %mm1 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%a, %b : tensor<32x1xf32>, tensor<1x32xf32>) outs(%fill32x32 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %mm2 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%mm1, %c : tensor<32x32xf32>, tensor<32x1xf32>) outs(%fill32x1 : tensor<32x1xf32>) -> tensor<32x1xf32>
    }
    return
  }
}

// -----

// Test fixpipe nz2nz f16: channel_split = false
// f16: numElemPerBlock = 256/16 = 16, channelSplit = (16 == 8) = false

// CHECK-LABEL: func.func @test_c2c_fixpipe_nz2nz_f16

// CHECK: memref.alloc() {{.*}} : memref<32x32xf16, #hivm.address_space<cbuf>>

// CHECK: hivm.hir.fixpipe {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins({{%.*}} : tensor<32x32xf16>) outs({{%.*}} : memref<32x32xf16, #hivm.address_space<cbuf>>)
// CHECK-NOT: channel_split

// CHECK: memref.memory_space_cast {{%.*}} {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf16, #hivm.address_space<cbuf>> to memref<32x32xf16>
// CHECK: bufferization.to_tensor {{%.*}} restrict writable {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf16> to tensor<32x32xf16>

// CHECK: return

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_c2c_fixpipe_nz2nz_f16(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %a: tensor<32x1xf16>, %b: tensor<1x32xf16>, %c: tensor<32x1xf16>, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i32 = arith.constant {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
    %c1_i32 = arith.constant {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "VECTOR"} 1 : i32
    %cst = arith.constant {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f16
    %init32x32 = tensor.empty() {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf16>
    %fill32x32 = linalg.fill {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%init32x32 : tensor<32x32xf16>) -> tensor<32x32xf16>
    %init32x1 = tensor.empty() {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x1xf16>
    %fill32x1 = linalg.fill {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%init32x1 : tensor<32x1xf16>) -> tensor<32x1xf16>
    scf.for %arg6 = %c0_i32 to %arg5 step %c1_i32  : i32 {
      %mm1 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%a, %b : tensor<32x1xf16>, tensor<1x32xf16>) outs(%fill32x32 : tensor<32x32xf16>) -> tensor<32x32xf16>
      %mm2 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%mm1, %c : tensor<32x32xf16>, tensor<32x1xf16>) outs(%fill32x1 : tensor<32x1xf16>) -> tensor<32x1xf16>
    }
    return
  }
}
