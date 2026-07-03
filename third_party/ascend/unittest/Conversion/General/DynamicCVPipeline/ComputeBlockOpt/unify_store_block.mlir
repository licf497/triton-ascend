// RUN: triton-opt --unify-store-block %s | FileCheck %s

module {
  // ============================================
  // Test 1: VECTOR producer hit - basic unification
  //
  // Pattern (from real kernel IR):
  //   arith.truncf(VECTOR, block_id=7)
  //     -> tensor.extract_slice(block_id=8)
  //     -> bufferization.materialize_in_destination(block_id=8)
  //   dest chain: memref.reinterpret_cast(block_id=8) -> memref.subview(block_id=8)
  //
  // producer = arith.truncf (VECTOR_ONLY, block_id=7).
  // The store chain (extract_slice, reinterpret_cast, subview, materialize)
  // and the scalar dep (%c0 used by reinterpret_cast) are all at block_id=8.
  // After pass: all unified to producer's block_id=7.
  // ============================================
  // CHECK-LABEL: func @test_vector_producer_hit
  func.func @test_vector_producer_hit(%arg0: memref<?xf16> {tt.divisibility = 16 : i32}, %in: tensor<64x64xf32>) {
    // CHECK: arith.constant {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c0 = arith.constant {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    %trunc = arith.truncf %in {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    %extract = tensor.extract_slice %trunc[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    // CHECK: memref.reinterpret_cast %{{.*}} {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    %reinterpret = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128, 64], strides: [64, 1] {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %subview = memref.subview %reinterpret[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: bufferization.materialize_in_destination %{{.*}} {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    bufferization.materialize_in_destination %extract in writable %subview {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    return
  }

  // ============================================
  // Test 2: CUBE producer skip - no unification
  //
  // Pattern:
  //   linalg.matmul(CUBE, block_id=1) -> tensor.extract_slice(block_id=2)
  //     -> bufferization.materialize_in_destination(block_id=2)
  //   (no truncf; producer traced directly to matmul)
  //
  // producer = linalg.matmul (CUBE_ONLY) -> pass skips this store.
  // All block_ids remain unchanged.
  // ============================================
  // CHECK-LABEL: func @test_cube_producer_skip
  func.func @test_cube_producer_skip(%arg0: memref<?xf16> {tt.divisibility = 16 : i32}, %A: tensor<64x64xf16>, %B: tensor<64x64xf16>, %init: tensor<64x64xf16>) {
    // CHECK: arith.constant {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c0 = arith.constant {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    // CHECK: linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"}
    %matmul = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%A, %B : tensor<64x64xf16>, tensor<64x64xf16>) outs(%init : tensor<64x64xf16>) -> tensor<64x64xf16>
    // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    %extract = tensor.extract_slice %matmul[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    // CHECK: memref.reinterpret_cast %{{.*}} {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    %reinterpret = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128, 64], strides: [64, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %subview = memref.subview %reinterpret[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: bufferization.materialize_in_destination %{{.*}} {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    bufferization.materialize_in_destination %extract in writable %subview {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    return
  }

  // ============================================
  // Test 3: Store inside scf.for with scalar index dependency
  //
  // store lives in scf.for body block (block_id=48). The dest chain has two
  // view ops: reinterpret_cast (in the function block) and subview (in the
  // loop body). Both are collected by collectDestViewOps and unified to 47.
  //
  // The scalar index op %offset (arith.muli) is inside the loop body; its
  // block_id (48) matches the store's (48), so it is collected as a scalar
  // dep and unified to 47.
  //
  // Outer-block scalars (%c0/%c1/%c10/%c64 in the function block) are also
  // collected: %c0 feeds reinterpret_cast's offset, %c64 feeds muli, and
  // %c0/%c1/%c10 feed the enclosing scf.for's lb/step/ub (control-flow
  // ancestor ops are seeded into the scalar-dep worklist). All are unified
  // to 47.
  //
  // producer = arith.truncf (VECTOR_ONLY, block_id=47).
  // ============================================
  // CHECK-LABEL: func @test_store_in_scf_for_scalar_deps
  func.func @test_store_in_scf_for_scalar_deps(%arg0: memref<?xf16> {tt.divisibility = 16 : i32}, %in: tensor<64x64xf32>) {
    // CHECK: arith.constant {ssbuffer.block_id = 47 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c0 = arith.constant {ssbuffer.block_id = 48 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    // CHECK: arith.constant {ssbuffer.block_id = 47 : i32, ssbuffer.core_type = "VECTOR"} 1 : index
    %c1 = arith.constant {ssbuffer.block_id = 48 : i32, ssbuffer.core_type = "VECTOR"} 1 : index
    // CHECK: arith.constant {ssbuffer.block_id = 47 : i32, ssbuffer.core_type = "VECTOR"} 10 : index
    %c10 = arith.constant {ssbuffer.block_id = 48 : i32, ssbuffer.core_type = "VECTOR"} 10 : index
    // CHECK: arith.constant {ssbuffer.block_id = 47 : i32, ssbuffer.core_type = "VECTOR"} 64 : index
    %c64 = arith.constant {ssbuffer.block_id = 48 : i32, ssbuffer.core_type = "VECTOR"} 64 : index
    // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 47 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    %trunc = arith.truncf %in {ssbuffer.block_id = 47 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    // dest-chain view op (in function block, block_id=48 == storeBlockId),
    // collected by collectDestViewOps and unified to 47.
    // CHECK: memref.reinterpret_cast %{{.*}} {ssbuffer.block_id = 47 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    %reinterpret = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128, 64], strides: [64, 1] {ssbuffer.block_id = 48 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    scf.for %iv = %c0 to %c10 step %c1 {
      // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 47 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
      %extract = tensor.extract_slice %trunc[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 48 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
      // scalar dep: %offset is in the same scf.for body block as the store,
      // its block_id (48) matches the store's, so it is unified to 47.
      // CHECK: arith.muli %{{.*}}, %{{.*}} {ssbuffer.block_id = 47 : i32, ssbuffer.core_type = "VECTOR"} : index
      %offset = arith.muli %iv, %c64 {ssbuffer.block_id = 48 : i32, ssbuffer.core_type = "VECTOR"} : index
      // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 47 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
      %subview = memref.subview %reinterpret[%offset, 0] [64, 64] [1, 1] {ssbuffer.block_id = 48 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
      // CHECK: bufferization.materialize_in_destination %{{.*}} {ssbuffer.block_id = 47 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
      bufferization.materialize_in_destination %extract in writable %subview {ssbuffer.block_id = 48 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    }
    return
  }

  // ============================================
  // Test 4: hivm.hir.store use case
  //
  // Same shape as Test 1 but using hivm.hir.store instead of
  // bufferization.materialize_in_destination. Store source operand is the
  // tensor %extract (getSrc()), dest is the memref %subview (getDst()).
  //
  // producer = arith.truncf (VECTOR_ONLY, block_id=57).
  // store + extract_slice + subview chain unify to 57.
  // ============================================
  // CHECK-LABEL: func @test_hivm_store
  func.func @test_hivm_store(%arg0: memref<?xf16> {tt.divisibility = 16 : i32}, %in: tensor<64x64xf32>) {
    // %c0 feeds reinterpret_cast's offset (scalar dep), unified to 57.
    // CHECK: arith.constant {ssbuffer.block_id = 57 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c0 = arith.constant {ssbuffer.block_id = 58 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 57 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    %trunc = arith.truncf %in {ssbuffer.block_id = 57 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 57 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    %extract = tensor.extract_slice %trunc[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 58 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    // CHECK: memref.reinterpret_cast %{{.*}} {ssbuffer.block_id = 57 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    %reinterpret = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128, 64], strides: [64, 1] {ssbuffer.block_id = 58 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 57 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %subview = memref.subview %reinterpret[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 58 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: hivm.hir.store ins({{.*}} : tensor<64x64xf16>) outs({{.*}} : memref<64x64xf16, strided<[64, 1], offset: ?>>) {ssbuffer.block_id = 57 : i32, ssbuffer.core_type = "VECTOR"}
    hivm.hir.store ins(%extract : tensor<64x64xf16>) outs(%subview : memref<64x64xf16, strided<[64, 1], offset: ?>>) {ssbuffer.block_id = 58 : i32, ssbuffer.core_type = "VECTOR"}
    return
  }

  // ============================================
  // Test 5: Multiple vector ops in chain - producer is the first vector
  // encountered when tracing back from store (closest to store), not the
  // upstream one.
  //
  // Data chain:
  //   %v1 = truncf (id=67) -> %v2 = addf (id=68) -> extract_slice (id=69) -> store (id=69)
  //
  // traceProducerOp pierces extract_slice (view), hits %v2 (VECTOR_ONLY) and
  // returns immediately. %v1 is NOT reached, so:
  //   producer = %v2 (id=68), stays 68
  //   %v1 stays 67 (not producer, not collected)
  //   store + extract_slice + dest view chain + %c0 unify to 68
  // ============================================
  // CHECK-LABEL: func @test_multiple_vector_producer_first
  func.func @test_multiple_vector_producer_first(%arg0: memref<?xf16> {tt.divisibility = 16 : i32}, %in: tensor<64x64xf32>) {
    // %c0 feeds reinterpret_cast's offset (scalar dep), unified to 68.
    // CHECK: arith.constant {ssbuffer.block_id = 68 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c0 = arith.constant {ssbuffer.block_id = 69 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    // upstream vector op, NOT the producer; stays 67.
    // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 67 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    %v1 = arith.truncf %in {ssbuffer.block_id = 67 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    // downstream vector op (closest to store), IS the producer; stays 68.
    // CHECK: arith.addf %{{.*}}, %{{.*}} {ssbuffer.block_id = 68 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16>
    %v2 = arith.addf %v1, %v1 {ssbuffer.block_id = 68 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16>
    // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 68 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    %extract = tensor.extract_slice %v2[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 69 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    // CHECK: memref.reinterpret_cast %{{.*}} {ssbuffer.block_id = 68 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    %reinterpret = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128, 64], strides: [64, 1] {ssbuffer.block_id = 69 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 68 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %subview = memref.subview %reinterpret[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 69 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: bufferization.materialize_in_destination %{{.*}} {ssbuffer.block_id = 68 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    bufferization.materialize_in_destination %extract in writable %subview {ssbuffer.block_id = 69 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    return
  }
}
