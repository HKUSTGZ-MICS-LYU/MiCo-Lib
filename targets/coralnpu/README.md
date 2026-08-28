# MiCo CoralNPU Target

This target builds MiCo C code as an ELF32 for the CoralNPU RVV core and runs it with CoralNPU's RVV Verilator simulator.

## ISA And Memory

The default profile matches CoralNPU V2:

    rv32imf_zve32x_zicsr_zifencei_zbb
    ilp32

The default linker map is CoralNPU's 8 KiB ITCM at 0x00000000 and 32 KiB DTCM at 0x00010000. Keep the first tests small enough for those regions. The CoralNPU high-memory linker script is not paired with the current RVV Verilator executable.

## Prerequisites

The CoralNPU Bazel workspace supplies the RV32 compiler and the RVV Verilator model. Activate the Chipyard environment for the host Verilator dependencies:

    source ~/work/chipyard/env.sh

Set CORALNPU_ROOT to any CoralNPU checkout. The target has no repository-relative default:

    export CORALNPU_ROOT=/path/to/coralnpu

The target resolves the Bazel execution root and toolchain from that checkout automatically. Override the toolchain only when needed:

    export CORALNPU_TOOLCHAIN=/path/to/external/toolchain_coralnpu_v2

## Build And Run

Build the supplied tohost smoke test from the repository root:

    make -C project TARGET=coralnpu MAIN=tests/coralnpu_rvv_smoke BUILD=build/coralnpu-rvv compile
    make -C project TARGET=coralnpu MAIN=tests/coralnpu_rvv_smoke BUILD=build/coralnpu-rvv run-coralnpu

The run target first builds CoralNPU's existing RVV simulator target:

    //tests/verilator_sim:rvv_core_mini_axi_sim

The smoke test calls MiCo_Q4_MatMul and MiCo_Q4x2_MatMul, compares their results with scalar golden values, and writes tohost = 1 on success. The simulator exits successfully when it observes that value.

## 8x8 Performance

The benchmark at project/tests/coralnpu_matmul_bench.c runs an 8x8 Q4 MatMul: batch=8, inputs=8, outputs=8. It reads the CoralNPU guest mcycle CSR around one MatMul call and uses a configurable cycle threshold for the simulator pass/fail result.

Build the vector and scalar variants in separate build directories:

    make -C project TARGET=coralnpu CORALNPU_ROOT=$CORALNPU_ROOT CORALNPU_RVV=1 EXTRA_CFLAGS="-DITERATIONS=1 -DCYCLE_LIMIT=4294967295" MAIN=tests/coralnpu_matmul_bench BUILD=build/coralnpu-matmul-rvv compile
    make -C project TARGET=coralnpu CORALNPU_ROOT=$CORALNPU_ROOT CORALNPU_RVV=0 EXTRA_CFLAGS="-DITERATIONS=1 -DCYCLE_LIMIT=4294967295" MAIN=tests/coralnpu_matmul_bench BUILD=build/coralnpu-matmul-scalar compile

CORALNPU_RVV=1 selects rv32imf_zve32x_zicsr_zifencei_zbb and the RVV kernel; CORALNPU_RVV=0 selects the scalar rv32imf_zicsr_zifencei_zbb baseline. The cycle threshold is intentionally passed through EXTRA_CFLAGS because the Makefile compiles MAIN directly at link time.

On the current VRvvCoreMiniAxi configuration, with ITERATIONS=1, the measured guest kernel region was:

    RVV:    1956 cycles
    scalar: 9710 cycles
    speedup: 4.96x
    reduction: 79.9%

These are guest mcycle values for the kernel region only, excluding startup and simulator host time. To reproduce a value, binary-search CYCLE_LIMIT: a run passes when the measured cycle count is at most the threshold and fails through tohost otherwise.

## ELF Checks

    file project/tests/coralnpu_rvv_smoke.elf
    $CORALNPU_TOOLCHAIN/bin/riscv32-unknown-elf-readelf -h -A -l project/tests/coralnpu_rvv_smoke.elf
    $CORALNPU_TOOLCHAIN/bin/riscv32-unknown-elf-size project/tests/coralnpu_rvv_smoke.elf
    $CORALNPU_TOOLCHAIN/bin/riscv32-unknown-elf-objdump -d project/tests/coralnpu_rvv_smoke.elf

The ELF must report ELF32 and the CoralNPU RV32 Zve32x architecture. Its ITCM and DTCM sections must remain within the CoralNPU memory map. The disassembly should contain integer RVV instructions and no vector floating-point instructions.

## Debugging

Pass simulator flags through CORALNPU_SIM_FLAGS, for example:

    make -C project TARGET=coralnpu MAIN=tests/coralnpu_rvv_smoke BUILD=build/coralnpu-rvv run-coralnpu CORALNPU_SIM_FLAGS=--instr_trace

A missing compiler usually means the CoralNPU Bazel workspace has not populated its external toolchain. A timeout or memory overflow usually means the application exceeds the 8 KiB ITCM or 32 KiB DTCM configuration.
