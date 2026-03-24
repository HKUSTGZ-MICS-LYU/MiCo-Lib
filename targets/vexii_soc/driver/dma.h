#pragma once

#include "type.h"
#include "io.h"

// Reader channel registers
#define DMA_READER_BASE              0x00
#define DMA_READER_LENGTH            0x04
#define DMA_READER_ENABLE            0x08
#define DMA_READER_LOOP              0x0C
#define DMA_READER_DONE              0x10
#define DMA_READER_OFFSET            0x14
#define DMA_READER_ERROR             0x18
#define DMA_READER_BUSY              0x1C

// Writer channel registers
#define DMA_WRITER_BASE              0x20
#define DMA_WRITER_LENGTH            0x24
#define DMA_WRITER_ENABLE            0x28
#define DMA_WRITER_LOOP              0x2C
#define DMA_WRITER_DONE              0x30
#define DMA_WRITER_OFFSET            0x34
#define DMA_WRITER_ERROR             0x38
#define DMA_WRITER_BUSY              0x3C

// Interrupt registers
#define DMA_INTERRUPT_ENABLE         0x40
#define DMA_INTERRUPT_PENDING        0x44

writeReg_u32(dma_reader_setBase,      DMA_READER_BASE)
readReg_u32 (dma_reader_getBase,      DMA_READER_BASE)
writeReg_u32(dma_reader_setLength,    DMA_READER_LENGTH)
readReg_u32 (dma_reader_getLength,    DMA_READER_LENGTH)
writeReg_u32(dma_reader_setEnable,    DMA_READER_ENABLE)
readReg_u32 (dma_reader_getEnable,    DMA_READER_ENABLE)
writeReg_u32(dma_reader_setLoop,      DMA_READER_LOOP)
readReg_u32 (dma_reader_getLoop,      DMA_READER_LOOP)
readReg_u32 (dma_reader_getDone,      DMA_READER_DONE)
readReg_u32 (dma_reader_getOffset,    DMA_READER_OFFSET)
readReg_u32 (dma_reader_getError,     DMA_READER_ERROR)
readReg_u32 (dma_reader_getBusy,      DMA_READER_BUSY)

writeReg_u32(dma_writer_setBase,      DMA_WRITER_BASE)
readReg_u32 (dma_writer_getBase,      DMA_WRITER_BASE)
writeReg_u32(dma_writer_setLength,    DMA_WRITER_LENGTH)
readReg_u32 (dma_writer_getLength,    DMA_WRITER_LENGTH)
writeReg_u32(dma_writer_setEnable,    DMA_WRITER_ENABLE)
readReg_u32 (dma_writer_getEnable,    DMA_WRITER_ENABLE)
writeReg_u32(dma_writer_setLoop,      DMA_WRITER_LOOP)
readReg_u32 (dma_writer_getLoop,      DMA_WRITER_LOOP)
readReg_u32 (dma_writer_getDone,      DMA_WRITER_DONE)
readReg_u32 (dma_writer_getOffset,    DMA_WRITER_OFFSET)
readReg_u32 (dma_writer_getError,     DMA_WRITER_ERROR)
readReg_u32 (dma_writer_getBusy,      DMA_WRITER_BUSY)

writeReg_u32(dma_setInterruptEnable,  DMA_INTERRUPT_ENABLE)
readReg_u32 (dma_getInterruptEnable,  DMA_INTERRUPT_ENABLE)
readReg_u32 (dma_getInterruptPending, DMA_INTERRUPT_PENDING)

static void dma_reset(u32 reg){
    dma_reader_setEnable(reg, 0);
    dma_writer_setEnable(reg, 0);
    dma_reader_setLoop(reg, 0);
    dma_writer_setLoop(reg, 0);
    dma_setInterruptEnable(reg, 0);
}

static void dma_config_reader(u32 reg, u32 base, u32 lengthBytes, u32 loop){
    dma_reader_setBase(reg, base);
    dma_reader_setLength(reg, lengthBytes);
    dma_reader_setLoop(reg, loop ? 1 : 0);
}

static void dma_config_writer(u32 reg, u32 base, u32 lengthBytes, u32 loop){
    dma_writer_setBase(reg, base);
    dma_writer_setLength(reg, lengthBytes);
    dma_writer_setLoop(reg, loop ? 1 : 0);
}

static void dma_start_reader(u32 reg){
    dma_reader_setEnable(reg, 1);
}

static void dma_start_writer(u32 reg){
    dma_writer_setEnable(reg, 1);
}

static void dma_stop_reader(u32 reg){
    dma_reader_setEnable(reg, 0);
}

static void dma_stop_writer(u32 reg){
    dma_writer_setEnable(reg, 0);
}

static void dma_start(u32 reg){
    dma_start_reader(reg);
    dma_start_writer(reg);
}

static void dma_stop(u32 reg){
    dma_stop_reader(reg);
    dma_stop_writer(reg);
}

static u32 dma_wait_reader_done(u32 reg, u32 timeoutCycles){
    while(timeoutCycles--){
        if(dma_reader_getDone(reg)) return 1;
    }
    return 0;
}

static u32 dma_wait_writer_done(u32 reg, u32 timeoutCycles){
    while(timeoutCycles--){
        if(dma_writer_getDone(reg)) return 1;
    }
    return 0;
}

static u32 dma_has_error(u32 reg){
    return (dma_reader_getError(reg) | dma_writer_getError(reg)) != 0;
}
