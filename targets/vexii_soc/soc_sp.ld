RAM_ORIGIN = 0x80000000;
RAM_LENGTH   = DEFINED(RAM_SIZE)   ? RAM_SIZE   : 16K;
SPARSE_RAM_ORIGIN = 0x82000000;
SPARSE_RAM_LENGTH = DEFINED(SPARSE_RAM_SIZE) ? SPARSE_RAM_SIZE : 64M;
__heap_size  = DEFINED(HEAP_SIZE)  ? HEAP_SIZE  : 1K;
__stack_size = DEFINED(STACK_SIZE) ? STACK_SIZE : 1K;


OUTPUT_ARCH( "riscv" )

ENTRY( _start )

MEMORY
{
  ram_exec  (wxai!r) : ORIGIN = RAM_ORIGIN, LENGTH = RAM_LENGTH
  ram_data  (wxai!r) : ORIGIN = SPARSE_RAM_ORIGIN, LENGTH = SPARSE_RAM_LENGTH
}

PHDRS
{
  ram_exec PT_LOAD;
  ram_data PT_LOAD;
}

SECTIONS
{
  __stack_size = DEFINED(__stack_size) ? __stack_size : 2K;

  .init           :
  {
    KEEP (*(SORT_NONE(.init)))
  } >ram_exec AT>ram_exec :ram_exec

  .text           :
  {
    *(.text.unlikely .text.unlikely.*)
    *(.text.startup .text.startup.*)
    *(.text .text.*)
    *(.gnu.linkonce.t.*)
    *(.note.gnu.build-id)
  } >ram_exec AT>ram_exec :ram_exec

  .fini           :
  {
    KEEP (*(SORT_NONE(.fini)))
  } >ram_exec AT>ram_exec :ram_exec

  PROVIDE (__etext = .);
  PROVIDE (_etext = .);
  PROVIDE (etext = .);

  . = ALIGN(4);

  .preinit_array  :
  {
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP (*(.preinit_array))
    PROVIDE_HIDDEN (__preinit_array_end = .);
  } >ram_data AT>ram_data :ram_data

  .init_array     :
  {
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP (*(SORT_BY_INIT_PRIORITY(.init_array.*) SORT_BY_INIT_PRIORITY(.ctors.*)))
    KEEP (*(.init_array EXCLUDE_FILE (*crtbegin.o *crtbegin?.o *crtend.o *crtend?.o ) .ctors))
    PROVIDE_HIDDEN (__init_array_end = .);
  } >ram_data AT>ram_data :ram_data

  .fini_array     :
  {
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP (*(SORT_BY_INIT_PRIORITY(.fini_array.*) SORT_BY_INIT_PRIORITY(.dtors.*)))
    KEEP (*(.fini_array EXCLUDE_FILE (*crtbegin.o *crtbegin?.o *crtend.o *crtend?.o ) .dtors))
    PROVIDE_HIDDEN (__fini_array_end = .);
  } >ram_data AT>ram_data :ram_data

  .ctors          :
  {
    /* gcc uses crtbegin.o to find the start of
       the constructors, so we make sure it is
       first.  Because this is a wildcard, it
       doesn't matter if the user does not
       actually link against crtbegin.o; the
       linker won't look for a file to match a
       wildcard.  The wildcard also means that it
       doesn't matter which directory crtbegin.o
       is in.  */
    KEEP (*crtbegin.o(.ctors))
    KEEP (*crtbegin?.o(.ctors))
    /* We don't want to include the .ctor section from
       the crtend.o file until after the sorted ctors.
       The .ctor section from the crtend file contains the
       end of ctors marker and it must be last */
    KEEP (*(EXCLUDE_FILE (*crtend.o *crtend?.o ) .ctors))
    KEEP (*(SORT(.ctors.*)))
    KEEP (*(.ctors))
  } >ram_data AT>ram_data :ram_data

  .dtors          :
  {
    KEEP (*crtbegin.o(.dtors))
    KEEP (*crtbegin?.o(.dtors))
    KEEP (*(EXCLUDE_FILE (*crtend.o *crtend?.o ) .dtors))
    KEEP (*(SORT(.dtors.*)))
    KEEP (*(.dtors))
  } >ram_data AT>ram_data :ram_data

  .lalign         :
  {
    . = ALIGN(4);
    PROVIDE( _data_lma = . );
  } >ram_data AT>ram_data :ram_data

  .dalign         :
  {
    . = ALIGN(4);
    PROVIDE( _data = . );
  } >ram_data AT>ram_data :ram_data

  .data          :
  {
    *(.rdata)
    *(.rodata .rodata.*)
    *(.gnu.linkonce.r.*)
    *(.data .data.*)
    *(.gnu.linkonce.d.*)
    . = ALIGN(8);
    PROVIDE( __global_pointer$ = . + 0x7F0 );
    *(.sdata .sdata.*)
    *(.gnu.linkonce.s.*)
    . = ALIGN(8);
    *(.srodata.cst16)
    *(.srodata.cst8)
    *(.srodata.cst4)
    *(.srodata.cst2)
    *(.srodata .srodata.*)
    . += 4; /* fix for linker false error message */
  } >ram_data AT>ram_data :ram_data

  . = ALIGN(4);
  PROVIDE( _edata = . );
  PROVIDE( edata = . );

  PROVIDE( _fbss = . );
  PROVIDE( __bss_start = . );
  .bss            :
  {
    *(.sbss*)
    *(.gnu.linkonce.sb.*)
    *(.bss .bss.*)
    *(.gnu.linkonce.b.*)
    *(COMMON)
    . = ALIGN(4);
  } >ram_data AT>ram_data :ram_data

  . = ALIGN(8);
  PROVIDE( _end = . );
  PROVIDE( end = . );

  .stack  : ALIGN(16)
  {
    PROVIDE( _heap_end = . );
    . = __stack_size;
    PROVIDE( _sp = . );
	__freertos_irq_stack_top = .;
  } >ram_data AT>ram_data :ram_data
}
