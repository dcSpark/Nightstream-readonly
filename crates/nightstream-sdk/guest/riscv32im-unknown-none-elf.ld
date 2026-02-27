ENTRY(_start);

SECTIONS
{
  . = 0x0;

  /* Provide a default fixed stack pointer (top of our tiny test RAM).
     Can be overridden by the linker via --defsym=_STACK_PTR=... */
  PROVIDE(_STACK_PTR = 0x7f0);

  /* Neo guest ABI addresses (RAM). */
  PROVIDE(_NEO_OUTPUT_ADDR = 0x100);
  PROVIDE(_NEO_INPUT_ADDR  = 0x104);

  /* Heap region for optional bump allocator (RAM). */
  PROVIDE(_NEO_HEAP_START = 0x200);
  PROVIDE(_NEO_HEAP_END   = 0x700);

  .neo_start : ALIGN(4)
  {
    KEEP(*(.neo_start))
    *(.text*)
  }

  .rodata : ALIGN(4)
  {
    *(.rodata*)
    *(.srodata*)
  }

  .data : ALIGN(4)
  {
    *(.data*)
    *(.sdata*)
  }

  .bss (NOLOAD) : ALIGN(4)
  {
    *(.bss*)
    *(.sbss*)
    *(COMMON)
  }

  /DISCARD/ :
  {
    *(.eh_frame*)
    *(.comment*)
  }
}
