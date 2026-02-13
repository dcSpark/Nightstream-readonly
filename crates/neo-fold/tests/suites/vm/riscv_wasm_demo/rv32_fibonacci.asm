# RV32 Fibonacci demo (mini-asm)
#
# Semantics expected by the wasm demo:
# - read n (u32) from RAM[0x104]
# - compute fib(n) with wrapping u32 arithmetic
# - write result (u32) to RAM[0x100]
# - halt via ecall
#
# Supported subset: addi, add, lw, sw, beq, jal, j, li (small), mv, ecall, nop

# load n
addi t0, x0, 0x104
lw   a0, 0(t0)        # a0 = n

# a=0, b=1
addi a1, x0, 0        # a
addi a2, x0, 1        # b

loop:
beq  a0, x0, done     # while (n != 0)
add  a3, a1, a2       # next = a + b
mv   a1, a2           # a = b
mv   a2, a3           # b = next
addi a0, a0, -1       # n--
jal  x0, loop

done:
addi t1, x0, 0x100
sw   a1, 0(t1)        # store result
ecall

