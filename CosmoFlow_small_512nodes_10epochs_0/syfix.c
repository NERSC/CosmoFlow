int sched_yield() { __asm__("pause"); return 0; }
