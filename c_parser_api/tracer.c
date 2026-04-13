/*
 * tracer.c - Function Call Flow Tracer
 *
 * Build:
 *   gcc -shared -fPIC -o libtracer.so tracer.c
 *
 * Run:
 *   LD_PRELOAD=./libtracer.so TRACE_OUTPUT=trace.log ./program
 *
 * Environment Variables:
 *   TRACE_OUTPUT      - Output file path (default: call_trace.log)
 *   TRACE_MAX_DEPTH   - Maximum call depth (default: 128)
 *   TRACE_MAX_ENTRIES - Maximum number of entries (default: 10000000)
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include <sys/syscall.h>

static __thread int depth = 0;
static FILE *trace_fp = NULL;
static int max_depth = 128;
static unsigned long max_entries = 10000000;
static volatile unsigned long entry_count = 0;
static pthread_mutex_t write_lock = PTHREAD_MUTEX_INITIALIZER;



__attribute__((constructor))
void tracer_init(void) {
    const char *path = getenv("TRACE_OUTPUT");
    const char *depth_str = getenv("TRACE_MAX_DEPTH");
    const char *entries_str = getenv("TRACE_MAX_ENTRIES");

    trace_fp = fopen(path ? path : "call_trace.log", "w");
    if (!trace_fp) {
        fprintf(stderr, "[tracer] failed to open trace file\n");
        return;
    }

    if (depth_str) max_depth = atoi(depth_str);
    if (entries_str) max_entries = strtoul(entries_str, NULL, 10);

    // /* Get the path of the executable via /proc/self/exe */
    // char exe_path[4096];
    // ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    // if (len > 0) {
    //     exe_path[len] = '\0';
    // } else {
    //     exe_path[0] = '\0';
    // }

    // /* Get the base address of the executable from /proc/self/maps */
    // unsigned long base_addr = 0;
    // FILE *maps = fopen("/proc/self/maps", "r");
    // if (maps) {
    //     char line[4096];
    //     while (fgets(line, sizeof(line), maps)) {
    //         if (exe_path[0] && strstr(line, exe_path)) {
    //             base_addr = strtoul(line, NULL, 16);
    //             break;
    //         }
    //     }
    //     fclose(maps);
    // }

    // fprintf(trace_fp, "# call_trace v1 max_depth=%d max_entries=%lu base=0x%lx exe=%s\n",
    //         max_depth, max_entries, base_addr, exe_path);
    // fflush(trace_fp);

    /* Get the path of the executable via /proc/self/exe */
    char exe_path[4096];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len > 0) {
        exe_path[len] = '\0';
    } else {
        exe_path[0] = '\0';
    }

    /* Dump the entire /proc/self/maps to the log */
    FILE *maps = fopen("/proc/self/maps", "r");
    if (maps) {
        char line[4096];
        while (fgets(line, sizeof(line), maps)) {
            line[strcspn(line, "\n")] = '\0';
            fprintf(trace_fp, "# map %s\n", line);
        }
        fclose(maps);
    }

    fprintf(trace_fp, "# call_trace v1 max_depth=%d max_entries=%lu exe=%s\n",
            max_depth, max_entries, exe_path);
    fflush(trace_fp);
}


static __thread pid_t cached_tid = 0;

__attribute__((no_instrument_function))
static pid_t get_tid(void) {
    if (!cached_tid) {
        cached_tid = (pid_t)syscall(SYS_gettid);
    }
    return cached_tid;
}


__attribute__((no_instrument_function))
void mcount(void) {
    void *func = __builtin_return_address(0);
    void *caller = __builtin_return_address(1);
    if (!trace_fp) return;
    if (depth >= max_depth) { depth++; return; }
    if (__sync_fetch_and_add(&entry_count, 1) >= max_entries) return;

    pthread_mutex_lock(&write_lock);
    //fprintf(trace_fp, "E %d %p %p\n", depth, func, caller);
    fprintf(trace_fp, "E %d %d %p %p\n", get_tid(), depth, func, caller);
    pthread_mutex_unlock(&write_lock);

    depth++;
}

__attribute__((destructor))
void tracer_fini(void) {
    if (trace_fp) {
        fprintf(trace_fp, "# total_entries=%lu\n", entry_count);
        fclose(trace_fp);
        trace_fp = NULL;
    }
}

__attribute__((no_instrument_function))
void __cyg_profile_func_enter(void *func, void *caller) {
    if (!trace_fp) return;
    if (depth >= max_depth) { depth++; return; }
    if (__sync_fetch_and_add(&entry_count, 1) >= max_entries) return;

    pthread_mutex_lock(&write_lock);
    //fprintf(trace_fp, "E %d %p %p\n", depth, func, caller);
    fprintf(trace_fp, "E %d %d %p %p\n", get_tid(), depth, func, caller);
    pthread_mutex_unlock(&write_lock);

    depth++;
}

__attribute__((no_instrument_function))
void __cyg_profile_func_exit(void *func, void *caller) {
    if (!trace_fp) return;
    depth--;
    if (depth >= max_depth) return;
    if (entry_count >= max_entries) return;

    __sync_fetch_and_add(&entry_count, 1);

    pthread_mutex_lock(&write_lock);
    //fprintf(trace_fp, "X %d %p %p\n", depth, func, caller);
    fprintf(trace_fp, "X %d %d %p %p\n", get_tid(), depth, func, caller);
    pthread_mutex_unlock(&write_lock);
}