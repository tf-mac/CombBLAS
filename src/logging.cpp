//
// Created by Yuxi Hong on 2/24/25.
//

#include "CombBLAS/logging.h"

namespace combblas
{
char *curr_time()
{
    time_t raw_time = time(nullptr);
    struct tm *time_info = localtime(&raw_time);
    static char now_time[64];
    now_time[strftime(now_time, sizeof(now_time), "%Y-%m-%d %H:%M:%S", time_info)] = '\0';
    return now_time;
}

int get_pid()
{
    static int pid = getpid();
    return pid;
}

long int get_tid()
{
    thread_local long int tid = syscall(SYS_gettid);
    return tid;
}

}  // namespace combblas