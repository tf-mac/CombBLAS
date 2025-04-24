#include "CombBLAS/Timer.h"

namespace combblas
{

Timer::Timer() : running_(false)
{
    startTime_ = {0, 0};
    endTime_ = {0, 0};
}

void Timer::start()
{
    gettimeofday(&startTime_, nullptr);
    running_ = true;
}

void Timer::stop()
{
    gettimeofday(&endTime_, nullptr);
    running_ = false;
}
double Timer::elapsedSeconds() const
{
    timeval stopTime = endTime_;
    if (running_) {
        gettimeofday(&stopTime, nullptr);
    }
    double seconds = stopTime.tv_sec - startTime_.tv_sec;
    double useconds = stopTime.tv_usec - startTime_.tv_usec;
    return seconds + useconds / 1e6;
}

}  // namespace combblas