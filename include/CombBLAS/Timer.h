#pragma once

#include <sys/time.h>

#include <string>

namespace combblas
{

class Timer
{
   public:
    Timer();
    void start();
    void stop();
    double elapsedSeconds() const;

   private:
    timeval startTime_;
    timeval endTime_;
    bool running_;
};
}  // namespace combblas