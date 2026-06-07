#ifndef TOOLS__STATS_HPP
#define TOOLS__STATS_HPP

#include <vector>

namespace tools
{
double percentile(std::vector<double> values, double q);
double p95(std::vector<double> values);
double max_value(const std::vector<double> & values);
double median(std::vector<int> values);
double mean(const std::vector<int> & values);
}  // namespace tools

#endif  // TOOLS__STATS_HPP
