#include "stats.hpp"

#include <algorithm>
#include <numeric>

namespace tools
{
double percentile(std::vector<double> values, double q)
{
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  auto idx = static_cast<size_t>(values.size() * q);
  if (idx >= values.size()) idx = values.size() - 1;
  return values[idx];
}

double p95(std::vector<double> values) { return percentile(std::move(values), 0.95); }

double max_value(const std::vector<double> & values)
{
  return values.empty() ? 0.0 : *std::max_element(values.begin(), values.end());
}

double median(std::vector<int> values)
{
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  return static_cast<double>(values[values.size() / 2]);
}

double mean(const std::vector<int> & values)
{
  if (values.empty()) return 0.0;
  const auto sum = std::accumulate(values.begin(), values.end(), 0.0);
  return sum / static_cast<double>(values.size());
}
}  // namespace tools
