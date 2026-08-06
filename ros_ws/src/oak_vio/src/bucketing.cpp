#include "oak_vio/bucketing.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace oak_vio {

std::vector<int> selectBucketed(const std::vector<Correspondence>& correspondences, int imageWidth, int imageHeight, const VioParams& params) {
    std::vector<int> selected;
    if(correspondences.empty() || imageWidth <= 0 || imageHeight <= 0) {
        return selected;
    }

    const int cols = std::max(1, params.bucketCols);
    const int rows = std::max(1, params.bucketRows);
    const int perBucket = std::max(1, params.maxFeaturesPerBucket);

    const double cellWidth = static_cast<double>(imageWidth) / cols;
    const double cellHeight = static_cast<double>(imageHeight) / rows;

    // Group indices by cell.
    std::vector<std::vector<int>> buckets(static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows));
    for(std::size_t i = 0; i < correspondences.size(); ++i) {
        const Correspondence& c = correspondences[i];
        const int col = std::clamp(static_cast<int>(c.zCur.x() / cellWidth), 0, cols - 1);
        const int row = std::clamp(static_cast<int>(c.zCur.y() / cellHeight), 0, rows - 1);
        buckets[static_cast<std::size_t>(row) * cols + col].push_back(static_cast<int>(i));
    }

    selected.reserve(std::min(correspondences.size(), buckets.size() * static_cast<std::size_t>(perBucket)));

    for(auto& bucket : buckets) {
        if(bucket.empty()) {
            continue;
        }
        if(bucket.size() > static_cast<std::size_t>(perBucket)) {
            // Partial sort is enough — we only need the best `perBucket`.
            std::partial_sort(bucket.begin(),
                              bucket.begin() + perBucket,
                              bucket.end(),
                              [&correspondences](int lhs, int rhs) {
                                  const Correspondence& a = correspondences[static_cast<std::size_t>(lhs)];
                                  const Correspondence& b = correspondences[static_cast<std::size_t>(rhs)];
                                  if(a.age != b.age) {
                                      return a.age > b.age;  // longer-lived tracks first
                                  }
                                  return a.trackingError < b.trackingError;
                              });
            bucket.resize(static_cast<std::size_t>(perBucket));
        }
        selected.insert(selected.end(), bucket.begin(), bucket.end());
    }

    std::sort(selected.begin(), selected.end());
    return selected;
}

double medianParallax(const std::vector<Correspondence>& correspondences, const std::vector<int>& indices) {
    if(indices.empty()) {
        return 0.0;
    }
    std::vector<double> values;
    values.reserve(indices.size());
    for(const int idx : indices) {
        values.push_back(correspondences[static_cast<std::size_t>(idx)].parallax());
    }
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle), values.end());
    return values[middle];
}

}  // namespace oak_vio
