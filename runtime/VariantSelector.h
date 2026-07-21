#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cut {

/// Loads tuning data from a JSON file and selects optimal shader variants
/// based on shape dimensions at runtime.
///
/// The tuning_data.json file is produced by:
///   1. Running the autotune benchmark binary
///   2. Processing results with scripts/bench/derive_rules.py
///
/// Rules are total_elements threshold-based and evaluated in order (first
/// match wins).
class VariantSelector {
public:
  /// Returns the global singleton instance.
  static VariantSelector &instance();

  /// Loads tuning data by searching standard locations:
  ///   1. CUT_TUNING_DATA env var
  ///   2. ./tuning_data.json
  ///   3. ~/.cut/tuning_data.json
  /// Returns true if loaded successfully.
  bool loadTuningData();

  /// Loads tuning data from a specific file path.
  bool loadFromFile(const std::string &path);

  /// Select the best variant for an operator given shape dimensions.
  /// Returns defaultVariant if no tuning data exists for this operator.
  int select(const std::string &operatorName,
             const std::vector<uint32_t> &shape,
             int defaultVariant) const;

  /// Returns true if tuning data has been loaded.
  bool isLoaded() const { return loaded_; }

private:
  VariantSelector() = default;

  struct Rule {
    int variant;
    int64_t totalElementsMin; // -1 = no constraint
    int64_t totalElementsMax; // -1 = no constraint
  };

  struct OperatorRules {
    int defaultVariant;
    std::vector<Rule> rules;
  };

  bool loaded_ = false;
  std::unordered_map<std::string, OperatorRules> operators_;
};

} // namespace cut
