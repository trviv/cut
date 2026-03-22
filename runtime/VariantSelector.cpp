#include "VariantSelector.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

// ============================================================================
// Minimal recursive-descent JSON parser for tuning_data.json
// ============================================================================

struct JsonValue {
  enum Type { Null, Bool, Number, String, Array, Object };
  Type type = Null;
  double number = 0;
  bool boolean = false;
  std::string str;
  std::vector<JsonValue> array;
  std::vector<std::pair<std::string, JsonValue>> object;

  const JsonValue *get(const std::string &key) const {
    for (const auto &pair : object)
      if (pair.first == key)
        return &pair.second;
    return nullptr;
  }

  int64_t asInt() const { return static_cast<int64_t>(number); }
};

class JsonParser {
public:
  explicit JsonParser(const std::string &input) : input_(input), pos_(0) {}

  JsonValue parse() {
    skipWS();
    return parseValue();
  }

private:
  char peek() { return pos_ < input_.size() ? input_[pos_] : '\0'; }
  char advance() { return pos_ < input_.size() ? input_[pos_++] : '\0'; }

  void skipWS() {
    while (pos_ < input_.size() &&
           (input_[pos_] == ' ' || input_[pos_] == '\t' ||
            input_[pos_] == '\n' || input_[pos_] == '\r'))
      pos_++;
  }

  JsonValue parseValue() {
    skipWS();
    char c = peek();
    if (c == '"')
      return parseString();
    if (c == '{')
      return parseObject();
    if (c == '[')
      return parseArray();
    if (c == 't' || c == 'f' || c == 'n')
      return parseLiteral();
    if (c == '-' || (c >= '0' && c <= '9'))
      return parseNumber();
    throw std::runtime_error("Unexpected character in JSON");
  }

  JsonValue parseString() {
    JsonValue v;
    v.type = JsonValue::String;
    advance(); // skip opening "
    while (pos_ < input_.size() && input_[pos_] != '"') {
      if (input_[pos_] == '\\') {
        pos_++;
        if (pos_ >= input_.size())
          throw std::runtime_error("Unterminated string");
        char esc = input_[pos_++];
        switch (esc) {
        case '"':
          v.str += '"';
          break;
        case '\\':
          v.str += '\\';
          break;
        case '/':
          v.str += '/';
          break;
        case 'n':
          v.str += '\n';
          break;
        case 'r':
          v.str += '\r';
          break;
        case 't':
          v.str += '\t';
          break;
        default:
          v.str += esc;
          break;
        }
      } else {
        v.str += input_[pos_++];
      }
    }
    if (pos_ >= input_.size())
      throw std::runtime_error("Unterminated string");
    advance(); // skip closing "
    return v;
  }

  JsonValue parseNumber() {
    JsonValue v;
    v.type = JsonValue::Number;
    size_t start = pos_;
    if (peek() == '-')
      pos_++;
    while (pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9')
      pos_++;
    if (pos_ < input_.size() && input_[pos_] == '.') {
      pos_++;
      while (pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9')
        pos_++;
    }
    if (pos_ < input_.size() && (input_[pos_] == 'e' || input_[pos_] == 'E')) {
      pos_++;
      if (pos_ < input_.size() && (input_[pos_] == '+' || input_[pos_] == '-'))
        pos_++;
      while (pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9')
        pos_++;
    }
    v.number = std::stod(input_.substr(start, pos_ - start));
    return v;
  }

  JsonValue parseObject() {
    JsonValue v;
    v.type = JsonValue::Object;
    advance(); // skip {
    skipWS();
    if (peek() == '}') {
      advance();
      return v;
    }
    while (true) {
      skipWS();
      JsonValue key = parseString();
      skipWS();
      if (advance() != ':')
        throw std::runtime_error("Expected ':' in object");
      JsonValue val = parseValue();
      v.object.emplace_back(key.str, std::move(val));
      skipWS();
      if (peek() == '}') {
        advance();
        break;
      }
      if (advance() != ',')
        throw std::runtime_error("Expected ',' in object");
    }
    return v;
  }

  JsonValue parseArray() {
    JsonValue v;
    v.type = JsonValue::Array;
    advance(); // skip [
    skipWS();
    if (peek() == ']') {
      advance();
      return v;
    }
    while (true) {
      v.array.push_back(parseValue());
      skipWS();
      if (peek() == ']') {
        advance();
        break;
      }
      if (advance() != ',')
        throw std::runtime_error("Expected ',' in array");
    }
    return v;
  }

  JsonValue parseLiteral() {
    JsonValue v;
    if (input_.compare(pos_, 4, "true") == 0) {
      v.type = JsonValue::Bool;
      v.boolean = true;
      pos_ += 4;
    } else if (input_.compare(pos_, 5, "false") == 0) {
      v.type = JsonValue::Bool;
      v.boolean = false;
      pos_ += 5;
    } else if (input_.compare(pos_, 4, "null") == 0) {
      v.type = JsonValue::Null;
      pos_ += 4;
    } else {
      throw std::runtime_error("Unknown literal");
    }
    return v;
  }

  const std::string &input_;
  size_t pos_;
};

} // anonymous namespace

namespace cut {

VariantSelector &VariantSelector::instance() {
  static VariantSelector inst;
  return inst;
}

bool VariantSelector::loadTuningData() {
  const char *envPath = std::getenv("CUT_TUNING_DATA");
  if (envPath) {
    if (loadFromFile(envPath))
      return true;
    std::cerr << "VariantSelector: CUT_TUNING_DATA set to " << envPath
              << " but failed to load" << std::endl;
  }
  if (loadFromFile("tuning_data.json"))
    return true;
  const char *home = std::getenv("HOME");
  if (home) {
    std::string homePath = std::string(home) + "/.cut/tuning_data.json";
    if (loadFromFile(homePath))
      return true;
  }
  std::cerr << "VariantSelector: no tuning data found, using default variants"
            << std::endl;
  return false;
}

bool VariantSelector::loadFromFile(const std::string &path) {
  std::ifstream file(path);
  if (!file)
    return false;

  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());

  try {
    JsonParser parser(content);
    JsonValue root = parser.parse();

    const JsonValue *operatorsObj = root.get("operators");
    if (!operatorsObj || operatorsObj->type != JsonValue::Object)
      return false;

    operators_.clear();
    for (const auto &opPair : operatorsObj->object) {
      const std::string &opName = opPair.first;
      const JsonValue &opVal = opPair.second;

      OperatorRules opRules;
      opRules.defaultVariant = 0;

      const JsonValue *dv = opVal.get("default_variant");
      if (dv && dv->type == JsonValue::Number)
        opRules.defaultVariant = static_cast<int>(dv->asInt());

      const JsonValue *rulesArr = opVal.get("rules");
      if (rulesArr && rulesArr->type == JsonValue::Array) {
        for (const JsonValue &rv : rulesArr->array) {
          Rule rule;
          rule.variant = 0;
          rule.totalElementsMin = -1;
          rule.totalElementsMax = -1;

          const JsonValue *vi = rv.get("variant");
          if (vi && vi->type == JsonValue::Number)
            rule.variant = static_cast<int>(vi->asInt());

          const JsonValue *conds = rv.get("conditions");
          if (conds && conds->type == JsonValue::Object) {
            const JsonValue *mn = conds->get("total_elements_min");
            if (mn && mn->type == JsonValue::Number)
              rule.totalElementsMin = mn->asInt();
            const JsonValue *mx = conds->get("total_elements_max");
            if (mx && mx->type == JsonValue::Number)
              rule.totalElementsMax = mx->asInt();
          }

          opRules.rules.push_back(rule);
        }
      }

      operators_[opName] = std::move(opRules);
    }

    loaded_ = true;
    std::cerr << "Loaded tuning data from " << path << " (" << operators_.size()
              << " operators)" << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error parsing tuning data from " << path << ": " << e.what()
              << std::endl;
    return false;
  }
}

int VariantSelector::select(const std::string &operatorName,
                            const std::vector<uint32_t> &shape,
                            int defaultVariant) const {
  if (!loaded_)
    return defaultVariant;

  auto it = operators_.find(operatorName);
  if (it == operators_.end())
    return defaultVariant;

  const OperatorRules &opRules = it->second;

  int64_t totalElements = 1;
  for (uint32_t d : shape)
    totalElements *= d;

  for (const Rule &rule : opRules.rules) {
    bool minOk =
        (rule.totalElementsMin < 0) || (totalElements >= rule.totalElementsMin);
    bool maxOk =
        (rule.totalElementsMax < 0) || (totalElements < rule.totalElementsMax);
    if (minOk && maxOk)
      return rule.variant;
  }

  return opRules.defaultVariant;
}

} // namespace cut
