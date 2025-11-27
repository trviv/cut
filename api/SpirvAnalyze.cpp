#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstring>
#include <iomanip>
#include <algorithm>

// SPIR-V Constants
constexpr uint32_t SPIRV_MAGIC = 0x07230203;

enum class SpvOpcode : uint16_t {
    OpTypePointer = 32,
    OpTypeStruct = 30,
    OpTypeImage = 25,
    OpTypeSampledImage = 27,
    OpTypeArray = 28,
    OpTypeVector = 23,
    OpTypeMatrix = 24,
    OpTypeFloat = 22,
    OpTypeInt = 21,
    OpVariable = 59,
    OpDecorate = 71,
    OpMemberDecorate = 72,
    OpName = 5,
    OpMemberName = 6
};

enum class SpvStorageClass : uint32_t {
    UniformConstant = 0,
    Input = 1,
    Uniform = 2,
    Output = 3,
    Workgroup = 4,
    CrossWorkgroup = 5,
    Private = 6,
    Function = 7,
    Generic = 8,
    PushConstant = 9,
    AtomicCounter = 10,
    Image = 11,
    StorageBuffer = 12
};

enum class SpvDecoration : uint32_t {
    RelaxedPrecision = 0,
    SpecId = 1,
    Block = 2,
    BufferBlock = 3,
    RowMajor = 4,
    ColMajor = 5,
    ArrayStride = 6,
    MatrixStride = 7,
    GLSLShared = 8,
    GLSLPacked = 9,
    CPacked = 10,
    BuiltIn = 11,
    NoPerspective = 13,
    Flat = 14,
    Patch = 15,
    Centroid = 16,
    Sample = 17,
    Invariant = 18,
    Restrict = 19,
    Aliased = 20,
    Volatile = 21,
    Constant = 22,
    Coherent = 23,
    NonWritable = 24,
    NonReadable = 25,
    Uniform = 26,
    UniformId = 27,
    SaturatedConversion = 28,
    Stream = 29,
    Location = 30,
    Component = 31,
    Index = 32,
    Binding = 33,
    DescriptorSet = 34,
    Offset = 35,
    XfbBuffer = 36,
    XfbStride = 37,
    FuncParamAttr = 38,
    FPRoundingMode = 39,
    FPFastMathMode = 40,
    LinkageAttributes = 41,
    NoContraction = 42,
    InputAttachmentIndex = 43,
    Alignment = 44
};

struct Decoration {
    SpvDecoration decoration;
    std::vector<uint32_t> params;
    
    Decoration(SpvDecoration dec, const std::vector<uint32_t>& p = {}) 
        : decoration(dec), params(p) {}
};

struct TypeInfo {
    std::string name;
    uint32_t size = 0;
    std::vector<uint32_t> member_types;
    
    TypeInfo(const std::string& n = "unknown") : name(n) {}
};

struct ResourceInfo {
    uint32_t id = 0;
    std::string name;
    std::string type;
    SpvStorageClass storage_class = SpvStorageClass::Private;
    std::optional<uint32_t> binding;
    std::optional<uint32_t> descriptor_set;
    std::optional<uint32_t> location;
    std::optional<uint32_t> offset;
    std::vector<Decoration> decorations;
    
    ResourceInfo() = default;
};

class SPIRVAnalyzer {
private:
    std::vector<std::vector<uint32_t>> instructions;
    std::unordered_map<uint32_t, std::string> names;
    std::unordered_map<uint32_t, std::vector<Decoration>> decorations;
    std::unordered_map<uint32_t, TypeInfo> types;
    std::unordered_map<uint32_t, ResourceInfo> variables;
    
    std::string readString(const std::vector<uint32_t>& instruction, size_t startIndex) {
        std::string result;
        
        for (size_t i = startIndex; i < instruction.size(); ++i) {
            uint32_t word = instruction[i];
            for (int j = 0; j < 4; ++j) {
                char ch = static_cast<char>((word >> (j * 8)) & 0xFF);
                if (ch == '\0') {
                    return result;
                }
                result += ch;
            }
        }
        return result;
    }
    
    std::string storageClassToString(SpvStorageClass sc) {
        switch (sc) {
            case SpvStorageClass::UniformConstant: return "UniformConstant";
            case SpvStorageClass::Input: return "Input";
            case SpvStorageClass::Uniform: return "Uniform";
            case SpvStorageClass::Output: return "Output";
            case SpvStorageClass::Workgroup: return "Workgroup";
            case SpvStorageClass::CrossWorkgroup: return "CrossWorkgroup";
            case SpvStorageClass::Private: return "Private";
            case SpvStorageClass::Function: return "Function";
            case SpvStorageClass::Generic: return "Generic";
            case SpvStorageClass::PushConstant: return "PushConstant";
            case SpvStorageClass::AtomicCounter: return "AtomicCounter";
            case SpvStorageClass::Image: return "Image";
            case SpvStorageClass::StorageBuffer: return "StorageBuffer";
            default: return "Unknown";
        }
    }
    
    std::string decorationToString(SpvDecoration dec) {
        switch (dec) {
            case SpvDecoration::Block: return "Block";
            case SpvDecoration::BufferBlock: return "BufferBlock";
            case SpvDecoration::RowMajor: return "RowMajor";
            case SpvDecoration::ColMajor: return "ColMajor";
            case SpvDecoration::ArrayStride: return "ArrayStride";
            case SpvDecoration::MatrixStride: return "MatrixStride";
            case SpvDecoration::Binding: return "Binding";
            case SpvDecoration::DescriptorSet: return "DescriptorSet";
            case SpvDecoration::Location: return "Location";
            case SpvDecoration::Offset: return "Offset";
            case SpvDecoration::NonReadable: return "NonReadable";
            case SpvDecoration::NonWritable: return "NonWritable";
            case SpvDecoration::Restrict: return "Restrict";
            case SpvDecoration::Volatile: return "Volatile";
            case SpvDecoration::Coherent: return "Coherent";
            default: return "Decoration" + std::to_string(static_cast<uint32_t>(dec));
        }
    }

public:
    bool readSpirvBinary(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filepath << std::endl;
            return false;
        }
        
        // Read entire file
        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        
        if (fileSize < 20) {
            std::cerr << "Error: File too small to be valid SPIR-V" << std::endl;
            return false;
        }
        
        std::vector<uint8_t> data(fileSize);
        file.read(reinterpret_cast<char*>(data.data()), fileSize);
        file.close();
        
        // Check magic number
        uint32_t magic = *reinterpret_cast<const uint32_t*>(data.data());
        if (magic != SPIRV_MAGIC) {
            std::cerr << "Error: Invalid SPIR-V magic number: 0x" 
                      << std::hex << magic << std::endl;
            return false;
        }
        
        // Read header
        uint32_t version = *reinterpret_cast<const uint32_t*>(data.data() + 4);
        uint32_t generator = *reinterpret_cast<const uint32_t*>(data.data() + 8);
        uint32_t bound = *reinterpret_cast<const uint32_t*>(data.data() + 12);
        uint32_t schema = *reinterpret_cast<const uint32_t*>(data.data() + 16);
        
        std::cout << "SPIR-V Version: " << (version >> 16) << "." 
                  << ((version >> 8) & 0xFF) << "." << (version & 0xFF) << std::endl;
        std::cout << "Generator: 0x" << std::hex << generator << std::dec << std::endl;
        std::cout << "ID Bound: " << bound << std::endl;
        std::cout << "Schema: " << schema << std::endl;
        
        // Parse instructions
        size_t offset = 20;
        while (offset < fileSize) {
            if (offset + 4 > fileSize) break;
            
            uint32_t firstWord = *reinterpret_cast<const uint32_t*>(data.data() + offset);
            uint16_t opcode = firstWord & 0xFFFF;
            uint16_t length = firstWord >> 16;
            
            if (length == 0 || offset + length * 4 > fileSize) break;
            
            std::vector<uint32_t> instruction;
            for (int i = 0; i < length; ++i) {
                uint32_t word = *reinterpret_cast<const uint32_t*>(data.data() + offset + i * 4);
                instruction.push_back(word);
            }
            
            instructions.push_back(instruction);
            offset += length * 4;
        }
        
        return true;
    }
    
    void analyzeInstructions() {
        // First pass: collect names, types, and decorations
        for (const auto& instruction : instructions) {
            if (instruction.empty()) continue;
            
            SpvOpcode opcode = static_cast<SpvOpcode>(instruction[0] & 0xFFFF);
            
            switch (opcode) {
                case SpvOpcode::OpName:
                    if (instruction.size() >= 3) {
                        uint32_t targetId = instruction[1];
                        std::string name = readString(instruction, 2);
                        names[targetId] = name;
                    }
                    break;
                    
                case SpvOpcode::OpDecorate:
                    if (instruction.size() >= 3) {
                        uint32_t targetId = instruction[1];
                        SpvDecoration decoration = static_cast<SpvDecoration>(instruction[2]);
                        std::vector<uint32_t> params(instruction.begin() + 3, instruction.end());
                        decorations[targetId].emplace_back(decoration, params);
                    }
                    break;
                    
                case SpvOpcode::OpTypePointer:
                    if (instruction.size() >= 4) {
                        uint32_t resultId = instruction[1];
                        uint32_t storageClass = instruction[2];
                        uint32_t typeId = instruction[3];
                        types[resultId] = TypeInfo("Pointer<" + storageClassToString(static_cast<SpvStorageClass>(storageClass)) + ">");
                    }
                    break;
                    
                case SpvOpcode::OpTypeStruct:
                    if (instruction.size() >= 2) {
                        uint32_t resultId = instruction[1];
                        TypeInfo typeInfo("Struct(" + std::to_string(instruction.size() - 2) + " members)");
                        typeInfo.member_types.assign(instruction.begin() + 2, instruction.end());
                        types[resultId] = typeInfo;
                    }
                    break;
                    
                case SpvOpcode::OpTypeImage:
                    if (instruction.size() >= 9) {
                        uint32_t resultId = instruction[1];
                        types[resultId] = TypeInfo("Image");
                    }
                    break;
                    
                case SpvOpcode::OpTypeSampledImage:
                    if (instruction.size() >= 3) {
                        uint32_t resultId = instruction[1];
                        types[resultId] = TypeInfo("SampledImage");
                    }
                    break;
                    
                case SpvOpcode::OpTypeVector:
                    if (instruction.size() >= 4) {
                        uint32_t resultId = instruction[1];
                        uint32_t componentCount = instruction[3];
                        types[resultId] = TypeInfo("Vector" + std::to_string(componentCount));
                    }
                    break;
                    
                case SpvOpcode::OpTypeMatrix:
                    if (instruction.size() >= 4) {
                        uint32_t resultId = instruction[1];
                        uint32_t columnCount = instruction[3];
                        types[resultId] = TypeInfo("Matrix" + std::to_string(columnCount));
                    }
                    break;
                    
                case SpvOpcode::OpTypeFloat:
                    if (instruction.size() >= 3) {
                        uint32_t resultId = instruction[1];
                        uint32_t width = instruction[2];
                        types[resultId] = TypeInfo("float" + std::to_string(width));
                    }
                    break;
                    
                case SpvOpcode::OpTypeInt:
                    if (instruction.size() >= 4) {
                        uint32_t resultId = instruction[1];
                        uint32_t width = instruction[2];
                        uint32_t signedness = instruction[3];
                        types[resultId] = TypeInfo((signedness ? "int" : "uint") + std::to_string(width));
                    }
                    break;
                    
                default:
                    break;
            }
        }
        
        // Second pass: collect variables
        for (const auto& instruction : instructions) {
            if (instruction.empty()) continue;
            
            SpvOpcode opcode = static_cast<SpvOpcode>(instruction[0] & 0xFFFF);
            
            if (opcode == SpvOpcode::OpVariable && instruction.size() >= 4) {
                uint32_t resultType = instruction[1];
                uint32_t resultId = instruction[2];
                uint32_t storageClass = instruction[3];
                
                ResourceInfo resource;
                resource.id = resultId;
                resource.name = names.count(resultId) ? names[resultId] : ("unnamed_" + std::to_string(resultId));
                resource.type = types.count(resultType) ? types[resultType].name : "unknown";
                resource.storage_class = static_cast<SpvStorageClass>(storageClass);
                
                // Apply decorations
                if (decorations.count(resultId)) {
                    resource.decorations = decorations[resultId];
                    for (const auto& decoration : decorations[resultId]) {
                        switch (decoration.decoration) {
                            case SpvDecoration::Binding:
                                if (!decoration.params.empty()) {
                                    resource.binding = decoration.params[0];
                                }
                                break;
                            case SpvDecoration::DescriptorSet:
                                if (!decoration.params.empty()) {
                                    resource.descriptor_set = decoration.params[0];
                                }
                                break;
                            case SpvDecoration::Location:
                                if (!decoration.params.empty()) {
                                    resource.location = decoration.params[0];
                                }
                                break;
                            case SpvDecoration::Offset:
                                if (!decoration.params.empty()) {
                                    resource.offset = decoration.params[0];
                                }
                                break;
                            default:
                                break;
                        }
                    }
                }
                
                variables[resultId] = resource;
            }
        }
    }
    
    void printResources() {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "SHADER RESOURCES" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // Group resources by type
        std::vector<ResourceInfo> uniformBuffers;
        std::vector<ResourceInfo> storageBuffers;
        std::vector<ResourceInfo> textures;
        std::vector<ResourceInfo> samplers;
        std::vector<ResourceInfo> inputs;
        std::vector<ResourceInfo> outputs;
        std::vector<ResourceInfo> pushConstants;
        
        for (const auto& [id, resource] : variables) {
            switch (resource.storage_class) {
                case SpvStorageClass::Uniform:
                    uniformBuffers.push_back(resource);
                    break;
                case SpvStorageClass::StorageBuffer:
                    storageBuffers.push_back(resource);
                    break;
                case SpvStorageClass::UniformConstant:
                    if (resource.type.find("Image") != std::string::npos || 
                        resource.type.find("Sampled") != std::string::npos) {
                        textures.push_back(resource);
                    } else {
                        samplers.push_back(resource);
                    }
                    break;
                case SpvStorageClass::Input:
                    inputs.push_back(resource);
                    break;
                case SpvStorageClass::Output:
                    outputs.push_back(resource);
                    break;
                case SpvStorageClass::PushConstant:
                    pushConstants.push_back(resource);
                    break;
                default:
                    break;
            }
        }
        
        auto printResourceGroup = [this](const std::string& title, std::vector<ResourceInfo>& resources) {
            if (resources.empty()) return;
            
            std::cout << "\n" << title << ":" << std::endl;
            std::cout << std::string(title.length() + 1, '-') << std::endl;
            
            // Sort by descriptor set and binding
            std::sort(resources.begin(), resources.end(), [](const ResourceInfo& a, const ResourceInfo& b) {
                if (a.descriptor_set.has_value() != b.descriptor_set.has_value()) {
                    return a.descriptor_set.has_value();
                }
                if (a.descriptor_set.has_value() && a.descriptor_set != b.descriptor_set) {
                    return a.descriptor_set.value() < b.descriptor_set.value();
                }
                if (a.binding.has_value() != b.binding.has_value()) {
                    return a.binding.has_value();
                }
                if (a.binding.has_value() && a.binding != b.binding) {
                    return a.binding.value() < b.binding.value();
                }
                return a.name < b.name;
            });
            
            for (const auto& resource : resources) {
                std::cout << "  " << resource.name << " : " << resource.type << std::endl;
                
                std::string bindingInfo;
                if (resource.binding.has_value()) {
                    bindingInfo += "binding=" + std::to_string(resource.binding.value());
                }
                if (resource.descriptor_set.has_value()) {
                    if (!bindingInfo.empty()) bindingInfo += ", ";
                    bindingInfo += "set=" + std::to_string(resource.descriptor_set.value());
                }
                if (resource.location.has_value()) {
                    if (!bindingInfo.empty()) bindingInfo += ", ";
                    bindingInfo += "location=" + std::to_string(resource.location.value());
                }
                
                if (!bindingInfo.empty()) {
                    std::cout << "    [" << bindingInfo << "]" << std::endl;
                }
                
                // Print other decorations
                std::vector<std::string> otherDecorations;
                for (const auto& decoration : resource.decorations) {
                    if (decoration.decoration != SpvDecoration::Binding &&
                        decoration.decoration != SpvDecoration::DescriptorSet &&
                        decoration.decoration != SpvDecoration::Location) {
                        
                        std::string decorationStr = decorationToString(decoration.decoration);
                        if (!decoration.params.empty()) {
                            decorationStr += "(";
                            for (size_t i = 0; i < decoration.params.size(); ++i) {
                                if (i > 0) decorationStr += ", ";
                                decorationStr += std::to_string(decoration.params[i]);
                            }
                            decorationStr += ")";
                        }
                        otherDecorations.push_back(decorationStr);
                    }
                }
                
                if (!otherDecorations.empty()) {
                    std::cout << "    Decorations: ";
                    for (size_t i = 0; i < otherDecorations.size(); ++i) {
                        if (i > 0) std::cout << ", ";
                        std::cout << otherDecorations[i];
                    }
                    std::cout << std::endl;
                }
            }
        };
        
        printResourceGroup("Uniform Buffers", uniformBuffers);
        printResourceGroup("Storage Buffers", storageBuffers);
        printResourceGroup("Textures/Images", textures);
        printResourceGroup("Samplers", samplers);
        printResourceGroup("Input Variables", inputs);
        printResourceGroup("Output Variables", outputs);
        printResourceGroup("Push Constants", pushConstants);
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <spirv_file>" << std::endl;
        return 1;
    }
    
    std::string spirvFile = argv[1];
    SPIRVAnalyzer analyzer;
    
    if (!analyzer.readSpirvBinary(spirvFile)) {
        return 1;
    }
    
    analyzer.analyzeInstructions();
    analyzer.printResources();
    
    return 0;
}