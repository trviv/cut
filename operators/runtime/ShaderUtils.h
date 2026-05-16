#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cut {

/**
 * Links a function from one SPIR-V module into another, replacing a
 * stub function in the template.
 *
 * The template module's function (identified by templateFuncName) has its
 * body stripped and marked as an Import. The module's function (identified
 * by moduleFuncName) is marked as an Export. The two modules are then
 * linked via SPIRV-Tools so the template calls the module's implementation.
 *
 * @param templateSpirv    SPIR-V of the template (contains the stub).
 * @param templateFuncName Name of the function to replace in the template.
 * @param moduleSpirv      SPIR-V containing the replacement function.
 * @param moduleFuncName   Name of the replacement function in the module.
 * @return Linked SPIR-V bytecode ready for pipeline creation.
 * @throws std::runtime_error if patching or linking fails.
 */
std::vector<uint32_t>
compileCustomShader(const std::vector<uint32_t> &templateSpirv,
                    const std::string &templateFuncName,
                    const std::vector<uint32_t> &moduleSpirv,
                    const std::string &moduleFuncName);

} // namespace cut
