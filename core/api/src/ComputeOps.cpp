#include <ComputeOps.h>

namespace cut {

const char *operatorName(OperatorEnum op) {
  switch (op) {
  // Binary vec-vec arithmetic
  case BinaryVecVecAdd:
    return "BinaryVecVecAdd";
  case BinaryVecVecSub:
    return "BinaryVecVecSub";
  case BinaryVecVecMul:
    return "BinaryVecVecMul";
  case BinaryVecVecDiv:
    return "BinaryVecVecDiv";
  case BinaryVecVecMod:
    return "BinaryVecVecMod";
  case BinaryVecVecPow:
    return "BinaryVecVecPow";
  case BinaryVecVecFloorDiv:
    return "BinaryVecVecFloorDiv";

  // Binary vec-vec comparison
  case BinaryVecVecEqual:
    return "BinaryVecVecEqual";
  case BinaryVecVecNotEqual:
    return "BinaryVecVecNotEqual";
  case BinaryVecVecLess:
    return "BinaryVecVecLess";
  case BinaryVecVecLessEqual:
    return "BinaryVecVecLessEqual";
  case BinaryVecVecGreater:
    return "BinaryVecVecGreater";
  case BinaryVecVecGreaterEqual:
    return "BinaryVecVecGreaterEqual";

  // Binary vec-vec min/max
  case BinaryVecVecMin:
    return "BinaryVecVecMin";
  case BinaryVecVecMax:
    return "BinaryVecVecMax";

  // Binary vec-vec bitwise
  case BinaryVecVecBitwiseAnd:
    return "BinaryVecVecBitwiseAnd";
  case BinaryVecVecBitwiseOr:
    return "BinaryVecVecBitwiseOr";
  case BinaryVecVecBitwiseXor:
    return "BinaryVecVecBitwiseXor";
  case BinaryVecVecLeftShift:
    return "BinaryVecVecLeftShift";
  case BinaryVecVecRightShift:
    return "BinaryVecVecRightShift";

  // Binary vec-vec logical
  case BinaryVecVecLogicalAnd:
    return "BinaryVecVecLogicalAnd";
  case BinaryVecVecLogicalOr:
    return "BinaryVecVecLogicalOr";
  case BinaryVecVecLogicalXor:
    return "BinaryVecVecLogicalXor";

  // Binary vec-vec math
  case BinaryVecVecAtan2:
    return "BinaryVecVecAtan2";
  case BinaryVecVecHypot:
    return "BinaryVecVecHypot";
  case BinaryVecVecCopysign:
    return "BinaryVecVecCopysign";
  case BinaryVecVecFmod:
    return "BinaryVecVecFmod";
  case BinaryVecVecLogaddexp:
    return "BinaryVecVecLogaddexp";
  case BinaryVecVecLogaddexp2:
    return "BinaryVecVecLogaddexp2";

  // Binary vec-scalar arithmetic
  case BinaryVecScalarAdd:
    return "BinaryVecScalarAdd";
  case BinaryVecScalarSub:
    return "BinaryVecScalarSub";
  case BinaryVecScalarMul:
    return "BinaryVecScalarMul";
  case BinaryVecScalarDiv:
    return "BinaryVecScalarDiv";
  case BinaryVecScalarMod:
    return "BinaryVecScalarMod";
  case BinaryVecScalarPow:
    return "BinaryVecScalarPow";
  case BinaryVecScalarFloorDiv:
    return "BinaryVecScalarFloorDiv";

  // Binary vec-scalar comparison
  case BinaryVecScalarEqual:
    return "BinaryVecScalarEqual";
  case BinaryVecScalarNotEqual:
    return "BinaryVecScalarNotEqual";
  case BinaryVecScalarLess:
    return "BinaryVecScalarLess";
  case BinaryVecScalarLessEqual:
    return "BinaryVecScalarLessEqual";
  case BinaryVecScalarGreater:
    return "BinaryVecScalarGreater";
  case BinaryVecScalarGreaterEqual:
    return "BinaryVecScalarGreaterEqual";

  // Binary vec-scalar min/max
  case BinaryVecScalarMin:
    return "BinaryVecScalarMin";
  case BinaryVecScalarMax:
    return "BinaryVecScalarMax";

  // Binary vec-scalar bitwise
  case BinaryVecScalarBitwiseAnd:
    return "BinaryVecScalarBitwiseAnd";
  case BinaryVecScalarBitwiseOr:
    return "BinaryVecScalarBitwiseOr";
  case BinaryVecScalarBitwiseXor:
    return "BinaryVecScalarBitwiseXor";
  case BinaryVecScalarLeftShift:
    return "BinaryVecScalarLeftShift";
  case BinaryVecScalarRightShift:
    return "BinaryVecScalarRightShift";

  // Binary vec-scalar logical
  case BinaryVecScalarLogicalAnd:
    return "BinaryVecScalarLogicalAnd";
  case BinaryVecScalarLogicalOr:
    return "BinaryVecScalarLogicalOr";
  case BinaryVecScalarLogicalXor:
    return "BinaryVecScalarLogicalXor";

  // Binary vec-scalar math
  case BinaryVecScalarAtan2:
    return "BinaryVecScalarAtan2";
  case BinaryVecScalarHypot:
    return "BinaryVecScalarHypot";
  case BinaryVecScalarCopysign:
    return "BinaryVecScalarCopysign";
  case BinaryVecScalarFmod:
    return "BinaryVecScalarFmod";

  // Binary vec-scalar activation
  case BinaryVecScalarLeakyRelu:
    return "BinaryVecScalarLeakyRelu";
  case BinaryVecScalarPrelu:
    return "BinaryVecScalarPrelu";
  case BinaryVecScalarHardshrink:
    return "BinaryVecScalarHardshrink";

  // Unary operations
  case UnaryNeg:
    return "UnaryNeg";
  case UnaryAbs:
    return "UnaryAbs";
  case UnarySqrt:
    return "UnarySqrt";
  case UnarySquare:
    return "UnarySquare";
  case UnaryReciprocal:
    return "UnaryReciprocal";
  case UnarySign:
    return "UnarySign";
  case UnaryExp:
    return "UnaryExp";
  case UnaryExp2:
    return "UnaryExp2";
  case UnaryExpm1:
    return "UnaryExpm1";
  case UnaryLog:
    return "UnaryLog";
  case UnaryLog2:
    return "UnaryLog2";
  case UnaryLog10:
    return "UnaryLog10";
  case UnaryLog1p:
    return "UnaryLog1p";
  case UnarySin:
    return "UnarySin";
  case UnaryCos:
    return "UnaryCos";
  case UnaryTan:
    return "UnaryTan";
  case UnaryAsin:
    return "UnaryAsin";
  case UnaryAcos:
    return "UnaryAcos";
  case UnaryAtan:
    return "UnaryAtan";
  case UnarySinh:
    return "UnarySinh";
  case UnaryCosh:
    return "UnaryCosh";
  case UnaryTanh:
    return "UnaryTanh";
  case UnaryFloor:
    return "UnaryFloor";
  case UnaryCeil:
    return "UnaryCeil";
  case UnaryRound:
    return "UnaryRound";
  case UnaryCbrt:
    return "UnaryCbrt";
  case UnaryDegrees:
    return "UnaryDegrees";
  case UnaryRadians:
    return "UnaryRadians";
  case UnaryLogicalNot:
    return "UnaryLogicalNot";
  case UnaryBitwiseNot:
    return "UnaryBitwiseNot";
  case UnaryRelu:
    return "UnaryRelu";
  case UnarySigmoid:
    return "UnarySigmoid";
  case UnaryGelu:
    return "UnaryGelu";
  case UnarySilu:
    return "UnarySilu";
  case UnarySoftplus:
    return "UnarySoftplus";
  case UnaryIsNan:
    return "UnaryIsNan";
  case UnaryIsInf:
    return "UnaryIsInf";

  // Extended unary activations
  case UnaryRelu6:
    return "UnaryRelu6";
  case UnaryElu:
    return "UnaryElu";
  case UnarySelu:
    return "UnarySelu";
  case UnaryCelu:
    return "UnaryCelu";
  case UnaryMish:
    return "UnaryMish";
  case UnaryHardswish:
    return "UnaryHardswish";
  case UnaryHardsigmoid:
    return "UnaryHardsigmoid";
  case UnaryHardtanh:
    return "UnaryHardtanh";
  case UnarySoftsign:
    return "UnarySoftsign";
  case UnaryLogSigmoid:
    return "UnaryLogSigmoid";
  case UnaryTanhshrink:
    return "UnaryTanhshrink";

  // Extended unary math
  case UnaryRsqrt:
    return "UnaryRsqrt";
  case UnaryTrunc:
    return "UnaryTrunc";
  case UnaryFrac:
    return "UnaryFrac";
  case UnaryAsinh:
    return "UnaryAsinh";
  case UnaryAcosh:
    return "UnaryAcosh";
  case UnaryAtanh:
    return "UnaryAtanh";
  case UnaryIsFinite:
    return "UnaryIsFinite";

  // Extended binary vec-scalar
  case BinaryVecScalarSoftshrink:
    return "BinaryVecScalarSoftshrink";
  case BinaryVecScalarLogaddexp:
    return "BinaryVecScalarLogaddexp";
  case BinaryVecScalarLogaddexp2:
    return "BinaryVecScalarLogaddexp2";

  // Ternary operations
  case TernaryClamp:
    return "TernaryClamp";
  case TernarySelect:
    return "TernarySelect";

  // Reduction operations
  case ReduceSum:
    return "ReduceSum";
  case ReduceMean:
    return "ReduceMean";
  case ReduceMin:
    return "ReduceMin";
  case ReduceMax:
    return "ReduceMax";
  case ReduceProd:
    return "ReduceProd";
  case ReduceAny:
    return "ReduceAny";
  case ReduceAll:
    return "ReduceAll";
  case ReduceDimSum:
    return "ReduceDimSum";
  case ReduceDimMean:
    return "ReduceDimMean";
  case ReduceDimMin:
    return "ReduceDimMin";
  case ReduceDimMax:
    return "ReduceDimMax";
  case ReduceDimProd:
    return "ReduceDimProd";
  case ReduceDimAny:
    return "ReduceDimAny";
  case ReduceDimAll:
    return "ReduceDimAll";
  case ReduceArgmax:
    return "ReduceArgmax";
  case ReduceArgmin:
    return "ReduceArgmin";
  case ReduceDimArgmax:
    return "ReduceDimArgmax";
  case ReduceDimArgmin:
    return "ReduceDimArgmin";

  // Cumulative operations
  case CumSum:
    return "CumSum";
  case CumProd:
    return "CumProd";

  // Matrix operations
  case MatMul:
    return "MatMul";
  case Transpose:
    return "Transpose";
  case Dot:
    return "Dot";

  // Tensor manipulation operations
  case Concat:
    return "Concat";
  case Stack:
    return "Stack";
  case Flatten:
    return "Flatten";

  // Norm operations
  case Norm:
    return "Norm";
  case NormDim:
    return "NormDim";

  // Tensor creation operations
  case Arange:
    return "Arange";
  case Linspace:
    return "Linspace";
  case Zeros:
    return "Zeros";
  case Ones:
    return "Ones";
  case Full:
    return "Full";
  case Copy:
    return "Copy";

  // Prefix scan operations
  case PrefixScanExclusiveSum:
    return "PrefixScanExclusiveSum";
  case PrefixScanInclusiveSum:
    return "PrefixScanInclusiveSum";

  // Sort operations
  case SortBitonic:
    return "SortBitonic";
  case SortRadix:
    return "SortRadix";

  // Dispatcher internal shader templates
  case InternalPartialReduce:
    return "InternalPartialReduce";
  case InternalFinalReduce:
    return "InternalFinalReduce";
  case InternalScanPerWg:
    return "InternalScanPerWg";
  case InternalScanPartialSums:
    return "InternalScanPartialSums";
  case InternalScanPropagate:
    return "InternalScanPropagate";
  case InternalBitonicStep:
    return "InternalBitonicStep";
  case InternalBitonicPadInit:
    return "InternalBitonicPadInit";
  case InternalBitonicCopyBack:
    return "InternalBitonicCopyBack";
  case InternalRadixHistogram:
    return "InternalRadixHistogram";
  case InternalRadixScatter:
    return "InternalRadixScatter";
  case InternalFillUint:
    return "InternalFillUint";
  case InternalScanUint:
    return "InternalScanUint";

  // Convolution operations
  case Conv1D:
    return "Conv1D";
  case Conv2D:
    return "Conv2D";
  case ConvTranspose2D:
    return "ConvTranspose2D";

  default:
    return "Unknown";
  }
}

} // namespace cut
