#include <ComputeOps.h>

namespace cut {

const char *operatorName(OperatorEnum op) {
  switch (op) {
  // Binary arithmetic
  case BinaryAdd:
    return "BinaryAdd";
  case BinarySub:
    return "BinarySub";
  case BinaryMul:
    return "BinaryMul";
  case BinaryDiv:
    return "BinaryDiv";
  case BinaryMod:
    return "BinaryMod";
  case BinaryPow:
    return "BinaryPow";
  case BinaryFloorDiv:
    return "BinaryFloorDiv";

  // Binary comparison
  case BinaryEqual:
    return "BinaryEqual";
  case BinaryNotEqual:
    return "BinaryNotEqual";
  case BinaryLess:
    return "BinaryLess";
  case BinaryLessEqual:
    return "BinaryLessEqual";
  case BinaryGreater:
    return "BinaryGreater";
  case BinaryGreaterEqual:
    return "BinaryGreaterEqual";

  // Binary min/max
  case BinaryMin:
    return "BinaryMin";
  case BinaryMax:
    return "BinaryMax";

  // Binary bitwise
  case BinaryBitwiseAnd:
    return "BinaryBitwiseAnd";
  case BinaryBitwiseOr:
    return "BinaryBitwiseOr";
  case BinaryBitwiseXor:
    return "BinaryBitwiseXor";
  case BinaryLeftShift:
    return "BinaryLeftShift";
  case BinaryRightShift:
    return "BinaryRightShift";

  // Binary logical
  case BinaryLogicalAnd:
    return "BinaryLogicalAnd";
  case BinaryLogicalOr:
    return "BinaryLogicalOr";
  case BinaryLogicalXor:
    return "BinaryLogicalXor";

  // Binary math
  case BinaryAtan2:
    return "BinaryAtan2";
  case BinaryHypot:
    return "BinaryHypot";
  case BinaryCopysign:
    return "BinaryCopysign";
  case BinaryFmod:
    return "BinaryFmod";
  case BinaryLogaddexp:
    return "BinaryLogaddexp";
  case BinaryLogaddexp2:
    return "BinaryLogaddexp2";

  // Binary activation
  case BinaryLeakyRelu:
    return "BinaryLeakyRelu";
  case BinaryPrelu:
    return "BinaryPrelu";
  case BinaryHardshrink:
    return "BinaryHardshrink";
  case BinarySoftshrink:
    return "BinarySoftshrink";

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
  case ReduceArgmax:
    return "ReduceArgmax";
  case ReduceArgmin:
    return "ReduceArgmin";

  // Cumulative operations
  case CumSum:
    return "CumSum";
  case CumProd:
    return "CumProd";

  // Matrix operations
  case MatMul:
    return "MatMul";
  case MatMulSiLU:
    return "MatMulSiLU";
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
  case Cast:
    return "Cast";

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

  // Pooling operations
  case MaxPool2D:
    return "MaxPool2D";
  case AvgPool2D:
    return "AvgPool2D";

  // Normalization operations
  case LayerNorm:
    return "LayerNorm";
  case BatchNorm:
    return "BatchNorm";
  case RMSNorm:
    return "RMSNorm";
  case ExtendedRMSNorm:
    return "ExtendedRMSNorm";

  // Embedding operations
  case Embedding:
    return "Embedding";

  // Padding operations
  case Pad:
    return "Pad";

  // Expand operations
  case Expand:
    return "Expand";

  // RoPE operations
  case RoPE:
    return "RoPE";

  // Attention operations
  case CacheWrite:
    return "CacheWrite";
  case Attention:
    return "Attention";

  // Quantized matmul
  case MatMulQ8:
    return "MatMulQ8";

  // Fused binary
  case FusedBinary:
    return "FusedBinary";

  default:
    return "Unknown";
  }
}

} // namespace cut
