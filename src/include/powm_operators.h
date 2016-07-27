#define AR_GEOMETRY 128
#define AR_DIGIT 8

typedef void (*powm_kernel)(powm_arguments_t powm_arguments, int32_t start, int32_t count);

template<class T>
void determineMaxBlocks(T *kernel, int32_t threads, int32_t *blocks_per_sm) {
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks_per_sm, kernel, threads, 0);
}

uint32_t windowBitsForPrecision(uint32_t precision) {
  uint32_t windowBits;

  // these are not tuned
  if(precision<400)
    windowBits=4;
  else if(precision<800)
    windowBits=5;
  else if(precision<1600)
    windowBits=6;
  else if(precision<=4096)
    windowBits=7;
  else
    windowBits=8;

  return windowBits;
}

template<int32_t geometry, int32_t min_blocks, int32_t words, int32_t kar_mult, int32_t kar_sqr>
xmpError_t XMPAPI internalPowmRegMP(xmpHandle_t handle, xmpIntegers_t out, const xmpIntegers_t a, const xmpIntegers_t exp, const xmpIntegers_t mod, uint32_t start, uint32_t count, uint32_t *instances_per_block, uint32_t *blocks_per_sm) {
  // geometry - # of threads per block
  // min_blocks - used for launch bounds
  // words is the size of the modulus in words
  // kar_mult and kar_sqr are the levels of Karatsuba and probably should be 0

  xmpExecutionPolicy_t policy=handle->policy;
  int32_t              bits, windowBits;
  size_t               windowBytes;
  int32_t              precision=out->precision;
  ar_arguments_t       ar_arguments;
  powm_arguments_t     powm_arguments;
  xmpError_t           error;

  XMP_SET_DEVICE(handle);

  a->requireFormat(handle, xmpFormatStrided);
  mod->requireFormat(handle, xmpFormatStrided);
  exp->requireFormat(handle, xmpFormatStrided);

  bits=exp->precision;
  windowBits=windowBitsForPrecision(bits);

  windowBytes=((1<<windowBits)+4) * words * 4 * ROUND_UP(count, geometry);

  error=xmpSetNecessaryScratchSize(handle, windowBytes);
  if(error!=xmpErrorSuccess)
    return error;

  if(instances_per_block!=NULL)
    *instances_per_block=geometry;

  if(blocks_per_sm!=NULL) {
    determineMaxBlocks(regmp_powm_kernel<geometry, min_blocks, false, words, kar_mult, kar_sqr>, geometry, (int32_t *)blocks_per_sm);
    XMP_CHECK_CUDA();
  }

  if(instances_per_block!=NULL || blocks_per_sm!=NULL)
    return xmpErrorSuccess;

  ar_arguments.window_data=(xmpLimb_t *)handle->scratch;
  ar_arguments.window_bits=windowBits;
  ar_arguments.a_data=a->slimbs;
  ar_arguments.a_len=DIV_ROUND_UP(a->precision, 32);
  ar_arguments.a_stride=a->stride;
  ar_arguments.a_count=a->count;
  ar_arguments.mod_data=mod->slimbs;
  ar_arguments.mod_len=DIV_ROUND_UP(mod->precision, 32);
  ar_arguments.mod_stride=mod->stride;
  ar_arguments.mod_count=mod->count;
  ar_arguments.a_indices=policy->indices[1];
  ar_arguments.mod_indices=policy->indices[3];
  ar_arguments.a_indices_count=policy->indices_count[1];
  ar_arguments.mod_indices_count=policy->indices_count[3];

  {
    dim3 blocks(DIV_ROUND_UP(count, AR_GEOMETRY)), threads(AR_GEOMETRY);

    regmp_ar_kernel<words><<<blocks, threads, 0, handle->stream>>>(ar_arguments, start, count);
  }

  powm_arguments.out_data=out->slimbs;
  powm_arguments.out_len=DIV_ROUND_UP(precision, 32);
  powm_arguments.out_stride=out->stride;
  powm_arguments.exp_data=exp->slimbs;
  powm_arguments.exp_stride=exp->stride;
  powm_arguments.exp_count=exp->count;
  powm_arguments.mod_count=0;
  powm_arguments.window_data=(xmpLimb_t *)handle->scratch;
  powm_arguments.bits=exp->precision;
  powm_arguments.window_bits=windowBits;
  powm_arguments.out_indices=policy->indices[0];
  powm_arguments.exp_indices=policy->indices[2];
  powm_arguments.exp_indices_count=policy->indices_count[2];

  {
    dim3 blocks(DIV_ROUND_UP(count, geometry)), threads(geometry);

    regmp_powm_kernel<geometry, min_blocks, false, words, kar_mult, kar_sqr><<<blocks, threads, 0, handle->stream>>>(powm_arguments, start, count);
  }

  out->setFormat(xmpFormatStrided);

  XMP_CHECK_CUDA();

  return xmpErrorSuccess;
}

template<int32_t geometry, int32_t min_blocks, int32_t size>
xmpError_t XMPAPI internalPowmDigitMP(xmpHandle_t handle, xmpIntegers_t out, const xmpIntegers_t a, const xmpIntegers_t exp, const xmpIntegers_t mod, uint32_t start, uint32_t count, uint32_t *instances_per_block, uint32_t *blocks_per_sm) {
  // geometry - # of threads per block
  // min_blocks - used for launch bounds
  // size is the size of the digit in words

  xmpExecutionPolicy_t policy=handle->policy;
  int32_t              bits, windowBits;
  size_t               windowBytes;
  int32_t              precision=out->precision;
  ar_arguments_t       ar_arguments;
  powm_arguments_t     powm_arguments;
  xmpError_t           error;

  XMP_SET_DEVICE(handle);

  a->requireFormat(handle, xmpFormatStrided);
  mod->requireFormat(handle, xmpFormatStrided);
  exp->requireFormat(handle, xmpFormatStrided);

  bits=exp->precision;
  windowBits=windowBitsForPrecision(bits);

  windowBytes=((1<<windowBits)+4) * ROUND_UP(precision, DIGIT*32)/8 * ROUND_UP(count, geometry);

  error=xmpSetNecessaryScratchSize(handle, windowBytes);
  if(error!=xmpErrorSuccess)
    return error;

  if(instances_per_block!=NULL)
    *instances_per_block=geometry;

  if(blocks_per_sm!=NULL) {
    determineMaxBlocks(digitmp_powm_kernel<geometry, min_blocks, false, size>, geometry, (int32_t *)blocks_per_sm);
    XMP_CHECK_CUDA();
  }

  if(instances_per_block!=NULL || blocks_per_sm!=NULL)
    return xmpErrorSuccess;

  ar_arguments.window_data=(xmpLimb_t *)handle->scratch;
  ar_arguments.window_bits=windowBits;
  ar_arguments.a_data=a->slimbs;
  ar_arguments.a_len=DIV_ROUND_UP(a->precision, 32);
  ar_arguments.a_stride=a->stride;
  ar_arguments.a_count=a->count;
  ar_arguments.mod_data=mod->slimbs;
  ar_arguments.mod_len=DIV_ROUND_UP(mod->precision, 32);
  ar_arguments.mod_stride=mod->stride;
  ar_arguments.mod_count=mod->count;
  ar_arguments.a_indices=policy->indices[1];
  ar_arguments.mod_indices=policy->indices[3];
  ar_arguments.a_indices_count=policy->indices_count[1];
  ar_arguments.mod_indices_count=policy->indices_count[3];

  {
    dim3 blocks(DIV_ROUND_UP(count, AR_GEOMETRY)), threads(AR_GEOMETRY);

    digitmp_ar_kernel<size><<<blocks, threads, 0, handle->stream>>>(ar_arguments, start, count);
  }

  powm_arguments.out_data=out->slimbs;
  powm_arguments.out_len=DIV_ROUND_UP(precision, 32);
  powm_arguments.out_stride=out->stride;
  powm_arguments.exp_data=exp->slimbs;
  powm_arguments.exp_stride=exp->stride;
  powm_arguments.exp_count=exp->count;
  powm_arguments.mod_count=0;
  powm_arguments.window_data=(xmpLimb_t *)handle->scratch;
  powm_arguments.digits=DIV_ROUND_UP(precision, DIGIT*32);
  powm_arguments.bits=exp->precision;
  powm_arguments.window_bits=windowBits;
  powm_arguments.out_indices=policy->indices[0];
  powm_arguments.exp_indices=policy->indices[2];
  powm_arguments.exp_indices_count=policy->indices_count[2];

  {
    dim3 blocks(DIV_ROUND_UP(count, geometry)), threads(geometry);

    digitmp_powm_kernel<geometry, min_blocks, false, size><<<blocks, threads, 0, handle->stream>>>(powm_arguments, start, count);
  }

  out->setFormat(xmpFormatStrided);

  XMP_CHECK_CUDA();

  return xmpErrorSuccess;
}

template<int32_t geometry, int32_t min_blocks, int32_t width, int32_t words>
xmpError_t XMPAPI internalPowmWarpDistributedMP(xmpHandle_t handle, xmpIntegers_t out, const xmpIntegers_t a, const xmpIntegers_t exp, const xmpIntegers_t mod, uint32_t start, uint32_t count, uint32_t *instances_per_block, uint32_t *blocks_per_sm) {
  // geometry - # of threads per block
  // min_blocks - used for launch bounds
  // width - threads per instance
  // words - words per thread

  xmpExecutionPolicy_t policy=handle->policy;
  int32_t              bits, windowBits, largeSize=1024;
  size_t               windowBytes, scratchBytes=0;
  int32_t              precision=out->precision;
  ar_arguments_t       ar_arguments;
  powm_arguments_t     powm_arguments;
  xmpError_t           error;

  XMP_SET_DEVICE(handle);

  a->requireFormat(handle, xmpFormatStrided);
  mod->requireFormat(handle, xmpFormatStrided);
  exp->requireFormat(handle, xmpFormatStrided);

  bits=exp->precision;
  windowBits=windowBitsForPrecision(bits);

  if(precision>largeSize) {
    scratchBytes=(width*words*5+AR_DIGIT)*4*ROUND_UP(count, AR_GEOMETRY);
  }
  windowBytes=((1<<windowBits)+1)*width*words*4*ROUND_UP(count, geometry);

  error=xmpSetNecessaryScratchSize(handle, windowBytes + scratchBytes);
  if(error!=xmpErrorSuccess)
    return error;

  if(instances_per_block!=NULL)
    *instances_per_block=geometry/width;

  if(blocks_per_sm!=NULL) {
    determineMaxBlocks(warpmp_powm_kernel<geometry, min_blocks, words>, geometry, (int32_t *)blocks_per_sm);
    XMP_CHECK_CUDA();
  }

  if(instances_per_block!=NULL || blocks_per_sm!=NULL)
    return xmpErrorSuccess;

  // package up the arguments
  ar_arguments.window_data=(xmpLimb_t *)handle->scratch;
  ar_arguments.window_bits=windowBits;
  ar_arguments.a_data=a->slimbs;
  ar_arguments.a_len=DIV_ROUND_UP(a->precision, 32);
  ar_arguments.a_stride=a->stride;
  ar_arguments.a_count=a->count;
  ar_arguments.mod_data=mod->slimbs;
  ar_arguments.mod_len=DIV_ROUND_UP(mod->precision, 32);
  ar_arguments.mod_stride=mod->stride;
  ar_arguments.mod_count=mod->count;
  ar_arguments.a_indices=policy->indices[1];
  ar_arguments.mod_indices=policy->indices[3];
  ar_arguments.a_indices_count=policy->indices_count[1];
  ar_arguments.mod_indices_count=policy->indices_count[3];
  ar_arguments.width=width;

  if(words*width*32<=largeSize) {
    dim3 blocks(DIV_ROUND_UP(count, AR_GEOMETRY)), threads(AR_GEOMETRY);

    warpmp_small_ar_kernel<width*words><<<blocks, threads, 0, handle->stream>>>(ar_arguments, start, count);
  }
  else {
    dim3 blocks(DIV_ROUND_UP(count, AR_GEOMETRY)), threads(AR_GEOMETRY);

    if(words*width*32>largeSize && (words*width % AR_DIGIT)!=0) {
      // Internal error - the warpmp_large_ar_kernel will fail if word*width is not evenly divisible by the digit size
      return xmpErrorUnsupported;
    }
    ar_arguments.precision=words*width;
    ar_arguments.scratch_data=(xmpLimb_t *)handle->scratch + (windowBytes/4);
    warpmp_large_ar_kernel<AR_DIGIT><<<blocks, threads, 0, handle->stream>>>(ar_arguments, start, count);
  }

  powm_arguments.out_data=out->slimbs;
  powm_arguments.out_len=DIV_ROUND_UP(precision, 32);
  powm_arguments.out_stride=out->stride;
  powm_arguments.exp_data=exp->slimbs;
  powm_arguments.exp_stride=exp->stride;
  powm_arguments.exp_count=exp->count;
  powm_arguments.mod_count=0;
  powm_arguments.window_data=(xmpLimb_t *)handle->scratch;
  powm_arguments.width=width;
  powm_arguments.bits=exp->precision;
  powm_arguments.window_bits=windowBits;
  powm_arguments.out_indices=policy->indices[0];
  powm_arguments.exp_indices=policy->indices[2];
  powm_arguments.exp_indices_count=policy->indices_count[2];

  {
    dim3 blocks(DIV_ROUND_UP(count*width, geometry)), threads(geometry);

    warpmp_powm_kernel<geometry, min_blocks, words><<<blocks, threads, 0, handle->stream>>>(powm_arguments, start, count);
  }

  out->setFormat(xmpFormatStrided);

  XMP_CHECK_CUDA();

  return xmpErrorSuccess;
}

xmpError_t XMPAPI xmpIntegersPowmAsync(xmpHandle_t handle, xmpIntegers_t out, const xmpIntegers_t a, const xmpIntegers_t exp, const xmpIntegers_t mod, uint32_t count) {
  int                  device=handle->device;
  xmpExecutionPolicy_t policy=handle->policy;

  //verify out, base, exp, mod devices all match handle device
  if(out->device!=device || a->device!=device || exp->device!=device || mod->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  int32_t precision=out->precision;

  if(out->count<count)
    return xmpErrorInvalidCount;

  if(policy->indices[0] && policy->indices_count[0]<count)
    return xmpErrorInvalidCount;

  if(out->precision!=precision || a->precision!=precision || mod->precision!=precision)
    return xmpErrorInvalidPrecision;

  xmpAlgorithm_t alg = policy->algorithm;

  if(alg==xmpAlgorithmDefault) {
    if(count<handle->smCount*512 && handle->arch>=30 && precision<=8192)
      alg=xmpAlgorithmDistributedMP;   // for a small number of instances, use distributed
    else if(precision<=512)
      alg=xmpAlgorithmRegMP;
    else if(precision<=8192)
      alg=xmpAlgorithmDistributedMP;   // for now... distributed seems to always outperfm digitized
    else
      alg=xmpAlgorithmDigitMP;
  }

  if(alg==xmpAlgorithmRegMP) {
    out->setFormat(xmpFormatStrided);
    if(precision<=128)
      return internalPowmRegMP<128, 4, 4, 0, 0>(handle, out, a, exp, mod, 0, count, NULL, NULL);
    else if(precision<=256)
      return internalPowmRegMP<128, 4, 8, 0, 0>(handle, out, a, exp, mod, 0, count, NULL, NULL);
    else if(precision<=384)
      return internalPowmRegMP<128, 4, 12, 0, 0>(handle, out, a, exp, mod, 0, count, NULL, NULL);
    else if(precision<=512)
      return internalPowmRegMP<128, 4, 16, 0, 0>(handle, out, a, exp, mod, 0, count, NULL, NULL);
    else
      return xmpErrorUnsupported;
  }

  if(alg==xmpAlgorithmDigitMP) {
    out->setFormat(xmpFormatStrided);
    return internalPowmDigitMP<128, 4, DIGIT>(handle, out, a, exp, mod, 0, count, NULL, NULL);
  }

  if(alg==xmpAlgorithmDistributedMP) {
    out->setFormat(xmpFormatCompact);   // so I can test before the copy out kernels are done
    if(count<handle->smCount*256) {
      // use smallest number of words, to achieve lowest latency
      if(precision<=128)
        return internalPowmWarpDistributedMP<128, 4, 4, 1>(handle, out, a, exp, mod, 0, count, NULL, NULL);
      else if(precision<=256)
        return internalPowmWarpDistributedMP<128, 4, 8, 1>(handle, out, a, exp, mod, 0, count, NULL, NULL);
      else if(precision<=384)
        return internalPowmWarpDistributedMP<128, 4, 4, 3>(handle, out, a, exp, mod, 0, count, NULL, NULL);
      else if(precision<=512)
        return internalPowmWarpDistributedMP<128, 4, 16, 1>(handle, out, a, exp, mod, 0, count, NULL, NULL);
      else if(precision<=1024)
        return internalPowmWarpDistributedMP<128, 4, 32, 1>(handle, out, a, exp, mod, 0, count, NULL, NULL);
      else if(precision<=1536)
        return internalPowmWarpDistributedMP<128, 4, 16, 3>(handle, out, a, exp, mod, 0, count, NULL, NULL);
      else if(precision<=2048)
        return internalPowmWarpDistributedMP<128, 4, 32, 2>(handle, out, a, exp, mod, 0, count, NULL, NULL);
      else if(precision<=3072)
        return internalPowmWarpDistributedMP<128, 4, 32, 3>(handle, out, a, exp, mod, 0, count, NULL, NULL);
      else if(precision<=4096)
        return internalPowmWarpDistributedMP<128, 4, 32, 4>(handle, out, a, exp, mod, 0, count, NULL, NULL);
      else if(precision<=6144)
        return internalPowmWarpDistributedMP<128, 4, 32, 6>(handle, out, a, exp, mod, 0, count, NULL, NULL);
      else if(precision<=8192)
        return internalPowmWarpDistributedMP<128, 4, 32, 8>(handle, out, a, exp, mod, 0, count, NULL, NULL);
      else
        return xmpErrorUnsupported;
    }
    else {
      // use largest number of words to achieve highest throughput
      if(precision<=128)
        return internalPowmWarpDistributedMP<128, 4, 2, 2>(handle, out, a, exp, mod, 0, count, NULL, NULL);
      else if(precision<=256)
        return internalPowmWarpDistributedMP<128, 4, 2, 4>(handle, out, a, exp, mod, 0, count, NULL, NULL);
      else if(precision<=384)
        return internalPowmWarpDistributedMP<128, 4, 2, 6>(handle, out, a, exp, mod, 0, count, NULL, NULL);
      else if(precision<=512)
        return internalPowmWarpDistributedMP<128, 4, 2, 8>(handle, out, a, exp, mod, 0, count, NULL, NULL);
      else if(precision<=1024)
        return internalPowmWarpDistributedMP<128, 4, 4, 8>(handle, out, a, exp, mod, 0, count, NULL, NULL);
      else if(precision<=1536)
        return internalPowmWarpDistributedMP<128, 4, 8, 6>(handle, out, a, exp, mod, 0, count, NULL, NULL);
      else if(precision<=2048)
        return internalPowmWarpDistributedMP<128, 4, 8, 8>(handle, out, a, exp, mod, 0, count, NULL, NULL);
      else if(precision<=3072)
        return internalPowmWarpDistributedMP<128, 4, 16, 6>(handle, out, a, exp, mod, 0, count, NULL, NULL);
      else if(precision<=4096)
        return internalPowmWarpDistributedMP<128, 4, 16, 8>(handle, out, a, exp, mod, 0, count, NULL, NULL);
      else if(precision<=6144)
        return internalPowmWarpDistributedMP<128, 4, 32, 6>(handle, out, a, exp, mod, 0, count, NULL, NULL);
      else if(precision<=8192)
        return internalPowmWarpDistributedMP<128, 4, 32, 8>(handle, out, a, exp, mod, 0, count, NULL, NULL);
      else
        return xmpErrorUnsupported;
    }
  }

  return xmpErrorUnsupported;
}
