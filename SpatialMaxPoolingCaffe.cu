#include <THC/THC.h>

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


// kernels borrowed from Caffe

__global__ void MaxPoolForward(const int nthreads, const float* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, float* top_data,
    int* mask, float* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    float maxval = -FLT_MAX;
    int maxidx = -1;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_data[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_data[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}


extern "C"
void SpatialMaxPoolingCaffe_updateOutput(THCState* state, THCudaTensor* input,
    THCudaTensor* output, THCudaTensor* indices, int kW, int kH, int dW, int dH, bool train)
{
  long nInputCols, nInputRows, nInputPlane, batchSize;

  if (input->nDimension == 3) {
    nInputCols = input->size[2];
    nInputRows = input->size[1];
    nInputPlane = input->size[0];
    batchSize = 1;
  }
  else
  {
    nInputCols = input->size[3];
    nInputRows = input->size[2];
    nInputPlane = input->size[1];
    batchSize = input->size[0];
  }

  long nOutputCols = ceil(float(nInputCols - kW) / float(dW)) + 1;
  long nOutputRows = ceil(float(nInputRows - kH) / float(dH)) + 1;
  int pW = 0, pH = 0; //TODO

  input = THCudaTensor_newContiguous(state, input);
  float* input_data = THCudaTensor_data(state, input);

  THCudaTensor_resize4d(state, output, batchSize, nInputPlane, nOutputRows, nOutputCols);
  THCudaTensor_resizeAs(state, indices, output);
  
  float* indices_data = THCudaTensor_data(state, indices);
  float* output_data = THCudaTensor_data(state, output);

  int count = THCudaTensor_nElement(state, output);

  MaxPoolForward <<< GET_BLOCKS(count), CUDA_NUM_THREADS >>> (count, input_data,
	batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
	kH, kW, dH, dW, pH, pW, output_data, NULL, indices_data);

  if(input->nDimension == 3)
    THCudaTensor_resize3d(state, output, nInputPlane, nOutputRows, nOutputCols);

  // clean
  THCudaTensor_free(state, input);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialMaxPoolingCaffe_updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
}


__global__ void MaxPoolBackward(const int nthreads, const float* top_diff,
	const int* mask, const float* top_mask, const int num, const int channels,
	const int height, const int width, const int pooled_height,
	const int pooled_width, const int kernel_h, const int kernel_w,
	const int stride_h, const int stride_w, const int pad_h, const int pad_w,
	float* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
	// find out the local index
	// find out the local offset
	int w = index % width;
	int h = (index / width) % height;
	int c = (index / width / height) % channels;
	int n = index / width / height / channels;
	int phstart =
		(h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
	int phend = min((h + pad_h) / stride_h + 1, pooled_height);
	int pwstart =
		(w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
	int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
	float gradient = 0;
	int offset = (n * channels + c) * pooled_height * pooled_width;
	top_diff += offset;
	if (mask) {
	  mask += offset;
	  for (int ph = phstart; ph < phend; ++ph) {
		for (int pw = pwstart; pw < pwend; ++pw) {
		  if (mask[ph * pooled_width + pw] == h * width + w) {
			gradient += top_diff[ph * pooled_width + pw];
		  }
		}
	  }
	} else {
	  top_mask += offset;
	  for (int ph = phstart; ph < phend; ++ph) {
		for (int pw = pwstart; pw < pwend; ++pw) {
		  if (top_mask[ph * pooled_width + pw] == h * width + w) {
			gradient += top_diff[ph * pooled_width + pw];
		  }
		}
	  }
	}
	bottom_diff[index] = gradient;
  }
}

extern "C"
void SpatialMaxPoolingCaffe_updateGradInput(THCState* state, THCudaTensor* input,
    THCudaTensor* gradInput, THCudaTensor* gradOutput, THCudaTensor* indices, int kW, int kH, int dW, int dH)
{
  long nInputCols, nInputRows, nInputPlane, batchSize;

  if (input->nDimension == 3) {
    nInputCols = input->size[2];
    nInputRows = input->size[1];
    nInputPlane = input->size[0];
    batchSize = 1;
  }
  else
  {
    nInputCols = input->size[3];
    nInputRows = input->size[2];
    nInputPlane = input->size[1];
    batchSize = input->size[0];
  }

  long nOutputCols = ceil(float(nInputCols - kW) / float(dW)) + 1;
  long nOutputRows = ceil(float(nInputRows - kH) / float(dH)) + 1;
  int pW = 0, pH = 0; //TODO

  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  float* gradOutput_data = THCudaTensor_data(state, gradOutput);

  THCudaTensor_resizeAs(state, gradInput, input);
  
  int count = THCudaTensor_nElement(state, input);

  MaxPoolBackward <<< GET_BLOCKS(count), CUDA_NUM_THREADS >>> (count,
	  gradOutput_data,
      NULL, THCudaTensor_data(state, indices),
      batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
      kH, kW, dH, dW, pH, pW,
      THCudaTensor_data(state, gradInput));

  // clean
  THCudaTensor_free(state, gradOutput);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialMaxPoolingCaffe_updateGradInput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
}
