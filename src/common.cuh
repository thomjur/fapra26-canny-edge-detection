#pragma once

/// Wraps a CUDA API call and aborts on failure.
/// Prints error string, file, and line to stderr before calling exit(EXIT_FAILURE).
#define CUDA_THROW_IF_FAILED(expr)					        \
	do								                        \
	{								                        \
		cudaError_t err = (expr);				            \
		if (err != cudaSuccess)					            \
		{							                        \
			fprintf(stderr, "CUDA Error: %s\n at %s:%d\n",	\
			cudaGetErrorString(err), __FILE__, __LINE__);   \
			exit(EXIT_FAILURE);                             \
		}                                                   \
	} while (0)
