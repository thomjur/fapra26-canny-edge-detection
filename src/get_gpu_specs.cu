// Hilsprogramm, um Infos über die Graka zu bekommen, die es mit nvidia-smi und
// nvidia-settings nicht gibt
#include <cstdio>
#include <cuda_runtime.h>

int main() {
  int device = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  double memClockGHz = prop.memoryClockRate * 1e-6; // kHz -> GHz
  double busWidthBytes = prop.memoryBusWidth / 8.0;
  // GDDR: effektive Datenrate = 2x Takt (DDR)
  double bandwidthGBs = 2.0 * memClockGHz * busWidthBytes;

  printf("GPU                            : %s\n", prop.name);
  printf("Compute Capability             : %d.%d\n", prop.major, prop.minor);
  printf("Gesamter GPU-Speicher          : %.0f MB\n",
         prop.totalGlobalMem / (1024.0 * 1024.0));
  printf("L2-Cache                       : %.0f KB\n",
         prop.l2CacheSize / 1024.0);
  printf("Streaming Multiprocessors (SM) : %d\n", prop.multiProcessorCount);
  printf("Max. Threads pro SM            : %d\n",
         prop.maxThreadsPerMultiProcessor);
  printf("Warpgroesse                    : %d\n", prop.warpSize);
  printf("Memory-Takt                    : %.0f MHz\n",
         prop.memoryClockRate / 1000.0);
  printf("Memory-Busbreite                : %d bit\n", prop.memoryBusWidth);
  printf("Peak-Speicherbandbreite        : %.0f GB/s\n", bandwidthGBs);
  printf("GPU-Takt                       : %.0f MHz\n",
         prop.clockRate / 1000.0);

  return 0;
}
