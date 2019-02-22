cmake_minimum_required(VERSION 2.6)
project(PatchMatchCuda)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
# 设置可执行文件在那个目录下生成
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

# 头文件的搜索路径
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR})



FIND_PACKAGE(CUDA)
FIND_PACKAGE(OpenCV REQUIRED)
#SET(CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)#根据自己的显卡性能配置




cuda_add_library(
	patch_match
	CudaErrorHandler.cuh
	UnifiedMemoryManaged.h
	AlgorithmParameters.h
	LineState.h
	GlobalState.h
	PatchMatch.h PatchMatch.cu
)
cuda_add_executable( main main.cu )
target_link_libraries( main ${OpenCV_LIBS}  patch_match)
