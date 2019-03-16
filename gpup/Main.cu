#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CUDA_helper/helper_cuda.h"

#include "Main_kernel.cuh"

#define BLOCK_SIZE 16

extern "C"
{
	void SphereWave(float3 *LocationNormal, int Segments, int Rings, float Radius, float Time)
	{
		dim3 Block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 Grid((Segments + Block.x - 1) / Block.x, (Rings + Block.y - 1) / Block.y);
		SphereKernel <<< Grid, Block >>> (LocationNormal, Segments, Rings, Radius, Time);

		// Check for any errors launching the kernel
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

	void AddCUDAArrays(float *First, float* Second, unsigned int Number)
	{
		dim3 Block(BLOCK_SIZE*BLOCK_SIZE);
		dim3 Grid((Number + Block.x - 1) / Block.x);
		AddKernel <<< Grid, Block >>> (First, Second, Number);

		// Check for any errors launching the kernel
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

	}

	void UpdateLocation(float* Location, float* Velocity, float* Acceleration, unsigned int Number)
	{
		dim3 Block(BLOCK_SIZE*BLOCK_SIZE);
		dim3 Grid((Number + Block.x - 1) / Block.x);
		UpdateKernel <<< Grid, Block >>> (Location, Velocity, Acceleration, Number);

		// Check for any errors launching the kernel
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

	}

	void SetCUDAArray(float3* Coordinates, float3 Vector, unsigned int Number)
	{
		dim3 Block(BLOCK_SIZE*BLOCK_SIZE);
		dim3 Grid((Number + Block.x - 1) / Block.x);
		SetVectorKernel <<< Grid, Block >>> (Coordinates, Vector, Number);

		// Check for any errors launching the kernel
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

	}

	void CheckBoundaries(float3* Location, float3* Velocity, unsigned int Number, float Radius, float Limit, float Friction)
	{
		dim3 Block(BLOCK_SIZE*BLOCK_SIZE);
		dim3 Grid((Number + Block.x - 1) / Block.x);
		BoundaryKernel <<< Grid, Block >>> (Location, Velocity, Number, Radius, Limit, Friction);

		// Check for any errors launching the kernel
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

	void ApplyAttractor(float3* Location, float3* Acceleration, float3 AttractorLocation, float AttractorGravity, unsigned int Number)
	{
		dim3 Block(BLOCK_SIZE*BLOCK_SIZE);
		dim3 Grid((Number + Block.x - 1) / Block.x);
		AttractorKernel <<< Grid, Block >>> (Location, Acceleration, AttractorLocation, AttractorGravity, Number);

		// Check for any errors launching the kernel
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

	void CheckCollisions(float3* Location, float3* Velocity, float3* Acceleration, unsigned int Number, float Radius, float Spring, float Damping, float Shear, float Attraction)
	{
		dim3 Block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 Grid((Number + Block.x - 1) / Block.x, (Number + Block.y - 1) / Block.y);
		CollisionsKernel <<< Grid, Block >>> (Location, Velocity, Acceleration, Number, Radius, Spring, Damping, Shear, Attraction);

		// Check for any errors launching the kernel
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

	void CheckLifespan(float3* Location, float3* Velocity, float* Lifespan, float4 * Color, unsigned int  Number)
	{
		dim3 Block(BLOCK_SIZE*BLOCK_SIZE);
		dim3 Grid((Number + Block.x - 1) / Block.x);
		LifespanKernel <<< Grid, Block >>> (Location, Velocity, Lifespan, Color, Number);

		// Check for any errors launching the kernel
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

	void RandomCUDAArray(float* Array, float Minimum, float Maximum, unsigned int Number)
	{
		dim3 Block(BLOCK_SIZE*BLOCK_SIZE);
		dim3 Grid((Number + Block.x - 1) / Block.x);
		RandomKernel <<< Grid, Block >>> (Array, Minimum, Maximum, Number);

		// Check for any errors launching the kernel
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}
}