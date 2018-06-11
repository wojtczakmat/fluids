#include "GL/glew.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

#include <stdio.h>

#include "particles.h"
#include "constants.cuh"
//#include "helpers.cuh"

#define getLastCudaError(msg) __getLastCudaError (msg, __FILE__, __LINE__)
inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err)
	{
		FILE* log = fopen("log.txt", "a+");

		fprintf(log, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
			file, line, errorMessage, (int)err, cudaGetErrorString(err));

		printf("%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
			file, line, errorMessage, (int)err, cudaGetErrorString(err));

		fclose(log);
	}
}

__forceinline__ __device__ float2 normalized(float2 vec)
{
	float len = vec.x * vec.x + vec.y * vec.y;
	if (len == 0.f)
		return vec;

	float invLen = 1.0f / sqrtf(len);

	return make_float2(vec.x * invLen, vec.y * invLen);
}

cudaArray_t deviceFishBuffer;
int numOfParticles;

cudaGraphicsResource* m_cuda_posvbo_resource;

__host__ int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = iDivUp(n, numThreads);
}

void initCuda(GLuint vbo, int numberOfFishes)
{
	numOfParticles = numberOfFishes;

	cudaSetDevice(0);
	getLastCudaError("cudaSetDevice failed");

	cudaGraphicsGLRegisterBuffer(&m_cuda_posvbo_resource, vbo, cudaGraphicsRegisterFlagsNone);
	getLastCudaError("cudaGraphicsGLRegisterBuffer failed");
}

Particle* mapVboPtr()
{
	void *ptr;
	cudaGraphicsMapResources(1, &m_cuda_posvbo_resource, 0);
	getLastCudaError("cudaGraphicsMapResources failed");

	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes, m_cuda_posvbo_resource);
	getLastCudaError("cudaGraphicsResourceGetMappedPointers failed");

	return (Particle*)ptr;
}

void unmapVboPtr()
{
	cudaGraphicsUnmapResources(1, &m_cuda_posvbo_resource, 0);
	getLastCudaError("cudaGraphicsUnmapResources failed");
}



__forceinline__ __device__ float dot(float2 a, float2 b)
{
	return a.x * b.x + a.y * b.y;
}

__global__ void initParticles(Particle* partBuffer, int n)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= n)
		return;

	__shared__ curandState state;

	if (blockIdx.x == 0)
	{
		curand_init(5555, 0, 0, &state);
	}

	Particle* p = partBuffer + index;
	//random float added to x to not place particles perfectly above each other
	p->x = make_float2(400 - 50*H + (index % 50) * H*2 + curand_normal(&state), 800 - index / 50 * H*2);
	//p->x = make_float2(0,0);
	p->v = make_float2(0,0);
	p->f = make_float2(0,0);

	p->p = p->rho = 0;

	//printf("init: %f %f\n", p->x.x, p->x.y);
}

__global__ void computeDensity(Particle* partBuffer, int n, float POLY6)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n)
		return;

	Particle* pi = partBuffer + idx;

	for(int j = 0; j < n; ++j)
	{
		Particle* pj = partBuffer + j;

		float2 rij = make_float2(pj->x.x - pi->x.x, pj->x.y - pi->x.y);
		float r2 = rij.x*rij.x + rij.y*rij.y;//squaredNorm(rij);
		float diff = HSQ-r2;

		if(diff > 0)
		{
			float pow3 = diff * diff * diff;
//			// this computation is symmetric
			pi->rho += MASS * POLY6 * pow3;
		}
	}
	//printf ("%f\n", pi->rho);
	pi->p = GAS_CONST*(pi->rho - REST_DENS);
}

__global__ void zeroRhos(Particle* partBuffer, int n)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n)
		return;

	Particle* p = partBuffer + idx;
	p->rho = 0;
}

__global__ void computeForces(Particle* partBuffer, int n, float SPIKY_GRAD, float VISC_LAP)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n)
		return;

	Particle* pi = partBuffer + idx;

	float2 fpress = make_float2(0.f, 0.f);
	float2 fvisc = make_float2(0.f, 0.f);

	Particle* pj;
	for(int j = 0; j < n; ++j)
	{
		pj = partBuffer + j;

		if(&pi == &pj)
			continue;

		float2 rij = make_float2(pj->x.x - pi->x.x, pj->x.y - pi->x.y);
		float normr = rij.x*rij.x + rij.y*rij.y;

		if(normr < HSQ)
		{
			float r = sqrtf(normr);
			float2 press = normalized(rij);
			//printf("r: %f\n", r);
			float pressMultiplier = MASS*(pi->p + pj->p)/(2.f * pj->rho) * SPIKY_GRAD*(H-r)*(H-r);
			float viscMultiplier = VISC*MASS/pj->rho * VISC_LAP*(H-r);

			// compute pressure force contribution
			//printf ("%f\n", pj->rho);

			fpress.x += -press.x * pressMultiplier;
			fpress.y += -press.y * pressMultiplier;

			// compute viscosity force contribution
			float2 visc = make_float2(pj->v.x - pi->v.x, pj->v.y - pi->v.y);

			fvisc.x += visc.x * viscMultiplier;
			fvisc.y += visc.y * viscMultiplier;
		}
	}

	float2 G = make_float2(0.f, 12000*-9.8f);
	float2 fgrav = make_float2(G.x * pi->rho, G.y * pi->rho);
	pi->f.x = fgrav.x + fvisc.x + fpress.x;
	pi->f.y = fgrav.y + fvisc.y + fpress.y;

	//printf("fpress: %f %f\n", fpress.x, fpress.y);
}

__global__ void integrate(Particle* partBuffer, int n)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n)
		return;

	Particle* p = partBuffer + idx;

	// forward Euler integration
	p->v.x += p->f.x * (DT / p->rho);
	p->v.y += p->f.y * (DT / p->rho);

	//printf("%f %f\n", p->x.x, p->x.y);

	p->x.x += p->v.x * DT;
	p->x.y += p->v.y * DT;

	// enforce boundary conditions
	if(p->x.x - EPS < 0.0f)
	{
		p->v.x *= BOUND_DAMPING;
		p->x.x = EPS;
	}
	if(p->x.x + EPS > VIEW_WIDTH)
	{
		p->v.x *= BOUND_DAMPING;
		p->x.x = VIEW_WIDTH-EPS;
	}
	if(p->x.y - EPS < 0.0f)
	{
		p->v.y *= BOUND_DAMPING;
		p->x.y = EPS;
	}
	if(p->x.y+EPS > VIEW_HEIGHT)
	{
		p->v.y *= BOUND_DAMPING;
		p->x.y = VIEW_HEIGHT-EPS;
	}
}

bool particlesInitialized = false;
void simulate()
{
	Particle* ptr = mapVboPtr();

	int n = numOfParticles;
	uint blocks, threads;
	computeGridSize(n, 512, blocks, threads);

	if (!particlesInitialized)
	{
		printf("run\n");
		particlesInitialized = true;
		initParticles <<<blocks, threads >>>(ptr, n);
		//cudaDeviceSynchronize();
	}

	zeroRhos<<< blocks, threads >>>(ptr, n);
	//cudaDeviceSynchronize();
	computeDensity<<< blocks, threads >>>(ptr, n, POLY6);
	//cudaDeviceSynchronize();
	computeForces<<< blocks, threads >>>(ptr, n, SPIKY_GRAD, VISC_LAP);
	//cudaDeviceSynchronize();
	integrate<<< blocks, threads >>>(ptr, n);
	//cudaDeviceSynchronize();

	unmapVboPtr();
}
