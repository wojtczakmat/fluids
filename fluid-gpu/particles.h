#pragma once
#include "cuda_runtime.h"

__align__(32)
struct Particle
{
	float2 x;
	float2 v;
	float2 f;
	float rho, p;
};

void initCuda(GLuint vbo, int numberOfParticles);
void simulate();
