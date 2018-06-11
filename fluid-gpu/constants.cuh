// solver parameters
const static float2 G = make_float2(0.f, 12000*-9.8f); // external (gravitational) forces
__constant__ const static float REST_DENS = 1000.f; // rest density
__constant__ const static float GAS_CONST = 2000.f; // const for equation of state
__constant__ const static float H = 4.f; // kernel radius
const static float HSQ = H*H; // radius^2 for optimization
const static float MASS = 65.f; // assume all particles have the same mass
__constant__ const static float VISC = 250.f; // viscosity constant
__constant__ const static float DT = 0.0002f; // integration timestep

// smoothing kernels defined in Müller and their gradients
const static float POLY6 = 315.f/(65.f*M_PI*pow(H, 9.f));
const static float SPIKY_GRAD = -45.f/(M_PI*pow(H, 6.f));
const static float VISC_LAP = 45.f/(M_PI*pow(H, 6.f));

// simulation parameters
__constant__ const static float EPS = H; // boundary epsilon
__constant__ const static float BOUND_DAMPING = -0.5f;

__constant__ const static int INF = 1000000000;

// interaction
__constant__ const static int MAX_PARTICLES = 25000;
__constant__ const static int DAM_PARTICLES = 1000;
__constant__ const static int BLOCK_PARTICLES = 250;

// rendering projection parameters
__constant__ const static int WINDOW_WIDTH = 800;
__constant__ const static int WINDOW_HEIGHT = 800;
__constant__ const static float VIEW_WIDTH = 800.f;
__constant__ const static float VIEW_HEIGHT = 800.f;
