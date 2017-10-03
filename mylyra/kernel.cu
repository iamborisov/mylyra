
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <stdint.h>

#include <Windows.h>
// MSVC defines this in winsock2.h!?
/*typedef struct timeval {
	long tv_sec;
	long tv_usec;
} timeval;*/

int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
	// Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
	// This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
	// until 00:00:00 January 1, 1970 
	static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

	SYSTEMTIME  system_time;
	FILETIME    file_time;
	uint64_t    time;

	GetSystemTime(&system_time);
	SystemTimeToFileTime(&system_time, &file_time);
	time = ((uint64_t)file_time.dwLowDateTime);
	time += ((uint64_t)file_time.dwHighDateTime) << 32;

	tp->tv_sec = (long)((time - EPOCH) / 10000000L);
	tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
	return 0;
}


typedef unsigned char byte;

#ifndef N_COLS
#define N_COLS 8
#endif

#ifndef nPARALLEL
#define nPARALLEL 2
#endif

#define ROW_LEN_INT64 (BLOCK_LEN_INT64 * N_COLS)                //Total length of a row: N_COLS blocks
#define ROW_LEN_BYTES (ROW_LEN_INT64 * 8)                       //Number of bytes per row

#if defined(__GNUC__)
#define ALIGN __attribute__ ((aligned(32)))
#elif defined(_MSC_VER)
#define ALIGN __declspec(align(32))
#else
#define ALIGN
#endif

//Block length required so Blake2's Initialization Vector (IV) is not overwritten (THIS SHOULD NOT BE MODIFIED)
#define BLOCK_LEN_BLAKE2_SAFE_INT64 8                                   //512 bits (=64 bytes, =8 uint64_t)
#define BLOCK_LEN_BLAKE2_SAFE_BYTES (BLOCK_LEN_BLAKE2_SAFE_INT64 * 8)   //same as above, in bytes

//default block lenght: 768 bits
#ifndef BLOCK_LEN_INT64
#define BLOCK_LEN_INT64 12                                      //Block length: 768 bits (=96 bytes, =12 uint64_t)
#endif

#define BLOCK_LEN_BYTES (BLOCK_LEN_INT64 * 8)                           //Block length, in bytes

#define STATESIZE_INT64 16
#define STATESIZE_BYTES (16 * sizeof (uint64_t))

#ifndef RHO
#define RHO 1                                                   //Number of reduced rounds performed
#endif

/*Blake2b IV Array*/
__device__ static const uint64_t blake2b_IV[8] =
{
	0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
	0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
	0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
	0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

/*Blake2b's rotation*/
__device__ static inline uint64_t rotr64(const uint64_t w, const unsigned c){
	return (w >> c) | (w << (64 - c));
}

/*Main change compared with Blake2b*/
__device__ static inline uint64_t fBlaMka(uint64_t x, uint64_t y){
	uint32_t lessX = (uint32_t)x;
	uint32_t lessY = (uint32_t)y;

	uint64_t lessZ = (uint64_t)lessX;
	lessZ = lessZ * lessY;
	lessZ = lessZ << 1;

	uint64_t z = lessZ + x + y;

	return z;
}

#define DIAGONALIZE(r,v) \
    t0=v[4];                      v[4]=v[5]; v[5]=v[6]; v[6]=v[7]; v[7]=t0; \
    t0=v[8]; t1=v[9];             v[8]=v[10]; v[9]=v[11]; v[10]=t0; v[11]=t1; \
    t0=v[12]; t1=v[13]; t2=v[14]; v[12]=v[15]; v[13]=t0; v[14]=t1; v[15]=t2;

/*Blake2b's G function*/
#define G(r,i,a,b,c,d) \
  do { \
    a = a + b; \
    d = rotr64(d ^ a, 32); \
    c = c + d; \
    b = rotr64(b ^ c, 24); \
    a = a + b; \
    d = rotr64(d ^ a, 16); \
    c = c + d; \
    b = rotr64(b ^ c, 63); \
    } while(0)

/*One Round of the Blake2b's compression function*/
#define ROUND_LYRA(r)  \
    G(r,0,v[ 0],v[ 4],v[ 8],v[12]); \
    G(r,1,v[ 1],v[ 5],v[ 9],v[13]); \
    G(r,2,v[ 2],v[ 6],v[10],v[14]); \
    G(r,3,v[ 3],v[ 7],v[11],v[15]); \
    G(r,4,v[ 0],v[ 5],v[10],v[15]); \
    G(r,5,v[ 1],v[ 6],v[11],v[12]); \
    G(r,6,v[ 2],v[ 7],v[ 8],v[13]); \
    G(r,7,v[ 3],v[ 4],v[ 9],v[14]);


//---- Initialization
__global__ void bootStrapGPU(uint64_t * memMatrixGPU, unsigned char * pkeysGPU, unsigned int kLen, unsigned char *pwdGPU, unsigned int pwdlen, unsigned char *saltGPU, unsigned int saltlen, unsigned int timeCost, unsigned int nRows, unsigned int nCols, uint64_t nBlocksInput, unsigned int totalPasswords);

//---- Housekeeping
__global__ void initState(uint64_t state[/*16*/], unsigned int totalPasswords);

//---- Squeezes
__global__ void reducedSqueezeRow0(uint64_t* row, uint64_t* state, unsigned int totalPasswords);
__global__ void squeeze(uint64_t *state, byte *out, unsigned int len, unsigned int totalPasswords);

//---- Absorbs
__global__ void absorbInput(uint64_t * memMatrixGPU, uint64_t * stateThreadGPU, uint64_t *stateIdxGPU, uint64_t nBlocksInput, unsigned int totalPasswords);

//---- Duplexes
__global__ void reducedDuplexRow1and2(uint64_t *rowIn, uint64_t *state, unsigned int totalPasswords, int first, int second);

//---- Setup and Wandering
__global__ void setupPhaseWanderingGPU(uint64_t * memMatrixGPU, uint64_t * stateThreadGPU, uint64_t sizeSlice, unsigned int totalPasswords, unsigned int timeCost);
__global__ void setupPhaseWanderingGPU_P1(uint64_t * memMatrixGPU, uint64_t * stateThreadGPU, uint64_t sizeSlice, unsigned int totalPasswords, unsigned int timeCost);

//---- Misc
void printArray(unsigned char *array, unsigned int size, char *name) {
	int i;
	printf("%s: ", name);
	for (i = 0; i < size; i++) {
		printf("%2x|", array[i]);
	}
	printf("\n");
}

int gpuMult(void *K, unsigned int kLen, unsigned char **passwords, unsigned int pwdlen, unsigned char **salts, unsigned int saltlen, unsigned int timeCost, unsigned int nRows, unsigned int nCols, unsigned int totalPasswords, unsigned int gridSize, unsigned int blockSize) {
	int result = 0;

	//============================= Basic variables ============================//
#if (nPARALLEL > 1)
	int64_t i, j, k; //auxiliary iteration counter
#endif   // nPARALLEL > 1

	cudaError_t errorCUDA;
	uint64_t sizeSlice = nRows / nPARALLEL;
	//==========================================================================/
	uint64_t nBlocksInput;

	//Checks kernel geometry configuration
	if ((gridSize * blockSize) != (totalPasswords * nPARALLEL)) {
		printf("Error in thread geometry: (gridSize * blockSize) != (totalPasswords * nPARALLEL).\n");
		return -1;
	}
	//Checks whether or not the salt+password are within the accepted limits
	if (pwdlen + saltlen > ROW_LEN_BYTES) {
		return -1;
	}

	//========== Initializing the Memory Matrix and Keys =============//
	//Allocates the keys
	unsigned char *pKeys = (unsigned char *)malloc(totalPasswords * nPARALLEL * kLen * sizeof(unsigned char));
	if (pKeys == NULL) {
		return -1;
	}

	// GPU memory matrix alloc:
	// Memory matrix: nRows of nCols blocks, each block having BLOCK_LEN_INT64 64-bit words
	uint64_t *memMatrixGPU;
	errorCUDA = cudaMalloc((void**)&memMatrixGPU, totalPasswords * nRows * ROW_LEN_BYTES);
	if (cudaSuccess != errorCUDA) {
		printf("CUDA memory allocation error in file %s, line %d!\n", __FILE__, __LINE__);
		printf("Error: %s \n", cudaGetErrorString(errorCUDA));
		return -2;
	}

	//Allocates the GPU keys
	unsigned char *pkeysGPU;
	errorCUDA = cudaMalloc((void**)&pkeysGPU, totalPasswords * nPARALLEL * kLen * sizeof(unsigned char));
	if (cudaSuccess != errorCUDA) {
		printf("CUDA memory allocation error in file %s, line %d!\n", __FILE__, __LINE__);
		printf("Error: %s \n", cudaGetErrorString(errorCUDA));
		return -2;
	}

	//Sponge state: 16 uint64_t, BLOCK_LEN_INT64 words of them for the bitrate (b) and the remainder for the capacity (c)
	uint64_t *stateThreadGPU;
	errorCUDA = cudaMalloc((void**)&stateThreadGPU, totalPasswords * nPARALLEL * STATESIZE_BYTES);
	if (cudaSuccess != errorCUDA) {
		printf("CUDA memory allocation error in file %s, line %d!\n", __FILE__, __LINE__);
		printf("Error: %s \n", cudaGetErrorString(errorCUDA));
		return -2;
	}

	// stateThreadGPU cleanup:
	cudaMemset(stateThreadGPU, 0, totalPasswords * nPARALLEL * STATESIZE_BYTES);
	if (cudaSuccess != cudaGetLastError()) {
		printf("CUDA memory setting error in file %s, line %d!\n", __FILE__, __LINE__);
		printf("Error: %s \n", cudaGetErrorString(cudaGetLastError()));
		return -2;
	}

	//Allocates the State Index to be absorbed by each thread.
	uint64_t *stateIdxGPU;
	errorCUDA = cudaMalloc((void**)&stateIdxGPU, totalPasswords * nPARALLEL * BLOCK_LEN_BLAKE2_SAFE_BYTES);
	if (cudaSuccess != errorCUDA) {
		printf("CUDA memory allocation error in file %s, line %d!\n", __FILE__, __LINE__);
		printf("Error: %s \n", cudaGetErrorString(errorCUDA));
		return -2;
	}

	//Allocates the Password in GPU.
	unsigned char *pwdGPU;
	errorCUDA = cudaMalloc((void**)&pwdGPU, totalPasswords * pwdlen);
	if (cudaSuccess != errorCUDA) {
		printf("CUDA memory allocation error in file %s, line %d!\n", __FILE__, __LINE__);
		printf("Error: %s \n", cudaGetErrorString(errorCUDA));
		return -2;
	}

	// Transfers the password to GPU.
	errorCUDA = cudaMemcpy(pwdGPU, passwords[0], totalPasswords * pwdlen, cudaMemcpyHostToDevice);
	if (cudaSuccess != errorCUDA) {
		printf("CUDA memory copy error in file %s, line %d!\n", __FILE__, __LINE__);
		printf("Error: %s \n", cudaGetErrorString(errorCUDA));
		return -2;
	}

	//Allocates the Salt in GPU.
	unsigned char *saltGPU;
	errorCUDA = cudaMalloc((void**)&saltGPU, totalPasswords * saltlen);
	if (cudaSuccess != errorCUDA) {
		printf("CUDA memory allocation error in file %s, line %d!\n", __FILE__, __LINE__);
		printf("Error: %s \n", cudaGetErrorString(errorCUDA));
		return -2;
	}

	// Transfers the salt to GPU.
	errorCUDA = cudaMemcpy(saltGPU, salts[0], totalPasswords * saltlen, cudaMemcpyHostToDevice);
	if (cudaSuccess != errorCUDA) {
		printf("CUDA memory copy error in file %s, line %d!\n", __FILE__, __LINE__);
		printf("Error: %s \n", cudaGetErrorString(errorCUDA));
		return -2;
	}

	//========================== BootStrapping Phase ==========================//
#if (nPARALLEL == 1)
	nBlocksInput = ((saltlen + pwdlen + 6 * sizeof(int)) / BLOCK_LEN_BLAKE2_SAFE_BYTES) + 1;
#endif  // nPARALLEL == 1

#if (nPARALLEL > 1)
	nBlocksInput = ((saltlen + pwdlen + 8 * sizeof(int)) / BLOCK_LEN_BLAKE2_SAFE_BYTES) + 1;
#endif   // nPARALLEL > 1

	bootStrapGPU << <gridSize, blockSize >> >(memMatrixGPU, pkeysGPU, kLen, pwdGPU, pwdlen, saltGPU, saltlen, timeCost, nRows, nCols, nBlocksInput, totalPasswords);

	// Needs to wait all threads:
	cudaThreadSynchronize();

	errorCUDA = cudaGetLastError();
	if (cudaSuccess != errorCUDA) {
		printf("CUDA kernel call error in file %s, line %d!\n", __FILE__, __LINE__);
		printf("Error: %s \n", cudaGetErrorString(errorCUDA));
		return -2;
	}

	//============== Initializing the Sponge State =============/
	initState << <gridSize, blockSize >> >(stateThreadGPU, totalPasswords);

	// Wait all threads to verify execution errors.
	cudaThreadSynchronize();

	errorCUDA = cudaGetLastError();
	if (cudaSuccess != errorCUDA) {
		printf("CUDA kernel call error in file %s, line %d!\n", __FILE__, __LINE__);
		printf("Error: %s \n", cudaGetErrorString(errorCUDA));
		return -2;
	}

	//============= Absorbing the input data with the sponge ===============//
	absorbInput << <gridSize, blockSize >> >(memMatrixGPU, stateThreadGPU, stateIdxGPU, nBlocksInput, totalPasswords);

	// Wait all threads to verify execution errors.
	cudaThreadSynchronize();

	errorCUDA = cudaGetLastError();
	if (cudaSuccess != errorCUDA) {
		printf("CUDA kernel call error in file %s, line %d!\n", __FILE__, __LINE__);
		printf("Error: %s \n", cudaGetErrorString(errorCUDA));
		return -2;
	}

	//================================ Setup and Wandering Phase =============================//
	//Initializes M[0]
	reducedSqueezeRow0 << <gridSize, blockSize >> >(memMatrixGPU, stateThreadGPU, totalPasswords); //The locally copied password is most likely overwritten here

	// Wait all threads to verify execution errors.
	cudaThreadSynchronize();

	errorCUDA = cudaGetLastError();
	if (cudaSuccess != errorCUDA) {
		printf("CUDA kernel call error in file %s, line %d!\n", __FILE__, __LINE__);
		printf("Error: %s \n", cudaGetErrorString(errorCUDA));
		return -2;
	}

	//Initializes M[1]
	reducedDuplexRow1and2 << <gridSize, blockSize >> >(memMatrixGPU, stateThreadGPU, totalPasswords, 0, 1);

	// Wait all threads to verify execution errors.
	cudaThreadSynchronize();

	errorCUDA = cudaGetLastError();
	if (cudaSuccess != errorCUDA) {
		printf("CUDA kernel call error in file %s, line %d!\n", __FILE__, __LINE__);
		printf("Error: %s \n", cudaGetErrorString(errorCUDA));
		return -2;
	}

	//Initializes M[2]
	reducedDuplexRow1and2 << <gridSize, blockSize >> >(memMatrixGPU, stateThreadGPU, totalPasswords, 1, 2);

	cudaThreadSynchronize();

	errorCUDA = cudaGetLastError();
	if (cudaSuccess != errorCUDA) {
		printf("CUDA kernel call error in file %s, line %d!\n", __FILE__, __LINE__);
		printf("Error: %s \n", cudaGetErrorString(errorCUDA));
		return -2;
	}

#if (nPARALLEL > 1)
	// Runs Setup and Wandering Phase
	setupPhaseWanderingGPU << <gridSize, blockSize >> >(memMatrixGPU, stateThreadGPU, sizeSlice, totalPasswords, timeCost);
#endif //nParallel > 1

	cudaThreadSynchronize();

	errorCUDA = cudaGetLastError();
	if (cudaSuccess != errorCUDA) {
		printf("CUDA kernel call error in file %s, line %d!\n", __FILE__, __LINE__);
		printf("Error after SetupWandering: %s \n", cudaGetErrorString(errorCUDA));
		return -2;
	}

	//Squeezes the keys
	squeeze << <gridSize, blockSize >> >(stateThreadGPU, pkeysGPU, kLen, totalPasswords);

	cudaThreadSynchronize();

	errorCUDA = cudaGetLastError();
	if (cudaSuccess != errorCUDA) {
		printf("CUDA kernel call error in file %s, line %d!\n", __FILE__, __LINE__);
		printf("Error: %s \n", cudaGetErrorString(errorCUDA));
		return -2;
	}

	// Getting the keys back.
	errorCUDA = cudaMemcpy(pKeys, pkeysGPU, totalPasswords * nPARALLEL * kLen * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaSuccess != errorCUDA) {
		printf("CUDA memory copy error in file %s, line %d!\n", __FILE__, __LINE__);
		printf("Error: %s \n", cudaGetErrorString(errorCUDA));
		return -2;
	}


#if (nPARALLEL > 1)
	// XORs all Keys
	for (k = 0; k < totalPasswords; k++) {
		for (i = 1; i < nPARALLEL; i++) {
			for (j = 0; j < kLen; j++) {
				pKeys[k * kLen * nPARALLEL + j] ^= pKeys[k * kLen * nPARALLEL + i * kLen + j];
			}
		}
	}

	//Move the keys to proper place
	for (k = 1; k < totalPasswords; k++) {
		for (j = 0; j < kLen; j++) {
			pKeys[k * kLen + j] = pKeys[k * kLen * nPARALLEL + j];
		}
	}
#endif //nParallel > 1

	// Returns in the correct variable
	memcpy(K, pKeys, totalPasswords * kLen * sizeof(unsigned char));

	//========== Frees the Memory Matrix and Keys =============//
	cudaFree(memMatrixGPU);
	cudaFree(pkeysGPU);
	cudaFree(stateThreadGPU);
	cudaFree(stateIdxGPU);
	cudaFree(saltGPU);
	cudaFree(pwdGPU);

	//Free allKeys
	free(pKeys);
	pKeys = NULL;

	return result;
}


//#if (nPARALLEL > 1)
__device__ uint64_t sizeSlicedRows;
//#endif //nParallel > 1

/**
* Execute G function, with all 12 rounds for Blake2 and  BlaMka, and 24 round for half-round BlaMka.
*
* @param v     A 1024-bit (16 uint64_t) array to be processed by Blake2b's or BlaMka's G function
*/
__device__ inline static void spongeLyra(uint64_t *v) {
	int i;

	for (i = 0; i < 12; i++){
		ROUND_LYRA(i);
	}
}

/**
* Executes a reduced version of G function with only RHO round
* @param v     A 1024-bit (16 uint64_t) array to be processed by Blake2b's or BlaMka's G function
*/
__device__ inline static void reducedSpongeLyra(uint64_t *v) {
	int i;

	for (i = 0; i < RHO; i++){
		ROUND_LYRA(i);
	}
}

/**
* Performs the initial organization of parameters
* And starts the setup phase.
* Initializes the Sponge's State
* Sets the passwords + salt + params and makes the padding
* Absorb this data to the state.
* From setup:
* Initializes M[0]
* Initializes M[1]
* Initializes M[2]
*
* @param memMatrixGPU                  Matrix start
* @param pkeysGPU			The derived keys of each thread
* @param kLen				Desired key length
* @param pwdGPU			User password
* @param pwdlen			Password length
* @param saltGPU			Salt
* @param saltlen			Salt length
* @param timeCost                      Parameter to determine the processing time (T)
* @param nRows				Matrix total number of rows
* @param nCols				Matrix total number of columns
* @param nBlocksInput                  The number of blocks to be absorbed
* @param totalPasswords                Total number of passwords being tested
*/
__global__ void bootStrapGPU(uint64_t * memMatrixGPU, unsigned char * pkeysGPU, unsigned int kLen, unsigned char *pwdGPU, unsigned int pwdlen, unsigned char *saltGPU, unsigned int saltlen, unsigned int timeCost, unsigned int nRows, unsigned int nCols, uint64_t nBlocksInput, unsigned int totalPasswords) {
	int i;
	// Size of each chunk that each thread will work with
	//updates global sizeSlicedRows;
	sizeSlicedRows = (nRows / nPARALLEL) * ROW_LEN_INT64;
	byte *ptrByte;
	byte *ptrByteSource;
	int threadNumber;

	// Thread index:
	threadNumber = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (threadNumber < (nPARALLEL * totalPasswords)) {

		uint64_t sliceStart = threadNumber*sizeSlicedRows;
		uint64_t thStart = ((uint64_t)(threadNumber / nPARALLEL));

		//============= Padding (password + salt + params) with 10*1 ===============//
		//OBS.:The memory matrix will temporarily hold the password: not for saving memory,
		//but this ensures that the password copied locally will be overwritten as soon as possible
		ptrByte = (byte*)& memMatrixGPU[sliceStart];
		ptrByteSource = (byte*)& pwdGPU[thStart * pwdlen];

		//First, we clean enough blocks for the password, salt, params and padding
		for (i = 0; i < nBlocksInput * BLOCK_LEN_BLAKE2_SAFE_BYTES; i++) {
			ptrByte[i] = (byte)0;
		}

		//Prepends the password
		memcpy(ptrByte, ptrByteSource, pwdlen);
		ptrByte += pwdlen;

		//The indexed salt
		ptrByteSource = (byte*)& saltGPU[thStart * saltlen];

		//Concatenates the salt
		memcpy(ptrByte, ptrByteSource, saltlen);
		ptrByte += saltlen;

		//Concatenates the basil: every integer passed as parameter, in the order they are provided by the interface
		memcpy(ptrByte, &kLen, sizeof(int));
		ptrByte += sizeof(int);
		memcpy(ptrByte, &pwdlen, sizeof(int));
		ptrByte += sizeof(int);
		memcpy(ptrByte, &saltlen, sizeof(int));
		ptrByte += sizeof(int);
		memcpy(ptrByte, &timeCost, sizeof(int));
		ptrByte += sizeof(int);
		memcpy(ptrByte, &nRows, sizeof(int));
		ptrByte += sizeof(int);
		memcpy(ptrByte, &nCols, sizeof(int));
		ptrByte += sizeof(int);

#if (nPARALLEL > 1)
		//The difference from sequential version:
		//Concatenates the total number of threads
		int p = nPARALLEL;
		memcpy(ptrByte, &p, sizeof(int));
		ptrByte += sizeof(int);
		//Concatenates thread number
		int thread = threadNumber % nPARALLEL;
		memcpy(ptrByte, &thread, sizeof(int));

		ptrByte += sizeof(int);
#endif //nParallel > 1

		//Now comes the padding
		*ptrByte = 0x80; //first byte of padding: right after the password

		//resets the pointer to the start of the memory matrix
		ptrByte = (byte*)& memMatrixGPU[sliceStart];
		ptrByte += nBlocksInput * BLOCK_LEN_BLAKE2_SAFE_BYTES - 1; //sets the pointer to the correct position: end of incomplete block
		*ptrByte ^= 0x01; //last byte of padding: at the end of the last incomplete block
	}
}

/**
* Initializes the Sponge State. The first 512 bits are set to zeros and the remainder
* receive Blake2b's IV as per Blake2b's specification. <b>Note:</b> Even though sponges
* typically have their internal state initialized with zeros, Blake2b's G function
* has a fixed point: if the internal state and message are both filled with zeros. the
* resulting permutation will always be a block filled with zeros; this happens because
* Blake2b does not use the constants originally employed in Blake2 inside its G function,
* relying on the IV for avoiding possible fixed points.
*
* @param state             The 1024-bit array to be initialized
* @param totalPasswords    Total number of passwords being tested
*/
__global__ void initState(uint64_t state[/*16*/], unsigned int totalPasswords) {
	int threadNumber;
	uint64_t start;

	// Thread index:
	threadNumber = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (threadNumber < (nPARALLEL * totalPasswords)) {

		start = threadNumber * STATESIZE_INT64;
		//First 512 bis are zeros
		state[start + 0] = 0x0ULL;
		state[start + 1] = 0x0ULL;
		state[start + 2] = 0x0ULL;
		state[start + 3] = 0x0ULL;
		state[start + 4] = 0x0ULL;
		state[start + 5] = 0x0ULL;
		state[start + 6] = 0x0ULL;
		state[start + 7] = 0x0ULL;
		//Remainder BLOCK_LEN_BLAKE2_SAFE_BYTES are reserved to the IV
		state[start + 8] = blake2b_IV[0];
		state[start + 9] = blake2b_IV[1];
		state[start + 10] = blake2b_IV[2];
		state[start + 11] = blake2b_IV[3];
		state[start + 12] = blake2b_IV[4];
		state[start + 13] = blake2b_IV[5];
		state[start + 14] = blake2b_IV[6];
		state[start + 15] = blake2b_IV[7];
	}
}

/**
* Performs an absorb operation for a single block (BLOCK_LEN_BLAKE2_SAFE_INT64
* words of type uint64_t), using G function as the internal permutation
*
* @param state         The current state of the sponge
* @param in            The block to be absorbed (BLOCK_LEN_BLAKE2_SAFE_INT64 words)
*/
__device__ inline void absorbBlockBlake2Safe(uint64_t *state, const uint64_t *in) {
	//XORs the first BLOCK_LEN_BLAKE2_SAFE_INT64 words of "in" with the current state
	state[0] ^= in[0];
	state[1] ^= in[1];
	state[2] ^= in[2];
	state[3] ^= in[3];
	state[4] ^= in[4];
	state[5] ^= in[5];
	state[6] ^= in[6];
	state[7] ^= in[7];

	//Applies the transformation f to the sponge's state
	spongeLyra(state);
}

/**
* Performs a initial absorb operation
* Absorbs salt, password and the other parameters
*
* @param memMatrixGPU		Matrix start
* @param stateThreadGPU	The current state of the sponge
* @param stateIdxGPU  		Index of the threads, to be absorbed
* @param nBlocksInput 		The number of blocks to be absorbed
* @param totalPasswords        Total number of passwords being tested
*/
__global__ void absorbInput(uint64_t * memMatrixGPU, uint64_t * stateThreadGPU, uint64_t *stateIdxGPU, uint64_t nBlocksInput, unsigned int totalPasswords) {
	uint64_t *ptrWord;
	uint64_t *threadState;
	int threadNumber;
	uint64_t kP;
	uint64_t sliceStart;

	// Thread index:
	threadNumber = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (threadNumber < (nPARALLEL * totalPasswords)) {

		sliceStart = threadNumber*sizeSlicedRows;
		threadState = (uint64_t *)& stateThreadGPU[threadNumber * STATESIZE_INT64];

		//Absorbing salt, password and params: this is the only place in which the block length is hard-coded to 512 bits, for compatibility with Blake2b and BlaMka
		ptrWord = (uint64_t *)& memMatrixGPU[sliceStart];      //threadSliceMatrix;
		for (kP = 0; kP < nBlocksInput; kP++) {
			absorbBlockBlake2Safe(threadState, ptrWord);        //absorbs each block of pad(pwd || salt || params)
			ptrWord += BLOCK_LEN_BLAKE2_SAFE_INT64;             //BLOCK_LEN_BLAKE2_SAFE_INT64;  //goes to next block of pad(pwd || salt || params)
		}
	}
}

/**
* Performs a reduced squeeze operation for a single row, from the highest to
* the lowest index, using the reduced-round G function as the
* internal permutation
*
* @param state             The current state of the sponge
* @param rowOut            Row to receive the data squeezed
* @param totalPasswords    Total number of passwords being tested
*/
__global__ void reducedSqueezeRow0(uint64_t* rowOut, uint64_t* state, unsigned int totalPasswords) {
	int threadNumber;
	uint64_t sliceStart;
	uint64_t stateStart;

	// Thread index:
	threadNumber = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (threadNumber < (nPARALLEL * totalPasswords)) {
		stateStart = threadNumber * STATESIZE_INT64;
		sliceStart = threadNumber * sizeSlicedRows;

		uint64_t* ptrWord = &rowOut[sliceStart + (N_COLS - 1) * BLOCK_LEN_INT64]; //In Lyra2: pointer to M[0][C-1]
		int i, j;
		//M[0][C-1-col] = H.reduced_squeeze()
		for (i = 0; i < N_COLS; i++) {
			for (j = 0; j < BLOCK_LEN_INT64; j++) {
				ptrWord[j] = state[stateStart + j];
			}

			//Goes to next block (column) that will receive the squeezed data
			ptrWord -= BLOCK_LEN_INT64;

			//Applies the reduced-round transformation f to the sponge's state
			reducedSpongeLyra(&state[stateStart]);
		}
	}
}

/**
* Performs a reduced duplex operation for a single row, from the highest to
* the lowest index of its columns, using the reduced-round G function
* as the internal permutation
*
* @param state		        The current state of the sponge
* @param rowIn		        Matrix start (base row)
* @param first		        Index used with rowIn to calculate wich row will feed the sponge
* @param second	        Index used with rowIn to calculate wich row will be feeded with sponge state
* @param totalPasswords        Total number of passwords being tested
*/
__global__ void reducedDuplexRow1and2(uint64_t *rowIn, uint64_t *state, unsigned int totalPasswords, int first, int second) {
	int i, j;

	int threadNumber;
	uint64_t sliceStart;
	uint64_t stateStart;

	// Thread index:
	threadNumber = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (threadNumber < (nPARALLEL * totalPasswords)) {

		stateStart = threadNumber * STATESIZE_INT64;
		sliceStart = threadNumber * sizeSlicedRows;

		//Row to feed the sponge
		uint64_t* ptrWordIn = (uint64_t*)& rowIn[sliceStart + first * ROW_LEN_INT64]; //In Lyra2: pointer to prev
		//Row to receive the sponge's output
		uint64_t* ptrWordOut = (uint64_t*)& rowIn[sliceStart + second * ROW_LEN_INT64 + (N_COLS - 1) * BLOCK_LEN_INT64]; //In Lyra2: pointer to row

		for (i = 0; i < N_COLS; i++) {

			//Absorbing "M[0][col]"
			for (j = 0; j < BLOCK_LEN_INT64; j++) {
				state[stateStart + j] ^= (ptrWordIn[j]);
			}

			//Applies the reduced-round transformation f to the sponge's state
			reducedSpongeLyra(&state[stateStart]);

			//M[1][C-1-col] = M[0][col] XOR rand
			for (j = 0; j < BLOCK_LEN_INT64; j++) {
				ptrWordOut[j] = ptrWordIn[j] ^ state[stateStart + j];
			}

			//Input: next column (i.e., next block in sequence)
			ptrWordIn += BLOCK_LEN_INT64;
			//Output: goes to previous column
			ptrWordOut -= BLOCK_LEN_INT64;
		}
	}
}

/**
* Performs a duplexing operation over
* "M[rowInOut0][col] [+] M[rowInOut1][col] [+] M[rowIn0][col_0] [+] M[rowIn1][col_1]",
* where [+] denotes wordwise addition, ignoring carries between words. The value of
* "col_0" is computed as "lsw(rot^2(rand)) mod N_COLS", and "col_1" as
* "lsw(rot^3(rand)) mod N_COLS", where lsw() means "the least significant word"
* where rot is a right rotation by 'omega' bits (e.g., 1 or more words)
* N_COLS is a system parameter, and "rand" corresponds
* to the sponge's output for each column absorbed.
* The same output is then employed to make
* "M[rowInOut0][col] = M[rowInOut0][col] XOR rand" and
* "M[rowInOut1][col] = M[rowInOut1][col] XOR rot(rand)".
*
* @param memMatrixGPU          Matrix start
* @param state                 The current state of the sponge
* @param prev0                 Another row used only as input
* @param prev1                 Stores the previous value of row1
* @param row0			        Row used as input and to receive output after rotation
* @param row1			        Pseudorandom indice to a row from another slice, used only as input
* @param totalPasswords        Total number of passwords being tested
*/
__device__ void reducedDuplexRowWandering_P1(uint64_t *memMatrixGPU, uint64_t *state, uint64_t prev0, uint64_t row0, uint64_t row1, uint64_t prev1, unsigned int totalPasswords) {
	int threadNumber;
	uint64_t sliceStart;
	uint64_t stateStart;
	uint64_t randomColumn0; //In Lyra2: col0
	uint64_t randomColumn1; //In Lyra2: col1

	// Thread index:
	threadNumber = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (threadNumber < (nPARALLEL * totalPasswords)) {

		stateStart = threadNumber * STATESIZE_INT64;
		sliceStart = threadNumber * sizeSlicedRows;


		uint64_t* ptrWordInOut0 = (uint64_t *)& memMatrixGPU[sliceStart + (row0 * ROW_LEN_INT64)]; //In Lyra2: pointer to row0
		uint64_t* ptrWordInOut1 = (uint64_t *)& memMatrixGPU[sliceStart + (row1 * ROW_LEN_INT64)]; //In Lyra2: pointer to row0_p
		uint64_t* ptrWordIn0; //In Lyra2: pointer to prev0
		uint64_t* ptrWordIn1; //In Lyra2: pointer to prev1

		int i, j;

		for (i = 0; i < N_COLS; i++) {
			//col0 = lsw(rot^2(rand)) mod N_COLS
			//randomColumn0 = ((uint64_t)state[stateStart + 4] & (N_COLS-1))*BLOCK_LEN_INT64;           /*(USE THIS IF N_COLS IS A POWER OF 2)*/
			randomColumn0 = ((uint64_t)state[stateStart + 4] % N_COLS) * BLOCK_LEN_INT64;              /*(USE THIS FOR THE "GENERIC" CASE)*/
			ptrWordIn0 = (uint64_t *)& memMatrixGPU[sliceStart + (prev0 * ROW_LEN_INT64) + randomColumn0];

			//col0 = LSW(rot^3(rand)) mod N_COLS
			//randomColumn1 = ((uint64_t)state[stateStart + 6] & (N_COLS-1))*BLOCK_LEN_INT64;           /*(USE THIS IF N_COLS IS A POWER OF 2)*/
			randomColumn1 = ((uint64_t)state[stateStart + 6] % N_COLS) * BLOCK_LEN_INT64;              /*(USE THIS FOR THE "GENERIC" CASE)*/
			ptrWordIn1 = (uint64_t *)& memMatrixGPU[sliceStart + (prev1 * ROW_LEN_INT64) + randomColumn1];

			//Absorbing "M[row0] [+] M[row1] [+] M[prev0] [+] M[prev1]"
			for (j = 0; j < BLOCK_LEN_INT64; j++) {
				state[stateStart + j] ^= (ptrWordInOut0[j] + ptrWordInOut1[j] + ptrWordIn0[j] + ptrWordIn1[j]);
			}

			//Applies the reduced-round transformation f to the sponge's state
			reducedSpongeLyra(&state[stateStart]);

			//M[rowInOut0][col] = M[rowInOut0][col] XOR rand
			for (j = 0; j < BLOCK_LEN_INT64; j++) {
				ptrWordInOut0[j] ^= state[stateStart + j];
			}

			//M[rowInOut1][col] = M[rowInOut1][col] XOR rot(rand)
			//rot(): right rotation by 'omega' bits (e.g., 1 or more words)
			//we rotate 2 words for compatibility with the SSE implementation
			for (j = 0; j < BLOCK_LEN_INT64; j++){
				ptrWordInOut1[j] ^= state[stateStart + ((j + 2) % BLOCK_LEN_INT64)];
			}

			//Goes to next block
			ptrWordInOut0 += BLOCK_LEN_INT64;
			ptrWordInOut1 += BLOCK_LEN_INT64;

		}
	}
}

/**
* Performs a duplexing operation over
* "M[rowInOut0][col] [+] M[rowInP][col] [+] M[rowIn0][col_0]",
* where [+] denotes wordwise addition, ignoring carries between words. The value of
* "col_0" is computed as "LSW(rot^3(rand)) mod N_COLS",where LSW means "the less significant word"
* (assuming 64-bit words), rot is a 128-bit  rotation to the right,
* N_COLS is a system parameter, and "rand" corresponds
* to the sponge's output for each column absorbed.
* The same output is then employed to make
* "M[rowInOut0][col] = M[rowInOut0][col] XOR rand".
*
* @param memMatrixGPU          Matrix start
* @param state                 The current state of the sponge
* @param prev0                 Another row used only as input
* @param row0			Row used as input and to receive output after rotation
* @param rowP			Pseudorandom indice to a row from another slice, used only as input
* @param window		Visitation window (equals a half slice)
* @param jP			Index to another slice of matrix
* @param totalPasswords        Total number of passwords being tested
*/
__device__ void reducedDuplexRowWanderingParallel(uint64_t *memMatrixGPU, uint64_t *state, uint64_t prev0, uint64_t row0, uint64_t rowP, uint64_t window, uint64_t jP, unsigned int totalPasswords) {
	int threadNumber;
	uint64_t sliceStart;
	uint64_t stateStart;
	uint64_t sliceStartjP;
	uint64_t randomColumn0; //In Lyra2: col0

	// Thread index:
	threadNumber = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (threadNumber < (nPARALLEL * totalPasswords)) {

		stateStart = threadNumber * STATESIZE_INT64;
		sliceStart = threadNumber * sizeSlicedRows;

		//jP slice must be inside the  password´s thread pool
		//The integer part of threadNumber/nPARALLEL multiplied by nPARALLEL is the Base Slice Start for the password thread pool
		sliceStartjP = ((((uint64_t)(threadNumber / nPARALLEL)) * nPARALLEL) + jP) * sizeSlicedRows;

		//Row used as input and to receive output after rotation
		uint64_t* ptrWordInOut0 = (uint64_t *)& memMatrixGPU[sliceStart + (row0 * ROW_LEN_INT64)]; //In Lyra2: pointer to row0
		//Row used only as input
		uint64_t* ptrWordInP = (uint64_t *)& memMatrixGPU[sliceStartjP + (rowP * ROW_LEN_INT64)]; //In Lyra2: pointer to row0_p
		//Another row used only as input
		uint64_t* ptrWordIn0; //In Lyra2: pointer to prev0

		int i, j;

		for (i = 0; i < N_COLS; i++) {
			//col0 = LSW(rot^3(rand)) mod N_COLS
			//randomColumn0 = ((uint64_t)state[stateStart + 6] & (N_COLS-1))*BLOCK_LEN_INT64;           /*(USE THIS IF N_COLS IS A POWER OF 2)*/
			randomColumn0 = ((uint64_t)state[stateStart + 6] % N_COLS) * BLOCK_LEN_INT64;              /*(USE THIS FOR THE "GENERIC" CASE)*/

			ptrWordIn0 = (uint64_t *)& memMatrixGPU[sliceStart + (prev0 * ROW_LEN_INT64) + randomColumn0];

			//Absorbing "M[row0] [+] M[prev0] [+] M[rowP]"
			for (j = 0; j < BLOCK_LEN_INT64; j++) {
				state[stateStart + j] ^= (ptrWordInOut0[j] + ptrWordIn0[j] + ptrWordInP[j]);
			}

			//Applies the reduced-round transformation f to the sponge's state
			reducedSpongeLyra(&state[stateStart]);

			//M[rowInOut0][col] = M[rowInOut0][col] XOR rand
			for (j = 0; j < BLOCK_LEN_INT64; j++) {
				ptrWordInOut0[j] ^= state[stateStart + j];
			}

			//Goes to next block
			ptrWordInOut0 += BLOCK_LEN_INT64;
			ptrWordInP += BLOCK_LEN_INT64;

		}
	}
}

/**
* Performs an absorb operation of single column from "in", the
* said column being pseudorandomly picked in the range [0, BLOCK_LEN_INT64[,
* using the full-round G function as the internal permutation
*
* @param state The current state of the sponge
* @param in    			Matrix start
* @param row0				The row whose column (BLOCK_LEN_INT64 words) should be absorbed
* @param randomColumn0                 The random column to be absorbed
* @param totalPasswords                Total number of passwords being tested
*/
__device__ void absorbRandomColumn(uint64_t *in, uint64_t *state, uint64_t row0, uint64_t randomColumn0, unsigned int totalPasswords) {
	int i;
	int threadNumber;
	uint64_t sliceStart;
	uint64_t stateStart;

	// Thread index:
	threadNumber = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (threadNumber < (nPARALLEL * totalPasswords)) {

		stateStart = threadNumber * STATESIZE_INT64;
		sliceStart = threadNumber * sizeSlicedRows;

		uint64_t* ptrWordIn = (uint64_t*)& in[sliceStart + (row0 * ROW_LEN_INT64) + randomColumn0];

		//absorbs the column picked
		for (i = 0; i < BLOCK_LEN_INT64; i++) {
			state[stateStart + i] ^= ptrWordIn[i];
		}

		//Applies the full-round transformation f to the sponge's state
		spongeLyra(&state[stateStart]);
	}
}

/**
* Wandering phase: performs the visitation loop
* Visitation loop chooses pseudo random rows (row0 and row1) based in state content
* And performs a reduced-round duplexing operation over:
* "M[row0][col] [+] M[row1][col] [+] M[prev0][col0] [+] M[prev1][col1]
* Updating both M[row0] and M[row1] using the output to make:
* M[row0][col] = M[row0][col] XOR rand;
* M[row1][col] = M[row1][col] XOR rot(rand)
* Where rot() is a right rotation by 'omega' bits (e.g., 1 or more words)
*
* @param stateThreadGPU      	The current state of the sponge
* @param memMatrixGPU          Array that will receive the data squeezed
* @param timeCost            	Parameter to determine the processing time (T)
* @param nRows         		Number of rows
* @param totalPasswords        Total number of passwords being tested
* @param prev0                 Stores the previous value of row0, the last row ever initialized
* @param prev1                 Stores the previous value of row1
*/
__device__ void wanderingPhaseGPU2_P1(uint64_t * memMatrixGPU, uint64_t * stateThreadGPU, unsigned int timeCost, uint64_t nRows, unsigned int totalPasswords, uint64_t prev0, uint64_t prev1) {
	uint64_t wCont;             //Time Loop iterator
	uint64_t row0;              //row0: sequentially written during Setup; randomly picked during Wandering
	uint64_t row1;              //rowP: revisited during Setup, and then read [and written]; randomly picked during Wandering
	uint64_t threadNumber;

	uint64_t stateStart;


	// Thread index:
	threadNumber = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (threadNumber < (nPARALLEL * totalPasswords)) {

		stateStart = threadNumber * STATESIZE_INT64;

		for (wCont = 0; wCont < timeCost * nRows; wCont++) {
			//Selects a pseudorandom indices row0 and rowP (row0 = LSW(rand) mod wnd and rowP = LSW(rot(rand)) mod wnd)
			//------------------------------------------------------------------------------------------
			//(USE THIS IF window IS A POWER OF 2)
			//row0 = (((uint64_t)stateThreadGPU[stateStart + 0]) & nRows);
			//row1 = (((uint64_t)stateThreadGPU[stateStart + 2]) & nRows);
			//(USE THIS FOR THE "GENERIC" CASE)
			row0 = (((uint64_t)stateThreadGPU[stateStart + 0]) % nRows);   //row0 = lsw(rand) mod nRows
			row1 = (((uint64_t)stateThreadGPU[stateStart + 2]) % nRows);   //row1 = lsw(rot(rand)) mod nRows
			//we rotate 2 words for compatibility with the SSE implementation

			//Performs a reduced-round duplexing operation over "M[row0][col] [+] M[row1][col] [+] M[prev0][col0] [+] M[prev1][col1], updating both M[row0] and M[row1]
			//M[row0][col] = M[row0][col] XOR rand;
			//M[row1][col] = M[row1][col] XOR rot(rand)                     rot(): right rotation by 'omega' bits (e.g., 1 or more words)
			reducedDuplexRowWandering_P1(memMatrixGPU, stateThreadGPU, prev0, row0, row1, prev1, totalPasswords);

			//update prev: they now point to the last rows ever updated
			prev0 = row0;
			prev1 = row1;

		}

		//============================ Wrap-up Phase ===============================//
		//Absorbs one last block of the memory matrix with the full-round sponge
		absorbRandomColumn(memMatrixGPU, stateThreadGPU, row0, 0, totalPasswords);
	}

}

/**
* Wandering phase: performs the visitation loop
* Visitation loop chooses pseudo random rows (row0 and rowP) based in state content
* And performs a reduced-round duplexing operation over:
* M[row0] [+] Mj[rowP] [+] M[prev0]
* Updating M[row0] using the output from reduced-round duplexing (rand):
* M[row0][col] = M[row0][col] XOR rand
*
* @param stateThreadGPU      	The current state of the sponge
* @param memMatrixGPU          Array that will receive the data squeezed
* @param timeCost        	Parameter to determine the processing time (T)
* @param sizeSlice		Number of rows for each thread
* @param totalPasswords        Total number of passwords being tested
* @param sqrt                  To control step changes in visitation
* @param prev0                 Stores the previous value of row0, the last row ever initialized
*/
__device__ void wanderingPhaseGPU2(uint64_t * memMatrixGPU, uint64_t * stateThreadGPU, unsigned int timeCost, uint64_t sizeSlice, unsigned int totalPasswords, uint64_t sqrt, uint64_t prev0) {
	uint64_t wCont;             //Time Loop iterator
	uint64_t window;            //Visitation window (used to define which rows can be revisited during Setup)
	uint64_t row0;              //row0: sequentially written during Setup; randomly picked during Wandering

	uint64_t rowP;              //rowP: revisited during Setup, and then read [and written]; randomly picked during Wandering
	uint64_t jP;                //Index to another thread
	uint64_t threadNumber;

	uint64_t stateStart;

	uint64_t off0;              //complementary offsets to calculate row0
	uint64_t offP;              //complementary offsets to calculate rowP
	uint64_t offTemp;

	uint64_t sync = sqrt;

	uint64_t halfSlice = sizeSlice / 2;

	// Thread index:
	threadNumber = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (threadNumber < (nPARALLEL * totalPasswords)) {

		stateStart = threadNumber * STATESIZE_INT64;

		window = halfSlice;
		off0 = 0;
		offP = window;

		for (wCont = 0; wCont < timeCost * sizeSlice; wCont++) {
			//Selects a pseudorandom indices row0 and rowP (row0 = LSW(rand) mod wnd and rowP = LSW(rot(rand)) mod wnd)
			//------------------------------------------------------------------------------------------
			//(USE THIS IF window IS A POWER OF 2)
			//row0  = off0 + (((uint64_t)stateThreadGPU[stateStart + 0]) & (window-1));
			//row0P = offP + (((uint64_t)stateThreadGPU[stateStart + 2]) & (window-1));
			//(USE THIS FOR THE "GENERIC" CASE)
			row0 = off0 + (((uint64_t)stateThreadGPU[stateStart + 0]) % window);
			rowP = offP + (((uint64_t)stateThreadGPU[stateStart + 2]) % window);

			//Selects a pseudorandom indices j0 (LSW(rot^2 (rand)) mod p)
			jP = ((uint64_t)stateThreadGPU[stateStart + 4]) % nPARALLEL;

			//Performs a reduced-round duplexing operation over M[row0] [+] Mj[rowP] [+] M[prev0], updating M[row0]
			//M[row0][col] = M[row0][col] XOR rand;
			reducedDuplexRowWanderingParallel(memMatrixGPU, stateThreadGPU, prev0, row0, rowP, window, jP, totalPasswords);

			//update prev: they now point to the last rows ever updated
			prev0 = row0;

			if (wCont == sync) {
				sync += sqrt;
				offTemp = off0;
				off0 = offP;
				offP = offTemp;
				__syncthreads();
			}
		}
		__syncthreads();

		//============================ Wrap-up Phase ===============================//
		//Absorbs one last block of the memory matrix with the full-round sponge
		absorbRandomColumn(memMatrixGPU, stateThreadGPU, row0, 0, totalPasswords);
	}

}

/**
* Performs a duplexing operation over
* "M[rowInOut][col] [+] M[rowIn0][col] [+] M[rowIn1][col]", where [+] denotes
* wordwise addition, ignoring carries between words, , for all values of "col"
* in the [0,N_COLS[ interval. The  output of this operation, "rand", is then
* employed to make
* "M[rowOut][(N_COLS-1)-col] = M[rowIn0][col] XOR rand" and
* "M[rowInOut][col] =  M[rowInOut][col] XOR rot(rand)",
* where rot is a right rotation by 'omega' bits (e.g., 1 or more words)
* and N_COLS is a system parameter.
*
* @param state                         The current state of the sponge
* @param memMatrixGPU                  Matrix start
* @param prev0                         Index to calculate rowIn0, the previous row0
* @param prev1                         Index to calculate rowIn1
* @param row0                          Index to calculate rowOut, the row being initialized
* @param row1                          Index to calculate rowInOut, the row to be revisited and updated
* @param totalPasswords                Total number of passwords being tested
*/
__device__ void reducedDuplexRowFilling_P1(uint64_t *state, uint64_t *memMatrixGPU, uint64_t prev0, uint64_t prev1, uint64_t row0, uint64_t row1, unsigned int totalPasswords) {
	int i, j;
	int threadNumber;

	uint64_t sliceStart;
	uint64_t stateStart;

	// Thread index:
	threadNumber = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (threadNumber < (nPARALLEL * totalPasswords)) {
		stateStart = threadNumber * STATESIZE_INT64;
		sliceStart = threadNumber * sizeSlicedRows;     //sizeSlicedRows = (nRows/nPARALLEL) * ROW_LEN_INT64

		//Row used only as input (rowIn0 or M[prev0])
		uint64_t* ptrWordIn0 = (uint64_t *)& memMatrixGPU[sliceStart + prev0 * ROW_LEN_INT64];         //In Lyra2: pointer to prev0, the last row ever initialized

		//Another row used only as input (rowIn1 or M[prev1])
		uint64_t* ptrWordIn1 = (uint64_t *)& memMatrixGPU[sliceStart + prev1 * ROW_LEN_INT64];     //In Lyra2: pointer to prev1, the last row ever revisited and updated

		//Row used as input and to receive output after rotation (rowInOut or M[row1])
		uint64_t* ptrWordInOut = (uint64_t *)& memMatrixGPU[sliceStart + row1 * ROW_LEN_INT64];    //In Lyra2: pointer to row1, to be revisited and updated

		//Row receiving the output (rowOut or M[row0])
		uint64_t* ptrWordOut = (uint64_t *)& memMatrixGPU[sliceStart + (row0 * ROW_LEN_INT64) + ((N_COLS - 1) * BLOCK_LEN_INT64)]; //In Lyra2: pointer to row0, to be initialized

		for (i = 0; i < N_COLS; i++) {
			//Absorbing "M[row1] [+] M[prev0] [+] M[prev1]"
			for (j = 0; j < BLOCK_LEN_INT64; j++) {
				state[stateStart + j] ^= (ptrWordInOut[j] + ptrWordIn0[j] + ptrWordIn1[j]);
			}

			//Applies the reduced-round transformation f to the sponge's state
			reducedSpongeLyra(&state[stateStart]);

			//M[row0][col] = M[prev0][col] XOR rand
			for (j = 0; j < BLOCK_LEN_INT64; j++) {
				ptrWordOut[j] = ptrWordIn0[j] ^ state[stateStart + j];
			}

			//M[row1][col] = M[row1][col] XOR rot(rand)
			//rot(): right rotation by 'omega' bits (e.g., 1 or more words)
			//we rotate 2 words for compatibility with the SSE implementation
			for (j = 0; j < BLOCK_LEN_INT64; j++) {
				ptrWordInOut[j] ^= state[stateStart + ((j + 2) % BLOCK_LEN_INT64)]; // BLOCK_LEN_INT64 = 12
			}

			//Inputs: next column (i.e., next block in sequence)
			ptrWordInOut += BLOCK_LEN_INT64;
			ptrWordIn0 += BLOCK_LEN_INT64;
			ptrWordIn1 += BLOCK_LEN_INT64;
			//Output: goes to previous column
			ptrWordOut -= BLOCK_LEN_INT64;
		}
	}
}



/**
* Performs a duplexing operation over
* "M[rowInOut][col] [+] M[rowIn0][col] [+] M[rowIn1][col]", where [+] denotes
* wordwise addition, ignoring carries between words, , for all values of "col"
* in the [0,N_COLS[ interval. The  output of this operation, "rand", is then
* employed to make
* "M[rowOut][(N_COLS-1)-col] = M[rowIn0][col] XOR rand" and
* "M[rowInOut][col] =  M[rowInOut][col] XOR rot(rand)",
* where rot is a right rotation by 'omega' bits (e.g., 1 or more words)
* and N_COLS is a system parameter.
*
* @param state                         The current state of the sponge
* @param memMatrixGPU                  Matrix start
* @param prev0                         Index to calculate rowIn0, the previous row0
* @param prevP                         Index to calculate rowIn1
* @param row0                          Index to calculate rowOut, the row being initialized
* @param rowP                          Index to calculate rowInOut, the row to be revisited and updated
* @param jP                            Index to another slice of matrix (slice belonging to another thread)
* @param totalPasswords                Total number of passwords being tested
*/
__device__ void reducedDuplexRowFilling(uint64_t *state, uint64_t *memMatrixGPU, uint64_t prev0, uint64_t prevP, uint64_t row0, uint64_t rowP, uint64_t jP, unsigned int totalPasswords) {
	int i, j;
	int threadNumber;

	uint64_t sliceStart;
	uint64_t sliceStartjP;
	uint64_t stateStart;

	// Thread index:
	threadNumber = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (threadNumber < (nPARALLEL * totalPasswords)) {
		stateStart = threadNumber * STATESIZE_INT64;
		sliceStart = threadNumber * sizeSlicedRows;     //sizeSlicedRows = (nRows/nPARALLEL) * ROW_LEN_INT64
		//jP slice must be inside the  password´s thread pool
		//The integer part of threadNumber/nPARALLEL multiplied by nPARALLEL is the Base Slice Start for the password thread pool
		sliceStartjP = ((((uint64_t)(threadNumber / nPARALLEL)) * nPARALLEL) + jP) * sizeSlicedRows;

		//Row used only as input
		uint64_t* ptrWordIn0 = (uint64_t *)& memMatrixGPU[sliceStart + prev0 * ROW_LEN_INT64];         //In Lyra2: pointer to prev0, the last row ever initialized

		//Another row used only as input
		uint64_t* ptrWordIn1 = (uint64_t *)& memMatrixGPU[sliceStartjP + (prevP * ROW_LEN_INT64)];     //In Lyra2: pointer to prev1, the last row ever revisited and updated

		//Row used as input and to receive output after rotation
		uint64_t* ptrWordInOut = (uint64_t *)& memMatrixGPU[sliceStartjP + (rowP * ROW_LEN_INT64)];    //In Lyra2: pointer to row1, to be revisited and updated

		//Row receiving the output
		uint64_t* ptrWordOut = (uint64_t *)& memMatrixGPU[sliceStart + (row0 * ROW_LEN_INT64) + ((N_COLS - 1) * BLOCK_LEN_INT64)]; //In Lyra2: pointer to row0, to be initialized

		for (i = 0; i < N_COLS; i++) {
			//Absorbing "M[rowP] [+] M[prev0] [+] M[prev1]"
			for (j = 0; j < BLOCK_LEN_INT64; j++) {
				state[stateStart + j] ^= (ptrWordInOut[j] + ptrWordIn0[j] + ptrWordIn1[j]);
			}

			//Applies the reduced-round transformation f to the sponge's state
			reducedSpongeLyra(&state[stateStart]);

			//M[row0][col] = M[prev0][col] XOR rand
			for (j = 0; j < BLOCK_LEN_INT64; j++) {
				ptrWordOut[j] = ptrWordIn0[j] ^ state[stateStart + j];
			}

			//M[rowP][col] = M[rowP][col] XOR rot(rand)
			//rot(): right rotation by 'omega' bits (e.g., 1 or more words)
			//we rotate 2 words for compatibility with the SSE implementation
			for (j = 0; j < BLOCK_LEN_INT64; j++) {
				ptrWordInOut[j] ^= state[stateStart + ((j + 2) % BLOCK_LEN_INT64)]; // BLOCK_LEN_INT64 = 12
			}

			//Inputs: next column (i.e., next block in sequence)
			ptrWordInOut += BLOCK_LEN_INT64;
			ptrWordIn0 += BLOCK_LEN_INT64;
			ptrWordIn1 += BLOCK_LEN_INT64;
			//Output: goes to previous column
			ptrWordOut -= BLOCK_LEN_INT64;
		}
	}
}

/**
* Performs matrix initialization and calls wandering phase
* During setup, performs a reduced-round duplexing operation over:
* "Mj[rowP][col] [+] Mi[prev0][col] [+] Mj[prevP][col]", filling Mi[row0] and updating Mj[rowP]
* M[row0][N_COLS-1-col] = M[prev0][col] XOR rand;
* Mj[rowP][col] = Mj[rowP][col] XOR rot(rand)
* Where rot() is a right rotation by 'omega' bits (e.g., 1 or more words)
* and N_COLS is a system parameter.
*
* @param memMatrixGPU		Matrix start
* @param stateThreadGPU	The current state of the sponge
* @param sizeSlice		Number of rows for each thread
* @param totalPasswords        Total number of passwords being tested
* @param timeCost        	Parameter to determine the processing time (T)
*/
__global__ void setupPhaseWanderingGPU(uint64_t * memMatrixGPU, uint64_t * stateThreadGPU, uint64_t sizeSlice, unsigned int totalPasswords, unsigned int timeCost) {
	uint64_t step = 1;          //Visitation step (used during Setup and Wandering phases)
	uint64_t window = 2;        //Visitation window (used to define which rows can be revisited during Setup)
	int64_t gap = 1;            //Modifier to the step, assuming the values 1 or -1

	uint64_t row0 = 3;          //row0: sequentially written during Setup; randomly picked during Wandering
	uint64_t prev0 = 2;         //prev0: stores the previous value of row0
	uint64_t rowP = 1;          //rowP: revisited during Setup, and then read [and written]; randomly picked during Wandering
	uint64_t prevP = 0;         //prevP: stores the previous value of rowP
	uint64_t jP;                //Index to another thread, starts with threadNumber
	uint64_t sync = 4;          //Synchronize counter
	uint64_t sqrt = 2;          //Square of window (i.e., square(window)), when a window is a square number;
	//otherwise, sqrt = 2*square(window/2)

	int threadNumber;

	// Thread index:
	threadNumber = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (threadNumber < (nPARALLEL * totalPasswords)) {

		//jP must be in the thread pool of the same password
		jP = threadNumber % nPARALLEL;

		//Filling Loop
		for (row0 = 3; row0 < sizeSlice; row0++) {
			//Performs a reduced-round duplexing operation over "Mj[rowP][col] [+] Mi[prev0][col] [+] Mj[prevP][col]", filling Mi[row0] and updating Mj[rowP]
			//Mi[row0][N_COLS-1-col] = Mi[prev0][col] XOR rand;
			//Mj[rowP][col] = Mj[rowP][col] XOR rot(rand)                    rot(): right rotation by 'omega' bits (e.g., 1 or more words)
			reducedDuplexRowFilling(stateThreadGPU, memMatrixGPU, prev0, prevP, row0, rowP, jP, totalPasswords);

			//Updates the "prev" indices: the rows more recently updated
			prev0 = row0;
			prevP = rowP;

			//updates the value of row1: deterministically picked, with a variable step
			rowP = (rowP + step) & (window - 1);

			//Checks if all rows in the window where visited.
			if (rowP == 0) {
				window *= 2;            //doubles the size of the re-visitation window
				step = sqrt + gap;      //changes the step
				gap = -gap;             //inverts the modifier to the step
				if (gap == -1) {
					sqrt *= 2;          //Doubles sqrt every other iteration
				}
			}
			if (row0 == sync) {
				sync += sqrt / 2;               //increment synchronize counter
				jP = (jP + 1) % nPARALLEL;      //change the visitation thread
				__syncthreads();
			}
		}

		//Waits all threads
		__syncthreads();

		//Now goes to Wandering Phase and the Absorb from Wrap-up
		//============================ Wandering Phase =============================//
		//=====Iteratively overwrites pseudorandom cells of the memory matrix=======//
		wanderingPhaseGPU2(memMatrixGPU, stateThreadGPU, timeCost, sizeSlice, totalPasswords, sqrt, prev0);

	}
}

/**
* Performs a squeeze operation, using G function as the
* internal permutation
*
* @param state             The current state of the sponge
* @param out               Array that will receive the data squeezed
* @param len               The number of bytes to be squeezed into the "out" array
* @param totalPasswords    Total number of passwords being tested
*/
__global__ void squeeze(uint64_t *state, byte *out, unsigned int len, unsigned int totalPasswords) {
	int i;
	int fullBlocks = len / BLOCK_LEN_BYTES;

	int threadNumber;
	uint64_t stateStart;

	// Thread index:
	threadNumber = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (threadNumber < (nPARALLEL * totalPasswords)) {

		stateStart = threadNumber * STATESIZE_INT64;

		byte *ptr = (byte *)& out[threadNumber * len];

		//Squeezes full blocks
		for (i = 0; i < fullBlocks; i++) {
			memcpy(ptr, &state[stateStart], BLOCK_LEN_BYTES);
			spongeLyra(&state[stateStart]);
			ptr += BLOCK_LEN_BYTES;
		}

		//Squeezes remaining bytes
		memcpy(ptr, &state[stateStart], (len % BLOCK_LEN_BYTES));
	}
}


/**
* Generates the passwords for Lyra2 attack.
*
* @param t_cost            Parameter to determine the processing time (T)
* @param m_cost            Memory cost parameter (defines the number of rows of the memory matrix, R)
* @param totalPasswords    Total number of passwords being tested
* @param gridSize          GPU grid configuration
* @param blockSize         GPU block configuration
* @param printKeys         Defines if the resulting keys will be in the output
*/
void multPasswordCUDA(unsigned int t_cost, unsigned int m_cost, unsigned int totalPasswords, unsigned int gridSize, unsigned int blockSize, unsigned int printKeys) {
	//=================== Basic variables, with default values =======================//
	int kLen = 32;
	unsigned char *ptrChar;
	int pwdLen = 10;
	int saltLen = 10;
	int i, j;
	int result;
	//==========================================================================/

	if (m_cost / nPARALLEL < 4) {
		printf("Number of rows too small\n");
		exit(0);
	}

	size_t sizeMemMatrix = (size_t)((size_t)m_cost * (size_t)ROW_LEN_BYTES);

	printf("Total time cost: %d \n", t_cost);
	printf("Total number of rows: %d \n", m_cost);
	printf("Total number of cols: %d \n", N_COLS);
	char *spongeName = "";
	spongeName = "Blake2";
	printf("Sponge: %s\n", spongeName);
	printf("Total number of password: %d \n", totalPasswords);
	printf("Password length: %d \n", pwdLen);
	printf("Parallelism inside password derivation: %d \n", nPARALLEL);
	printf("Grid Size (blocks): %d\n", gridSize);
	printf("Block Size (threads): %d\n", blockSize);
	printf("BlockSize x GridSize (threads): %d\n", gridSize*blockSize);
	printf("Total number of threads: %d \n", nPARALLEL*totalPasswords);
	printf("Memory per password: %ld bytes (%ld MB)\n", (long int)sizeMemMatrix, (long int)(sizeMemMatrix) / (1024 * 1024));
	printf("Total Memory: %ld bytes (%ld MB)\n", (long int)sizeMemMatrix * totalPasswords, (long int)(sizeMemMatrix * totalPasswords) / (1024 * 1024));
	fflush(stdout);

	// All Keys:
	unsigned char *K = (unsigned char *)malloc(totalPasswords * kLen * sizeof(unsigned char));

	//Pointer to each passwords in the Matrix:
	unsigned char **passwords = (unsigned char **)malloc(totalPasswords * sizeof(unsigned char *));
	if (passwords == NULL) {
		printf("Memory allocation error in file: %s and line: %d\n", __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Matrix with all passwords:
	unsigned char *passwdMatrix = (unsigned char *)malloc(totalPasswords * pwdLen * sizeof(unsigned char));
	if (passwdMatrix == NULL) {
		printf("Memory allocation error in file: %s and line: %d\n", __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Pointer to each salt in the Matrix:
	unsigned char **salts = (unsigned char **)malloc(totalPasswords * sizeof(unsigned char *));
	if (salts == NULL) {
		printf("Memory allocation error in file: %s and line: %d\n", __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Matrix with all salts:
	unsigned char *saltMatrix = (unsigned char *)malloc(totalPasswords * saltLen * sizeof(unsigned char));
	if (saltMatrix == NULL) {
		printf("Memory allocation error in file: %s and line: %d\n", __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Places the pointers in the correct positions
	ptrChar = passwdMatrix;
	for (i = 0; i < totalPasswords; i++) {
		passwords[i] = ptrChar;
		ptrChar += pwdLen; // pwdLen * sizeof (unsigned char);
	}

	//Places the pointers in the correct positions
	ptrChar = saltMatrix;
	for (i = 0; i < totalPasswords; i++) {
		salts[i] = ptrChar;
		ptrChar += saltLen; // pwdLen * sizeof (unsigned char);
	}


#ifndef SAMEPASSWORD
#define SAMEPASSWORD 0
#endif
	//fills passwords
	for (i = 0; i < totalPasswords; i++) {
		for (j = 0; j < pwdLen; j++) {
#if SAMEPASSWORD == 1
			//Same password:
			passwords[i][j] = (0x30 + j);
#else
			//Different passwords:
			passwords[i][j] = (j + i*pwdLen) % 255;
#endif
		}
	}
	// fills salts
	for (i = 0; i < totalPasswords; i++) {
		for (j = 0; j < saltLen; j++) {
			salts[i][j] = (0x30 + j);
		}
	}

	/*
	printf("Number of Passwords: %d\n", totalPasswords);
	//Prints passwords
	printf("Passwords:\n");
	for (i = 0; i < totalPasswords; i++) {
	for (j = 0; j < pwdLen; j++) {
	printf("%2x|", passwords[i][j]);
	}
	printf("\n");
	}

	//Prints salts
	printf("Salts:\n");
	for (i = 0; i < totalPasswords; i++) {
	for (j = 0; j < saltLen; j++) {
	printf("%x|", salts[i][j]);
	}
	printf("\n");
	}
	*/

	struct timeval start;
	struct timeval end;
	gettimeofday(&start, NULL);

	//Calls the interface to the GPU program
	result = gpuMult(K, kLen, passwords, pwdLen, salts, saltLen, t_cost, m_cost, N_COLS, totalPasswords, gridSize, blockSize);

	gettimeofday(&end, NULL);
	unsigned long elapsed = (end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec;

	if (result >= 0){
		//Prints returned keys
		if (printKeys == 1) {
			printf("Result of %d Keys:\n", totalPasswords);
			for (i = 0; i < totalPasswords; i++) {
				printf("Key #: %3d: ", i);
				for (j = 0; j < kLen; j++) {
					printf("%2x|", K[i*kLen + j]);
				}
				printf("\n");
			}
		}
	}

	if (result < 0) {
		printf("Execution Error!!!\n");
	}
	else {
		printf("Execution Time: %lu us (%.3f ms, %.3f seg)\n", elapsed, (float)elapsed / 1000, (float)elapsed / (1000 * 1000));
		printf("Execution Time per password: %.3f us (%.3f ms, %.3f seg)\n", (float)((float)elapsed / totalPasswords), (float)(((float)elapsed / totalPasswords) / 1000), (float)(((float)elapsed / totalPasswords) / (1000 * 1000)));
	}
	printf("------------------------------------------------------------------------------------------------------------------------------------------\n");

	cudaDeviceReset();
	free(passwords);
	free(passwdMatrix);
	free(saltMatrix);
	free(salts);
	free(K);
}


int main(int argc, char *argv[]) {

	//=================== Basic variables, with default values =======================//
	unsigned int t_cost = 0;
	unsigned int m_cost = 0;
	unsigned int gridSize;
	unsigned int blockSize;
	unsigned int numberPasswds;
	//==========================================================================/

	//	Defines in which GPU will execute
	cudaSetDevice(0);

	switch (argc) {
	case 2:
		if (strcmp(argv[1], "--help") == 0) {
			printf("Usage: \n");
			printf("%s tCost nRows --multPasswordCUDA totalPasswordsToTest totalBlocksToUse threadsPerBlock [optional print hash] (to test multiple GPU derivations in parallel)\n\n", argv[0]);
			return 0;
		}
		else {
			printf("Invalid options.\nFor more information, try \"%s --help\".\n", argv[0]);
			return 0;
		}

	case 7:
		if (strcmp(argv[3], "--multPasswordCUDA") == 0) {
			t_cost = atoi(argv[1]);
			m_cost = atoi(argv[2]);
			numberPasswds = atoi(argv[4]);
			gridSize = atoi(argv[5]);
			blockSize = atoi(argv[6]);
			multPasswordCUDA(t_cost, m_cost, numberPasswds, gridSize, blockSize, 0);
			return 0;
		}
		break;

	case 8:
		if (strcmp(argv[3], "--multPasswordCUDA") == 0) {
			t_cost = atoi(argv[1]);
			m_cost = atoi(argv[2]);
			numberPasswds = atoi(argv[4]);
			gridSize = atoi(argv[5]);
			blockSize = atoi(argv[6]);
			multPasswordCUDA(t_cost, m_cost, numberPasswds, gridSize, blockSize, 1);
			return 0;
		}
		break;
	default:
		printf("Invalid options.\nTry \"%s --help\" for help.\n", argv[0]);
		return 0;
	}
}
