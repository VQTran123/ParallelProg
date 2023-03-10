/*********************************************************************/
//
// 02/02/2023: Revised Version for 32M bit adder with 32 bit blocks
//
/*********************************************************************/

#include "main.h"

#include <iostream>
#include <math.h>

//Touch these defines
#define input_size 8388608 // hex digits 
#define block_size 32
#define verbose 0

//Do not touch these defines
#define digits (input_size+1)
#define bits (digits*4)
#define ngroups bits/block_size
#define nsections ngroups/block_size
#define nsupersections nsections/block_size
#define nsupersupersections nsupersections/block_size

//Global definitions of the various arrays used in steps for easy access
/***********************************************************************************************************/
// ADAPT AS CUDA managedMalloc memory - e.g., change to pointers and allocate in main function. 
/***********************************************************************************************************/
// bits
int * gi;
int * pi;
int * ci;

// ngroups
int * ggj;
int * gpj;
int * gcj;

// nsections
int * sgk;
int * spk;
int * sck;

// nsupersections
int * ssgl;
int * sspl;
int * sscl;

// nsupersupersections
int * sssgm;
int * ssspm;
int * ssscm;

// bits
int * sumi;
int * sumrca;

//Integer array of inputs in binary form
int* bin1=NULL;
int* bin2=NULL;

//Character array of inputs in hex form
char* hex1=NULL;
char* hex2=NULL;

void read_input()
{
  char* in1 = (char *)calloc(input_size+1, sizeof(char));
  char* in2 = (char *)calloc(input_size+1, sizeof(char));

  if( 1 != scanf("%s", in1))
    {
      printf("Failed to read input 1\n");
      exit(-1);
    }
  if( 1 != scanf("%s", in2))
    {
      printf("Failed to read input 2\n");
      exit(-1);
    }
  
  hex1 = grab_slice_char(in1,0,input_size+1);
  hex2 = grab_slice_char(in2,0,input_size+1);
  
  free(in1);
  free(in2);
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_gp()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    gi[index] = bin1[index] & bin2[index];
    pi[index] = bin1[index] | bin2[index];
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_group_gp()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int jstart = index*block_size;
    int* ggj_group = grab_slice(gi,jstart,block_size);
    int* gpj_group = grab_slice(pi,jstart,block_size);

    int sum = 0;
    for(int i = 0; i < block_size; i++)
    {
        int mult = ggj_group[i]; //grabs the g_i term for the multiplication
        for(int ii = block_size-1; ii > i; ii--)
        {
            mult &= gpj_group[ii]; //grabs the p_i terms and multiplies it with the previously multiplied stuff (or the g_i term if first round)
        }
        sum |= mult; //sum up each of these things with an or
    }
    ggj[index] = sum;

    int mult = gpj_group[0];
    for(int i = 1; i < block_size; i++)
    {
        mult &= gpj_group[i];
    }
    gpj[index] = mult;

    // free from grab_slice allocation
    free(ggj_group);
    free(gpj_group);
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_section_gp()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int kstart = index*block_size;
    int* sgk_group = grab_slice(ggj,kstart,block_size);
    int* spk_group = grab_slice(gpj,kstart,block_size);

    int sum = 0;
    for(int i = 0; i < block_size; i++)
    {
    int mult = sgk_group[i];
    for(int ii = block_size-1; ii > i; ii--)
        {
        mult &= spk_group[ii];
        }
    sum |= mult;
    }
    sgk[index] = sum;

    int mult = spk_group[0];
    for(int i = 1; i < block_size; i++)
    {
    mult &= spk_group[i];
    }
    spk[index] = mult;

    // free from grab_slice allocation
    free(sgk_group);
    free(spk_group);
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_super_section_gp()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int lstart = index*block_size;
    int* ssgl_group = grab_slice(sgk,lstart,block_size);
    int* sspl_group = grab_slice(spk,lstart,block_size);
    
    int sum = 0;
    for(int i = 0; i < block_size; i++)
    {
    int mult = ssgl_group[i];
    for(int ii = block_size-1; ii > i; ii--)
        {
        mult &= sspl_group[ii];
        }
    sum |= mult;
    }
    ssgl[index] = sum;
    
    int mult = sspl_group[0];
    for(int i = 1; i < block_size; i++)
    {
    mult &= sspl_group[i];
    }
    sspl[index] = mult;
    
    // free from grab_slice allocation
    free(ssgl_group);
    free(sspl_group);
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_super_super_section_gp()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int mstart = index*block_size;
    int* sssgm_group = grab_slice(ssgl,mstart,block_size);
    int* ssspm_group = grab_slice(sspl,mstart,block_size);
    
    int sum = 0;
    for(int i = 0; i < block_size; i++)
    {
    int mult = sssgm_group[i];
    for(int ii = block_size-1; ii > i; ii--)
        {
        mult &= ssspm_group[ii];
        }
    sum |= mult;
    }
    sssgm[index] = sum;
    
    int mult = ssspm_group[0];
    for(int i = 1; i < block_size; i++)
    {
    mult &= ssspm_group[i];
    }
    ssspm[index] = mult;
    
    // free from grab_slice allocation
    free(sssgm_group);
    free(ssspm_group);
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_super_super_section_carry()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int ssscmlast=0;
    if(index==0)
    {
    ssscmlast = 0;
    }
    else
    {
    ssscmlast = ssscm[index-1];
    }
    
    ssscm[index] = sssgm[index] | (ssspm[index]&ssscmlast);
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_super_section_carry()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int sscllast=0;
    if(index%block_size == block_size-1)
    {
    sscllast = ssscm[index/block_size];
    }
    else if( index != 0 )
    {
    sscllast = sscl[index-1];
    }
    
    sscl[index] = ssgl[index] | (sspl[index]&sscllast);
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_section_carry()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int scklast=0;
    if(index%block_size==block_size-1)
    {
    scklast = sscl[index/block_size];
    }
    else if( index != 0 )
    {
    scklast = sck[index-1];
    }
    
    sck[index] = sgk[index] | (spk[index]&scklast);
}


/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_group_carry()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int gcjlast=0;
    if(index%block_size==block_size-1)
    {
    gcjlast = sck[index/block_size];
    }
    else if( index != 0 )
    {
    gcjlast = gcj[index-1];
    }
    
    gcj[index] = ggj[index] | (gpj[index]&gcjlast);
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_carry()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int clast=0;
    if(index%block_size==block_size-1)
    {
    clast = gcj[index/block_size];
    }
    else if( index != 0 )
    {
    clast = ci[index-1];
    }
    
    ci[index] = gi[index] | (pi[index]&clast);
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_sum()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int clast=0;
    if(index==0)
    {
    clast = 0;
    }
    else
    {
    clast = ci[index-1];
    }
    sumi[index] = bin1[index] ^ bin2[index] ^ clast;
}

void cla()
{
  /***********************************************************************************************************/
  // ADAPT ALL THESE FUNCTUIONS TO BE SEPARATE CUDA KERNEL CALL
  // NOTE: Kernel calls are serialized by default per the CUDA kernel call scheduler
  // NOTE: Make sure you set the right CUDA Block Size (e.g., threads per block) for different runs per 
  //       assignment description.
  /***********************************************************************************************************/
    compute_gp<<<(bits+block_size+1)/block_size, block_size>>>();
    compute_group_gp<<<(ngroups+block_size+1)/block_size, block_size>>>();
    compute_section_gp<<<(nsections+block_size+1)/block_size, block_size>>>();
    compute_super_section_gp<<<(nsupersections+block_size+1)/block_size, block_size>>>();
    compute_super_super_section_gp<<<(nsupersupersections+block_size+1)/block_size, block_size>>>();
    compute_super_super_section_carry<<<(nsupersupersections+block_size+1)/block_size, block_size>>>();
    compute_super_section_carry<<<(nsupersections+block_size+1)/block_size, block_size>>>();
    compute_section_carry<<<(nsections+block_size+1)/block_size, block_size>>>();
    compute_group_carry<<<(ngroups+block_size+1)/block_size, block_size>>>();
    compute_carry<<<(bits+block_size+1)/block_size, block_size>>>();
    compute_sum<<<(bits+block_size+1)/block_size, block_size>>>();

  /***********************************************************************************************************/
  // INSERT RIGHT CUDA SYNCHRONIZATION AT END!
  /***********************************************************************************************************/
}

void ripple_carry_adder()
{
  int clast=0, cnext=0;

  for(int i = 0; i < bits; i++)
    {
      cnext = (bin1[i] & bin2[i]) | ((bin1[i] | bin2[i]) & clast);
      sumrca[i] = bin1[i] ^ bin2[i] ^ clast;
      clast = cnext;
    }
}

void check_cla_rca()
{
  for(int i = 0; i < bits; i++)
    {
      if( sumrca[i] != sumi[i] )
	{
	  printf("Check: Found sumrca[%d] = %d, not equal to sumi[%d] = %d - stopping check here!\n",
		 i, sumrca[i], i, sumi[i]);
	  printf("bin1[%d] = %d, bin2[%d]=%d, gi[%d]=%d, pi[%d]=%d, ci[%d]=%d, ci[%d]=%d\n",
		 i, bin1[i], i, bin2[i], i, gi[i], i, pi[i], i, ci[i], i-1, ci[i-1]);
	  return;
	}
    }
  printf("Check Complete: CLA and RCA are equal\n");
}

int main(int argc, char *argv[])
{
  cudaMallocManaged(&sumi, (bits)*sizeof(int));
  cudaMallocManaged(&sumrca, (bits)*sizeof(int));
  cudaMallocManaged(&gi, (bits)*sizeof(int));
  cudaMallocManaged(&pi, (bits)*sizeof(int));
  cudaMallocManaged(&ci, (bits)*sizeof(int));
  cudaMallocManaged(&ggj, (ngroups)*sizeof(int));
  cudaMallocManaged(&gpj, (ngroups)*sizeof(int));
  cudaMallocManaged(&gcj, (ngroups)*sizeof(int));
  cudaMallocManaged(&sgk, (nsections)*sizeof(int));
  cudaMallocManaged(&spk, (nsections)*sizeof(int));
  cudaMallocManaged(&sck, (nsections)*sizeof(int));
  cudaMallocManaged(&ssgl, (nsupersections)*sizeof(int));
  cudaMallocManaged(&sscl, (nsupersections)*sizeof(int));
  cudaMallocManaged(&sspl, (nsupersections)*sizeof(int));
  cudaMallocManaged(&sssgm, (nsupersupersections)*sizeof(int));
  cudaMallocManaged(&ssspm, (nsupersupersections)*sizeof(int));
  cudaMallocManaged(&ssscm, (nsupersupersections)*sizeof(int));
  int randomGenerateFlag = 1;
  int deterministic_seed = (1<<30) - 1;
  char* hexa=NULL;
  char* hexb=NULL;
  char* hexSum=NULL;
  char* int2str_result=NULL;
  unsigned long long start_time=clock_now(); // dummy clock reads to init
  unsigned long long end_time=clock_now();   // dummy clock reads to init

  if( nsupersupersections != block_size )
    {
      printf("Misconfigured CLA - nsupersupersections (%d) not equal to block_size (%d) \n",
	     nsupersupersections, block_size );
      return(-1);
    }
  
  if (argc == 2) {
    if (strcmp(argv[1], "-r") == 0)
      randomGenerateFlag = 1;
  }
  
  if (randomGenerateFlag == 0)
    {
      read_input();
    }
  else
    {
      srand( deterministic_seed );
      hex1 = generate_random_hex(input_size);
      hex2 = generate_random_hex(input_size);
    }
  
  hexa = prepend_non_sig_zero(hex1);
  hexb = prepend_non_sig_zero(hex2);
  hexa[digits] = '\0'; //double checking
  hexb[digits] = '\0';
  
  bin1 = gen_formated_binary_from_hex(hexa);
  bin2 = gen_formated_binary_from_hex(hexb);

  start_time = clock_now();
  cla();
  end_time = clock_now();

  printf("CLA Completed in %llu cycles\n", (end_time - start_time));

  cudaDeviceSynchronize();

  start_time = clock_now();
  ripple_carry_adder();
  end_time = clock_now();

  printf("RCA Completed in %llu cycles\n", (end_time - start_time));

  check_cla_rca();

  if( verbose==1 )
    {
      int2str_result = int_to_string(sumi,bits);
      hexSum = revbinary_to_hex( int2str_result,bits);
    }

  // free inputs fields allocated in read_input or gen random calls
  free(int2str_result);
  free(hex1);
  free(hex2);
  
  // free bin conversion of hex inputs
  free(bin1);
  free(bin2);
  
  if( verbose==1 )
    {
      printf("Hex Input\n");
      printf("a   ");
      print_chararrayln(hexa);
      printf("b   ");
      print_chararrayln(hexb);
    }
  
  if ( verbose==1 )
    {
      printf("Hex Return\n");
      printf("sum =  ");
    }
  
  // free memory from prepend call
  free(hexa);
  free(hexb);

  if( verbose==1 )
    printf("%s\n",hexSum);
  
  free(hexSum);

  cudaFree(gi);
  cudaFree(pi);
  cudaFree(ci);
  cudaFree(ggj);
  cudaFree(gpj);
  cudaFree(gcj);
  cudaFree(sgk);
  cudaFree(spk);
  cudaFree(sck);
  cudaFree(ssgl);
  cudaFree(sspl);
  cudaFree(sscl);
  cudaFree(sssgm);
  cudaFree(ssspm);
  cudaFree(ssscm);
  cudaFree(sumi);
  cudaFree(sumrca);
  
  return 0;
}