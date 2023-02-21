/*********************************************************************/
//
// 02/02/2023: Revised Version for 32M bit adder with 32 bit blocks
//
/*********************************************************************/

#include "main.h"

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
__managed__ int *gi, *pi, *ci;
__managed__ int *ggj, *gpj, *gcj;
__managed__ int *sgk, *spk, *sck;
__managed__ int *ssgl, *sspl, *sscl;
__managed__ int *sssgm, *ssspm, *ssscm;
__managed__ int *sumi, *sumrca;

//Integer array of inputs in binary form
__managed__ int *bin1, *bin2;

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

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index; i < bits; i += stride){
    gi[i] = bin1[i] & bin2[i];
    pi[i] = bin1[i] | bin2[i];
  }
    
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_group_gp()
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for(int j = index; j < ngroups; j += stride){
    int jstart = j*block_size;
    //int* ggj_group = *(gi+jstart);
    //int* gpj_group = *(pi+jstart);

    int sum = 0;
    for(int i = jstart; i < jstart+block_size; i++)
    {
        int mult = gi[i]; //grabs the g_i term for the multiplication
        for(int ii = block_size-1; ii > i; ii--)
        {
            mult &= pi[ii]; //grabs the p_i terms and multiplies it with the previously multiplied stuff (or the g_i term if first round)
        }
        sum |= mult; //sum up each of these things with an or
    }
    ggj[j] = sum;

    int mult = pi[jstart];
    for(int i = jstart+1; i < block_size+jstart; i++)
    {
        mult &= pi[i];
    }
    gpj[j] = mult;
      // free from grab_slice allocation
    //free(ggj_group);
    //free(gpj_group);
  }
  
    
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_section_gp()
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for(int k = index; k < nsections; k += stride){
    int kstart = k*block_size;
    //int* sgk_group = ggj;
    //int* spk_group = gpj;
    
    int sum = 0;
    for(int i = kstart; i < block_size+kstart; i++)
      {
        int mult = ggj[i];
        for(int ii = block_size-1; ii > i; ii--)
        {
            mult &= gpj[ii];
          }
        sum |= mult;
      }
    sgk[k] = sum;
    
    int mult = gpj[kstart];
    for(int i = kstart+1; i < block_size+kstart; i++)
      {
  mult &= gpj[i];
      }
    spk[k] = mult;
      // free from grab_slice allocation
    //free(sgk_group);
    //free(spk_group);
  }
  

    
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_super_section_gp()
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for(int l = index; l < nsupersections; l += stride){
      int lstart = l*block_size;
      //int* ssgl_group = &sgk;
      //int* sspl_group = &spk;
      
      int sum = 0;
      for(int i = lstart; i < block_size+lstart; i++)
        {
	  int mult = sgk[i];
	  for(int ii = block_size-1; ii > i; ii--)
            {
	      mult &= spk[ii];
            }
	  sum |= mult;
        }
      ssgl[l] = sum;
      
      int mult = spk[lstart];
      for(int i = lstart+1; i < lstart+block_size; i++)
        {
	  mult &= spk[i];
        }
      sspl[l] = mult;
      
      // free from grab_slice allocation
      //free(ssgl_group);
      //free(sspl_group);
    }
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_super_super_section_gp()
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for(int m = index; m < nsupersections; m += stride){
      int mstart = m*block_size;
      //int* sssgm_group = &ssgl;
      //int* ssspm_group = &sspl;
      
      int sum = 0;
      for(int i = mstart; i < mstart+block_size; i++)
        {
	  int mult = ssgl[i];
	  for(int ii = block_size-1; ii > i; ii--)
            {
	      mult &= sspl[ii];
            }
	  sum |= mult;
        }
      sssgm[m] = sum;
      
      int mult = sspl[0];
      for(int i = mstart+1; i < mstart+block_size; i++)
        {
	  mult &= sspl[i];
        }
      ssspm[m] = mult;
      
      // free from grab_slice allocation
      //free(sssgm_group);
      //free(ssspm_group);
    }
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_super_super_section_carry()
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for(int m = index; m < nsupersupersections; m += stride){
    {
      int ssscmlast=0;
      if(m==0)
        {
	  ssscmlast = 0;
        }
      else
        {
	  ssscmlast = ssscm[m-1];
        }
      
      ssscm[m] = sssgm[m] | (ssspm[m]&ssscmlast);
    }
  }
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_super_section_carry()
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for(int l = index; l < nsupersections; l += stride){
      int sscllast=0;
      if(l%block_size == block_size-1)
        {
	  sscllast = ssscm[l/block_size];
        }
      else if( l != 0 )
        {
	  sscllast = sscl[l-1];
        }
      
      sscl[l] = ssgl[l] | (sspl[l]&sscllast);
    }
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_section_carry()
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for(int k = index; k < nsections; k += stride){
      int scklast=0;
      if(k%block_size==block_size-1)
        {
	  scklast = sscl[k/block_size];
        }
      else if( k != 0 )
        {
	  scklast = sck[k-1];
        }
      
      sck[k] = sgk[k] | (spk[k]&scklast);
    }
}


/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_group_carry()
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for(int j = index; j < ngroups; j += stride){
      int gcjlast=0;
      if(j%block_size==block_size-1)
        {
	  gcjlast = sck[j/block_size];
        }
      else if( j != 0 )
        {
	  gcjlast = gcj[j-1];
        }
      
      gcj[j] = ggj[j] | (gpj[j]&gcjlast);
    }
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_carry()
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int clast=0;/*********************************************************************/
//
// 02/02/2023: Revised Version for 32M bit adder with 32 bit blocks
//
/*********************************************************************/

  if(i%block_size==block_size-1)
    {
clast = gcj[i/block_size];
    }
  else if( i != 0 )
    {
clast = ci[i-1];
    }
  
  ci[i] = gi[i] | (pi[i]&clast);
    
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_sum()
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int clast=0;
  if(i==0)
    {
clast = 0;
    }
  else
    {
clast = ci[i-1];
    }
  sumi[i] = bin1[i] ^ bin2[i] ^ clast;

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

  cudaMallocManaged(&gi, bits*sizeof(int));
  cudaMallocManaged(&pi, bits*sizeof(int));
  cudaMallocManaged(&ci, bits*sizeof(int));

  cudaMallocManaged(&ggj, ngroups*sizeof(int));
  cudaMallocManaged(&gpj, ngroups*sizeof(int));
  cudaMallocManaged(&gcj, ngroups*sizeof(int));

  cudaMallocManaged(&sgk, nsections*sizeof(int));
  cudaMallocManaged(&spk, nsections*sizeof(int));
  cudaMallocManaged(&sck, nsections*sizeof(int));

  cudaMallocManaged(&ssgl, nsupersections*sizeof(int));
  cudaMallocManaged(&sspl, nsupersections*sizeof(int));
  cudaMallocManaged(&sscl, nsupersections*sizeof(int));

  cudaMallocManaged(&sssgm, nsupersections*sizeof(int));
  cudaMallocManaged(&ssspm, nsupersections*sizeof(int));
  cudaMallocManaged(&ssscm, nsupersections*sizeof(int));

  cudaMallocManaged(&sumi, bits*sizeof(int));
  cudaMallocManaged(&sumrca, bits*sizeof(int));

  cudaMallocManaged(&bin1, bits*sizeof(int));
  cudaMallocManaged(&bin2, bits*sizeof(int));
  
  hexa = prepend_non_sig_zero(hex1);
  hexb = prepend_non_sig_zero(hex2);
  hexa[digits] = '\0'; //double checking
  hexb[digits] = '\0';
  
  bin1 = gen_formated_binary_from_hex(hexa);
  bin2 = gen_formated_binary_from_hex(hexb);

  start_time = clock_now();
  cla();
  end_time = clock_now();

  cudaDeviceSynchronize();

  printf("CLA Completed in %llu cycles\n", (end_time - start_time));

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
