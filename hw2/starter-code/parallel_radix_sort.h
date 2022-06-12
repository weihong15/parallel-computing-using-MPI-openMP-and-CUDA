#ifndef PARALLEL_RADIX_SORT
#define PARALLEL_RADIX_SORT

#include <algorithm>
#include <vector>
#include "test_util.h"


constexpr const uint kSizeTestVector = 4000000;
constexpr const uint kNumBits = 16; // must be a divider of 32 for this program to work
constexpr const uint kRandMax = 1 << 31;

/* Function: computeBlockHistograms
 * --------------------------------
 * Splits keys into numBlocks and computes an histogram with numBuckets buckets
 * Remember that numBuckets and numBits are related; same for blockSize and numBlocks.
 * Should work in parallel.
 */
std::vector<uint> computeBlockHistograms(
    const std::vector<uint> &keys,
    uint numBlocks, 
    uint numBuckets,
    uint numBits, 
    uint startBit, 
    uint blockSize
) {
    std::vector<uint> blockHistograms(numBlocks * numBuckets, 0);
#pragma omp parallel for
    for (uint i=0; i< numBlocks; i++){
	    for (uint j=0; j< blockSize; j++){
		    //make sure still within index
		    if (i*blockSize+j< keys.size()){
			    uint bucket = (numBuckets -1) & (keys[i*blockSize +j]>> startBit);
			    blockHistograms[i*numBuckets +bucket]++;}}}
    return blockHistograms;
}

/* Function: reduceLocalHistoToGlobal
 * ----------------------------------
 * Takes as input the local histogram of size numBuckets * numBlocks and "merges"
 * them into a global histogram of size numBuckets.
 */
std::vector<uint> reduceLocalHistoToGlobal(
    const std::vector<uint> &blockHistograms,
    uint numBlocks, 
    uint numBuckets
) {
    std::vector<uint> globalHisto(numBuckets, 0);
    // blockHist length is numBlocks * numBuckets, so will not go out of range
    //#pragma omp parallel for collapse(2)
    // to do this, we need reduce, but it's hard to input vector in
    for (uint blockNum = 0; blockNum<numBlocks; blockNum++)
	    for (uint bucketNum = 0; bucketNum < numBuckets; bucketNum++)
		    globalHisto[bucketNum] += blockHistograms[blockNum*numBuckets +bucketNum];
    return globalHisto;
}

/* Function: scanGlobalHisto
 * -------------------------
 * This function should simply scan the global histogram.
 */
std::vector<uint> scanGlobalHisto(
    const std::vector<uint> &globalHisto,
    uint numBuckets
) {
    std::vector<uint> globalHistoExScan(numBuckets, 0);
    // TODO
    for (uint i=1; i<globalHisto.size(); i++)
	    globalHistoExScan[i] = globalHistoExScan[i-1] + globalHisto[i-1];
    return globalHistoExScan;
}

/* Function: computeBlockExScanFromGlobalHisto
 * -------------------------------------------
 * Takes as input the globalHistoExScan that contains the global histogram after the scan
 * and the local histogram in blockHistograms. Returns a local histogram that will be used
 * to populate the sorted array.
 */
std::vector<uint> computeBlockExScanFromGlobalHisto(
    uint numBuckets,
    uint numBlocks,
    const std::vector<uint> &globalHistoExScan,
    const std::vector<uint> &blockHistograms
) {
    std::vector<uint> blockExScan(numBuckets * numBlocks, 0);
    //start with globalhistoExScan, keep addingi
    std::copy(globalHistoExScan.begin(), globalHistoExScan.end(), blockExScan.begin());
    for (uint i = 1; i<numBlocks; i++)
	    for (uint j = 0; j< numBuckets; j++)
		    blockExScan[i*numBuckets +j] = blockExScan[(i-1)*numBuckets+j]\
						   +blockHistograms[(i-1)*numBuckets+j];
    return blockExScan;
}

/* Function: populateOutputFromBlockExScan
 * ---------------------------------------
 * Takes as input the blockExScan produced by the splitting of the global histogram
 * into blocks and populates the vector sorted.
 */
void populateOutputFromBlockExScan(
    const std::vector<uint> &blockExScan,
    uint numBlocks, 
    uint numBuckets, 
    uint startBit,
    uint numBits, 
    uint blockSize, 
    const std::vector<uint> &keys,
    std::vector<uint> &sorted
) {
	//can only parallel for each bucket
#pragma omp parallel for
	for (uint i = 0; i< numBlocks;i++){
		std::vector<uint> offset(numBuckets,0);
		auto start = blockExScan.begin()+i*numBuckets;
		std::copy(start,start+numBuckets, offset.begin());
		for (uint j = 0; j<blockSize; j++){
			//need to check index does not exceed block size
			if(i*blockSize + j < keys.size()){
				uint bucket = (numBuckets - 1) & (keys[i*blockSize + j] >> startBit);
				sorted[offset[bucket]++] = keys[i*blockSize +j];}
		}}
			
	}


/* Function: radixSortParallelPass
 * -------------------------------
 * A pass of radixSort on numBits starting after startBit.
 */
void radixSortParallelPass(
    std::vector<uint> &keys, 
    std::vector<uint> &sorted,
    uint numBits, 
    uint startBit,
    uint blockSize
) {
    uint numBuckets = 1 << numBits;

    // Choose numBlocks so that numBlocks * blockSize is always greater than keys.size().
    uint numBlocks = (keys.size() + blockSize - 1) / blockSize;

    // go over each block and compute its local histogram
    std::vector<uint> blockHistograms = computeBlockHistograms(keys, numBlocks,
                                        numBuckets, numBits, startBit, blockSize);

    // first reduce all the local histograms into a global one
    std::vector<uint> globalHisto = reduceLocalHistoToGlobal(blockHistograms,
                                    numBlocks, numBuckets);

    // now we scan this global histogram
    std::vector<uint> globalHistoExScan = scanGlobalHisto(globalHisto, numBuckets);

    // now we do a local histogram in each block and add in the global value to get global position
    std::vector<uint> blockExScan = computeBlockExScanFromGlobalHisto(numBuckets,
                                    numBlocks, globalHistoExScan, blockHistograms);

    // populate the sorted vector
    populateOutputFromBlockExScan(blockExScan, numBlocks, numBuckets, startBit,
                                  numBits, blockSize, keys, sorted);
}

int radixSortParallel(
    std::vector<uint> &keys, 
    std::vector<uint> &keys_tmp,
    uint numBits, 
    uint numBlocks
) {
    for (uint startBit = 0; startBit < 32; startBit += 2 * numBits) 
    {
        radixSortParallelPass(keys, keys_tmp, numBits, startBit, keys.size() / numBlocks);
        radixSortParallelPass(keys_tmp, keys, numBits, startBit + numBits, keys.size() / numBlocks);
    }

    return 0;
}


void runBenchmark(std::vector<uint>& keys_parallel, std::vector<uint>& temp_keys, uint kNumBits, uint n_blocks, uint n_threads){
    // TODO: Call omp_set_num_threads() with the correct argument
	omp_set_num_threads(n_threads);
    double startRadixParallel = omp_get_wtime();
    // TODO: Call radixSortParallel() with the correct arguments
	radixSortParallel(keys_parallel,temp_keys,kNumBits,n_blocks);
    double endRadixParallel = omp_get_wtime();
    printf("%8.3f", endRadixParallel - startRadixParallel);
}
#endif
