#include<vector>

std::vector<uint> serialSum(const std::vector<uint> &v) 
{
    std::vector<uint> sums(2);
    // TODO
    //for(auto itr:v) sums[itr%2]= sums[itr%2]+ itr;
    for (uint i = 0; i<v.size();i++) sums[v[i]%2] = sums[v[i]%2] + v[i];

    return sums;
}

std::vector<uint> parallelSum(const std::vector<uint> &v) 
{
    std::vector<uint> sums(2);
    // TODO
    //std::cout << sums[0] <<std::endl;
//#pragma omp parallel for reduction(+: sums [0:2])
    //for(auto itr:v) sums[itr%2]= sums[itr%2]+ itr;
    //for (uint i = 0; i<v.size();i++) sums[v[i]%2] += v[i];
    
   int s1 = 0;
  int s2 = 0;
#pragma omp parallel for reduction(+: s1,s2)
  for (uint i = 0; i<v.size();i++){
  if (v[i]%2) s1+= v[i];
  else s2 +=v[i];
  } 
    sums[0] = s2;
    sums[1] = s1;
    return sums;
}
