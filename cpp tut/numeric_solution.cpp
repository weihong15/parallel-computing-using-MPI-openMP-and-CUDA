#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>


double sum_every_other_element(const std::vector<double>& vec)
{
    // We want to sum every other element in this vector i.e. vec[0] + vec[2] + ....
    // To exercise std::iota and std::accumulate, we'll do it in the following fashion:
    // We'll generate a vector of indices to access within vec, and then 

    // TODO: Since we'll look at half of vec, make a new vector with half the size of type unsigned int
    // std::iota(begin, end, start_val)
    std::vector<unsigned int> indices(vec.size() / 2);
    std::iota(indices.begin(), indices.end(), 0);

    // TODO: Since we're hitting every other element, multiply each element by two
    // modify in place using for each. The lambda has a reference, so just modify that parameter.
    // std::for_each(begin, end, [](T& val) {  ...  })
    std::for_each(indices.begin(), indices.end(), [](unsigned int& idx) { idx = idx * 2; });

    // TODO: Now that we have all of the even indices, we can sum them up by iterating over indices
    // and accessing the vec array via a capture
    // std::accumulate(begin, end, [&](double sum, unsigned int& val) {....})
    return std::accumulate(
        indices.begin(), 
        indices.end(), 
        0.0,
        [&vec] (double& sum, unsigned int& val) { return sum + vec[val]; }
    );
}


int main()
{
    std::vector<double> vec(10);
    std::iota(vec.begin(), vec.end(), 0.0);

    // GOLD
    unsigned gold_sum = 0;
    for (unsigned int i = 0; i < vec.size(); i += 2)
        gold_sum += vec[i];

    std::cout << "Gold: " << gold_sum << std::endl;
    std::cout << "Test: " << sum_every_other_element(vec) << std::endl;
    return 0;    
}