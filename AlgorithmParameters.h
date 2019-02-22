#ifndef ALGORITHM_PARAMETERS_H
#define ALGORITHM_PARAMETERS_H

#include "UnifiedMemoryManaged.h"


class AlgorithmParameters : public UnifiedMemoryManaged
{
public:

    bool color_processing = true;
    /* bool color_processing = false; */
    int iterations = 3;
	
	int sign = -1; // -1 left, 1 right


    float min_disparity = 0.0f;
    float max_disparity = 80.0f;

    int box_width = 19;
    int box_height = 19;


    float tau_color = 10.0f;
    float tau_gradient = 2.0f;
    float alpha = 0.9f;
    float gamma = 10.0f;


};



#endif //ALGORITHM_PARAMETERS_H
