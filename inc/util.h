#include "cublas-norm/syrk.h"
#include <torch/torch.h>
double get_time();
void tensor_transformation(Info *input, int filter_h, int filter_w);

class Tmp {
public:
    torch::Tensor tensor;
    int iteration;
    double value;
    double time;

    Tmp(torch::Tensor t, int iter, double val, double t_time)
        : tensor(t), iteration(iter), value(val), time(t_time) {}

    void printShape() {
        auto sizes = tensor.sizes();
        std::cout << "Tensor shape: " << sizes << std::endl;
    }
};