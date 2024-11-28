#include <sys/time.h>
#include "cublas-norm/syrk.h"
#include "util.h"

double get_time() {
   struct timeval t;
   gettimeofday(&t, NULL);
   return t.tv_sec + t.tv_usec / 1000000.0;
}

void tensor_transformation(Info *input, int filter_h, int filter_w){
    // Extract the dimensions from input->shape
    int C = input->shape[0];  // Number of channels
    int H = input->shape[1];  // Height of the input tensor
    int W = input->shape[2];  // Width of the input tensor

    // Calculate the dimensions of the output tensor
    int H_new = H - filter_h + 1;
    int W_new = W - filter_w + 1;

    // Ensure the filter dimensions are valid
    if (H_new <= 0 || W_new <= 0) {
        // Handle error: Filter size is larger than input dimensions
        return;
    }

    // Allocate memory for the reshaped tensor
    size_t output_size = H_new * W_new * filter_h * filter_w * C;
    float* reshape_tensor = (float*)malloc(output_size * sizeof(float));
    if (reshape_tensor == NULL) {
        // Handle memory allocation failure
        return;
    }

    // Perform the unfolding and reshaping
    for(int hi = 0; hi < H_new; hi++){
        for(int wi = 0; wi < W_new; wi++){
            for(int fi = 0; fi < filter_h; fi++){
                for(int fj = 0; fj < filter_w; fj++){
                    for(int c = 0; c < C; c++){
                        int h_in = hi + fi;
                        int w_in = wi + fj;
                        int index_in = c * H * W + h_in * W + w_in;
                        int index_out = hi * W_new * filter_h * filter_w * C
                                      + wi * filter_h * filter_w * C
                                      + fi * filter_w * C
                                      + fj * C
                                      + c;
                        reshape_tensor[index_out] = input->tensor[index_in];
                    }
                }
            }
        }
    }

    // Update input->tensor and input->shape
    free(input->tensor);  // Free the old tensor if it's dynamically allocated
    input->tensor = reshape_tensor;
    input->shape[0] = H_new;      // (X - filter_h + 1)
    input->shape[1] = W_new;      // (Y - filter_w + 1)
    input->shape[2] = filter_h;   // Filter height
    input->shape[3] = filter_w;   // Filter width
    input->shape[4] = C;          // Number of channels
}
// Include <torch/extension.h> and register the function only if compiling with setup.py
#ifdef BUILD_WITH_PYTORCH
#include <pybind11/numpy.h>
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_time", &get_time, "Get Current Time");
    pybind11::class_<Matrix>(m, "Matrix")
        .def(pybind11::init<int, double, int, double>(),
             pybind11::arg("A_N"), pybind11::arg("A_coff"),
             pybind11::arg("C_N"), pybind11::arg("C_coff"))
        .def("getA", [](const Matrix &matrix) {
            // Use the public member `A_N` directly
            return pybind11::array_t<double>({matrix.A_N, matrix.A_N}, matrix.getA());
        })
        .def("getC", [](const Matrix &matrix) {
            // Use the public member `C_N` directly
            return pybind11::array_t<double>({matrix.C_N, matrix.C_N}, matrix.getC());
        })
        .def("getAlpha", &Matrix::getAlpha)
        .def("getBeta", &Matrix::getBeta)
        .def_readonly("A_N", &Matrix::A_N)
        .def_readonly("C_N", &Matrix::C_N);

    pybind11::class_<Info>(m, "Info")
        .def(pybind11::init<int, double, int, double, int, double, double>(),
             pybind11::arg("A_N"), pybind11::arg("A_coff"),
             pybind11::arg("C_N"), pybind11::arg("C_coff"),
             pybind11::arg("iteration"), pybind11::arg("value"),
             pybind11::arg("time"))
        .def(pybind11::init<>()) // Default constructor
        .def_property_readonly("matrix", [](const Info &info) {
            return &info.matrix;
        }, pybind11::return_value_policy::reference)
        .def_readwrite("iteration", &Info::iteration)
        .def_readwrite("value", &Info::value)
        .def_readwrite("time", &Info::time);
    m.def("it_syrk", [](Info &result) {
        it_syrk(&result);
    }, "Perform iterative SYRK operation");
}

#endif