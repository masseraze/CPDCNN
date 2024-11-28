#include <gtest/gtest.h>
#include "util.h"
// Example function to test
int add(int a, int b) {
    return a + b;
}

// Test case
TEST(AdditionTest, BasicAssertions) {
    EXPECT_EQ(add(1, 2), 3);
    EXPECT_EQ(add(-1, 1), 0);
    EXPECT_EQ(add(0, 0), 0);
}

TEST(tensor_transformation, threetofive){
    // Initialize input tensor with known values
    // Let's assume C=1, H=3, W=3
    Info input;
    int C = 1;
    int H = 3;
    int W = 3;
    input.shape[0] = C;
    input.shape[1] = H;
    input.shape[2] = W;
    input.shape[3] = 0;
    input.shape[4] = 0;

    // Allocate memory for input tensor
    size_t input_size = C * H * W;
    input.tensor = (float*)malloc(input_size * sizeof(float));
    ASSERT_TRUE(input.tensor != NULL);

    // Initialize tensor with values from 1 to 9
    // Tensor data:
    // [[1, 2, 3],
    //  [4, 5, 6],
    //  [7, 8, 9]]
    for (int i = 0; i < input_size; ++i) {
        input.tensor[i] = (float)(i + 1);
    }

    // Set filter size
    int filter_h = 2;
    int filter_w = 2;

    // Call the tensor_transformation function
    tensor_transformation(&input, filter_h, filter_w);

    // Expected output dimensions
    int H_new = H - filter_h + 1;  // 2
    int W_new = W - filter_w + 1;  // 2

    // Expected shape: [(X - filter_h + 1), (Y - filter_w + 1), filter_h, filter_w, C]
    int expected_shape[5] = { H_new, W_new, filter_h, filter_w, C };
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(input.shape[i], expected_shape[i]);
    }

    // Expected output tensor:
    // The expected reshaped tensor will have dimensions [H_new, W_new, filter_h, filter_w, C]
    // We can flatten this to compare easily.

    // Manually compute expected output
    // The patches extracted are:
    // Patch at (0,0):
    // [[1,2],
    //  [4,5]]
    // Patch at (0,1):
    // [[2,3],
    //  [5,6]]
    // Patch at (1,0):
    // [[4,5],
    //  [7,8]]
    // Patch at (1,1):
    // [[5,6],
    //  [8,9]]

    // Expected output tensor in order:
    // For each position (hi, wi), we have a patch of size [filter_h, filter_w, C]

    float expected_output[] = {
        // hi=0, wi=0
        1, 2, 4, 5,
        // hi=0, wi=1
        2, 3, 5, 6,
        // hi=1, wi=0
        4, 5, 7, 8,
        // hi=1, wi=1
        5, 6, 8, 9
    };
    // Note: Since C=1, we omit the channel dimension for simplicity

    // Now compare input.tensor with expected_output
    size_t output_size = H_new * W_new * filter_h * filter_w * C;
    ASSERT_EQ(output_size, sizeof(expected_output)/sizeof(float));

    for (size_t i = 0; i < output_size; ++i) {
        EXPECT_FLOAT_EQ(input.tensor[i], expected_output[i]);
    }

    // Clean up
    free(input.tensor);
}
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    auto tensor = torch::rand({3, 4, 5});
    Tmp info(tensor, 10, 3.14, 2.71);
    info.printShape();
    return RUN_ALL_TESTS();
}