#include <ATen/ATen.h>
#include <torch/script.h>

#include <cstdio>
#include <string>

/** Example program for running a PyTorch model in C++
 * to build:
 *     source envsetup.sh
 *     mkdir -p build && cd build
 *     cmake ..
 *     make
 */

void print_help(const char *cmd)
{
    printf("Usage: %s <model> <data>\n", cmd);
    printf("\t<model>   path to the exported script module (.pt file)\n");
    printf("\t<data>    path to the input data (.txt file)\n");
}

bool extract_label(const std::string &filename, int &label)
{
    constexpr int MIN_LENGTH = 5;
    int n = static_cast<int>(filename.size());
    if (n < MIN_LENGTH)
    {
        return false;
    }
    label = std::stoi(filename.substr(n - MIN_LENGTH, 1));
    return true;
}

bool extract_data(const std::string &filename, std::vector<float> &data)
{
    FILE *fp = fopen(filename.c_str(), "r");
    if (!fp)
    {
        return false;
    }
    float value;
    while (fscanf(fp, "%f", &value) == 1)
    {
        data.push_back(value);
    }
    fclose(fp);
    return true;
}

int main(int argc, const char* argv[])
{
    if (argc != 3)
    {
        print_help(argv[0]);
        return -1;
    }
    const std::string model_path = argv[1];
    const std::string data_path = argv[2];

    // Deserialize the ScriptModule from a file using torch::jit::load().
    torch::jit::script::Module module;
    try
    {
        module = torch::jit::load(model_path);
    }
    catch (const c10::Error& e)
    {
        fprintf(stderr, "error loading model: %s\n%s", model_path.c_str(), e.what());
        return -1;
    }
    printf("loaded model: %s\n", model_path.c_str());

    // Load the input data as a tensor, and extract label
    std::vector<float> raw_data;
    int label;
    if (!extract_data(data_path, raw_data) || !extract_label(data_path, label))
    {
        fprintf(stderr, "failed to load data: %s\n", data_path.c_str());
        return -1;
    }
    //at::Tensor data = torch::from_blob(raw_data.data(), {1, 728});
    at::Tensor tensor_data = torch::tensor(raw_data);

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor_data);

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    int y_hat = output.argmax(1).item().toInt();
    printf("prediction: %d, truth: %d\n", y_hat, label);
    return 0;
}
