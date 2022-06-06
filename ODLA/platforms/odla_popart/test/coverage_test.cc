//===- Halo Compiler Generated File --------------------------------===//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <ODLA/odla.h>
#include <stdio.h>

#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "ODLA/odla_common.h"
#include "doctest.h"
#include "odla_popart.h"
#include "common.h"
#include "popart_config.h"

//#include "json.hpp"
#include <dlfcn.h>

#include <iostream>
#include <popart/builder.hpp>
#include <popart/devicemanager.hpp>
#include <popart/logging.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op.hpp>
#include <popart/op/l1.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/session.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

// namespace CustomOperators {
//   extern const popart::OperatorIdentifier Rsqrt_1;
// }


void shapeInfer(std::vector<std::vector<int64_t>> &shapes, std::vector<int64_t> &output1, std::vector<int64_t> &output2) {
  auto bb13_shape = shapes[2];
  unsigned num_anchors = 3, dim_[3], n[3];
  for (int i = 0; i < 3; i++) {
    auto shape = shapes[2 + i];
    if (shape[0] != 1) {
      printf("We assumed that dim0 = 1");
    } else if (shape[2] != shape[3]) {
      printf("BBox has incorrect shape %d %d\n", shape[2], shape[3]);
      

    }
    dim_[i] = shape[2];
    n[i] = dim_[i] * dim_[i] * num_anchors;
  }
  unsigned N = n[0] + n[1] + n[2];
  unsigned cls_num = bb13_shape[1] / num_anchors - 5;
  printf("aaaaam %d %d %d \n",cls_num, bb13_shape[1], num_anchors);

  std::cout << " xxx  "<<  cls_num << " sssss " << N / 169 << std::endl;

  output1.push_back(cls_num);
  output1.push_back(N / 169 );
  output1.push_back(5);

  output2.push_back(cls_num);
}

TEST_CASE("cache_file_test") {
  SUBCASE("test post") {
  std::cout << "=====> 1, OK" << std::endl;

  void* handle = dlopen("build/libcustom_ops.so", RTLD_LAZY);
  if (!handle) {
    std::cerr << "Cannot open library: " << dlerror() << std::endl;
    ;
  }
  std::cout << "=====> 2, OK" << std::endl;

  auto builder = popart::Builder::create();

  // Add input tensors


  popart::TensorInfo orig_img_w_info{popart::DataType::UINT32,
                                     std::vector<int64_t>{}};
  std::cout << "Adding input tensor orig_img_w\n";
  auto orig_img_w = builder->addInputTensor(orig_img_w_info);

  popart::TensorInfo orig_img_h_info{popart::DataType::UINT32,
                               std::vector<int64_t>{}};
  std::cout << "Adding input tensor orig_img_h\n";
  auto orig_img_h = builder->addInputTensor(orig_img_h_info);

  popart::TensorInfo bb13_info{popart::DataType::FLOAT,
                                     std::vector<int64_t>{1, 30, 13, 13}};        //
  std::cout << "Adding input tensor bb13\n";
  auto bb13 = builder->addInputTensor(bb13_info);

  popart::TensorInfo bb26_info{popart::DataType::FLOAT,
                               std::vector<int64_t>{1, 30, 26, 26}};
  std::cout << "Adding input tensor bb26\n";
  auto bb26 = builder->addInputTensor(bb26_info);

  popart::TensorInfo bb52_info{popart::DataType::FLOAT,
                               std::vector<int64_t>{1, 30, 52, 52}};
  std::cout << "Adding input tensor bb52\n";
  auto bb52 = builder->addInputTensor(bb52_info);

  // Add operation
  std::cout << "Adding custom operation PostProcess\n";
  const static popart::OperatorIdentifier postprocess(
    popart::Domain::ai_graphcore, "PostProcess", 1, 5, 2);

  auto o = builder->customOp(postprocess, 1, {orig_img_w, orig_img_h, bb13, bb26, bb52}, 2, {});

  std::cout << "Get the tensor type and tensor shape of the output of "
               "AttentionMask with tensorid: "
            << o << std::endl;
  auto data_type = builder->getTensorDataType(o[0]);
  auto data_shape = builder->getTensorShape(bb13);
  std::cout << "=================================================="
            << data_shape << std::endl;
  
  auto shape_imgw = builder->getTensorShape(orig_img_w);
  auto shape_imgh = builder->getTensorShape(orig_img_h);
  auto shape_bb13 = builder->getTensorShape(bb13);
  auto shape_bb26 = builder->getTensorShape(bb26);
  auto shape_bb52 = builder->getTensorShape(bb52);
  std::vector<std::vector<int64_t>> shapes;
  std::vector<int64_t> out1;
  std::vector<int64_t> out2;
  shapes.push_back(shape_imgw);
  shapes.push_back(shape_imgh);
  shapes.push_back(shape_bb13);
  shapes.push_back(shape_bb26);
  shapes.push_back(shape_bb52);
  //shapeInfer(shapes, out1, out2);

  std::cout << "Getting model proto\n";
  auto proto = builder->getModelProto();
  builder->saveModelProto("postprocess_test.onnx");

  std::cout << "Constructing DataFlow\n";
  auto dataFlow =
      popart::DataFlow(1, {{o[0], popart::AnchorReturnType("ALL")}});

  std::map<std::string, std::string> deviceOpts{{"numIPUs", "1"}};
  auto ipuModelDevice = 
  popart::DeviceManager::createDeviceManager().acquireAvailableDevice(1);

  std::cout << "Creating session from Onnx Model...\n";
  auto session = popart::InferenceSession::createFromOnnxModel(proto, dataFlow,
                                                               ipuModelDevice);
  std::cout << "Creating session from Onnx Model...done\n";

  // // Prepare input tensor
  uint32_t rawInputData1 = 255;
  popart::NDArrayWrapper<uint32_t> orig_img_w_(&rawInputData1, {1});

  uint32_t rawInputData2 = 255;
  popart::NDArrayWrapper<uint32_t> orig_img_h_(&rawInputData2, {1});

  float* rawInputData3 = new float[1 * 13 * 13 * 30];
  std::fill_n(rawInputData3, 1 * 13 * 13 * 30, 0.f);

  rawInputData3[0 + 85] = 0.5;
  rawInputData3[169 + 85] = 0.5;
  rawInputData3[169*2 + 85] = 0.7;
  rawInputData3[169*3 + 85] = 0.7;
  rawInputData3[169*4 + 85] = 0.9;
  rawInputData3[169*5 + 85] = 0.9;

  popart::NDArrayWrapper<float> bb13_(rawInputData3, {1, 30, 13, 13});


  float* rawInputData4 = new float[1 * 26 * 26 * 30];
  std::fill_n(rawInputData4, 1 * 26 * 26 * 30, 0.f);
  popart::NDArrayWrapper<float> bb26_(rawInputData4, {1, 30, 26, 26});

  float *rawInputData5 = new float[1 * 52 * 52 * 30];
  std::fill_n(rawInputData5, 1 * 52 * 52 * 30, 0.f);
  popart::NDArrayWrapper<float> bb52_(rawInputData5, {1, 30, 52, 52});

  std::map<popart::TensorId, popart::IArray&> inputs = {
      {orig_img_w, orig_img_w_}, 
      {orig_img_h, orig_img_h_},
      {bb13, bb13_},
      {bb26, bb26_},
      {bb52, bb52_},
      };

  uint64_t _len = 1;
  for (auto i : out1)
  {
    _len *= i;
  }
  printf("XXXX %ld\n", _len);

  float* rawOutputData = new float[_len];
  popart::NDArrayWrapper<float> outData(rawOutputData, {out1[0], out1[1], out1[2]});
  std::map<popart::TensorId, popart::IArray&> anchors = {{o[0], outData}};

  std::cout << "Preparing session device...\n";
  session->prepareDevice();
  std::cout << "Preparing session device...done\n";

  popart::StepIO stepio(inputs, anchors);

  std::cout << "Running..."
            << "\n";
  session->run(stepio);
  std::cout << "Running...done"
            << "\n";

  std::cout << "Output Data: " << outData << "\n";  
  }
}

typedef unsigned short uint16_t;
using namespace std;


void set_computationItem(odla_computation comp, int ipu_nums) {
  bool use_ipu_model = 0;
  int ipu_num = ipu_nums;
  int batches_per_step = 1;

  odla_SetComputationItem(comp, ODLA_USE_SIM_MODE,
                          (odla_item_value)&use_ipu_model);
  odla_SetComputationItem(comp, ODLA_PROCESSOR_NUM, (odla_item_value)&ipu_num);
  odla_SetComputationItem(comp, ODLA_BATCHES_PER_STEP,
                          (odla_item_value)&batches_per_step);

}

odla_status generate_model() 
{
  std::vector<float> py = {2};
  std::vector<float> pz = {3};
  auto Input =
    odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                        (const odla_value_id)("Input"));
  auto py_ = odla_CreateConstant({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                                py.data(), (const odla_value_id) "Mul_const");
  auto pz_ = odla_CreateConstant({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                                pz.data(), (const odla_value_id) "Add_const");
  auto Mul = odla_Mul(py_, Input, (const odla_value_id) "Mul");
  auto Add = odla_Add(pz_, Mul, (const odla_value_id) "Add");
  odla_SetValueAsOutput(Add);
}

odla_status model_helper() 
{
  std::vector<float> py = {2};
  std::vector<float> pz = {3};
  auto Input =
    odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                        (const odla_value_id)("Input"));
  auto py_ = odla_CreateConstant({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                                py.data(), (const odla_value_id) "Mul_const");
  auto pz_ = odla_CreateConstant({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                                pz.data(), (const odla_value_id) "Add_const");
  auto Mul = odla_Mul(py_, Input, (const odla_value_id) "Mul");
  auto Add = odla_Add(pz_, Mul, (const odla_value_id) "Add");
  odla_SetValueAsOutput(Add);
}

void test_bind_funciton_multithread(float* in, float* out) 
{
  odla_context ctx_multithread;
  CHECK_EQ(odla_CreateContext(&ctx_multithread), ODLA_SUCCESS);
  CHECK_EQ(odla_BindToArgumentById((const odla_value_id)"Input", in, ctx_multithread), ODLA_SUCCESS);
  CHECK_EQ(odla_BindToOutputById((const odla_value_id) "Add", out, ctx_multithread), ODLA_SUCCESS);
  odla_DestroyContext(ctx_multithread);

}

void execute_multithread(odla_computation comp, float* in, float* out)
{
    odla_context ctx_multithread;
    odla_CreateContext(&ctx_multithread);

    odla_BindToArgumentById((const odla_value_id) "Input", in, ctx_multithread);
    odla_BindToOutputById((const odla_value_id) "Add", out, ctx_multithread);
    odla_ExecuteComputation(comp, ctx_multithread, ODLA_COMPUTE_INFERENCE, nullptr);

    odla_DestroyContext(ctx_multithread);
}

 json default_json2(
        float amp = 0.6,
        const std::string& sdk_version = popart::core::packageHash(),
        const std::string& version = std::string("1.0.0"), int batches_per_step = 1,
        int ipu_num = 1, bool save_model = false,
        const std::string& save_model_path = std::string("odla_popart_saved.onnx"),
        bool load_onnx = false,
        const std::string& load_onnx_path = std::string("not_set.onnx"),
        const std::string& execution_mode = std::string("sequence"),
        const std::string& queue_type = std::string("LockFreeQueue"),
        int queue_capacity = 1024 * 1024, bool debug = false) {
      // Create a json object & fill with default value
      json jsonfile;
      jsonfile["amp"] = amp;
      jsonfile["sdk_version"] = sdk_version;
      jsonfile["version"] = version;
      jsonfile["batches_per_step"] = batches_per_step;
      jsonfile["ipu_num"] = ipu_num;
      jsonfile["save_model"] = save_model;
      jsonfile["save_model_path"] = save_model_path;
      jsonfile["load_onnx"] = false;
      jsonfile["load_onnx_path"] = load_onnx_path;
      jsonfile["execution_mode"] = execution_mode;
      jsonfile["queue_type"] = queue_type;
      jsonfile["queue_capacity"] = queue_capacity;
      jsonfile["debug"] = debug;

      json pipeline;
      std::vector<int> vec1, vec2;
      vec1.push_back(0);
      vec1.push_back(0);
      vec2.push_back(1);
      vec2.push_back(1);
      pipeline["Input"] = vec1;
      pipeline["Mul"] = vec2;
      pipeline["Mul_const"] = vec2;
      pipeline["Add"] = vec2;
      pipeline["Add_const"] = vec2;

      jsonfile["pipeline"] = pipeline;

      std::ofstream file("/tmp/tmp.json");
      file << jsonfile;
      return jsonfile;
    }

 json default_json(
        float amp = 0.6,
        const std::string& sdk_version = popart::core::packageHash(),
        const std::string& version = std::string("1.0.0"), int batches_per_step = 1,
        int ipu_num = 1, bool save_model = false,
        const std::string& save_model_path = std::string("odla_popart_saved.onnx"),
        bool load_onnx = false,
        const std::string& load_onnx_path = std::string("not_set.onnx"),
        const std::string& execution_mode = std::string("sequence"),
        const std::string& queue_type = std::string("LockFreeQueue"),
        int queue_capacity = 1024 * 1024, bool debug = false) {
      // Create a json object & fill with default value
      json jsonfile;
      jsonfile["amp"] = amp;
      jsonfile["sdk_version"] = sdk_version;
      jsonfile["version"] = version;
      jsonfile["batches_per_step"] = batches_per_step;
      jsonfile["ipu_num"] = ipu_num;
      jsonfile["save_model"] = save_model;
      jsonfile["save_model_path"] = save_model_path;
      jsonfile["load_onnx"] = false;
      jsonfile["load_onnx_path"] = load_onnx_path;
      jsonfile["execution_mode"] = execution_mode;
      jsonfile["queue_type"] = queue_type;
      jsonfile["queue_capacity"] = queue_capacity;
      jsonfile["debug"] = debug;

      json pipeline;
      jsonfile["pipeline"] = pipeline;

      std::ofstream file("/tmp/tmp.json");
      file << jsonfile;
      file.close();

      return jsonfile;
    }

TEST_CASE("MATH OPS TESTING") {
    
    SUBCASE("MATH OPS ABS TEST") {
    
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);


    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto abs_value = odla_Abs(input, (const odla_value_id) "Abs");
    odla_SetValueAsOutput(abs_value);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, -3.0, -10.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_Abs[2 * 2] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Abs", out_Abs, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Abs = [";
    for (int i = 0; i < 4; i++) {
      std::cout << out_Abs[i] << ", ";
    }
    std::cout << "]" << std::endl;

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);

  }
    
    SUBCASE("MATH OPS ARG MIN TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));
    odla_int32 axis = 0;
    odla_bool keep_dims = 1;
    odla_bool return_last_index = 0;

    odla_value_type unused_arg;
    auto ArgMin = odla_ArgMin(input, axis, keep_dims, return_last_index,
                              unused_arg, (const odla_value_id) "ArgMin");
    odla_SetValueAsOutput(ArgMin);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, 3.0, 10.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    int32_t out_ArgMin[2] = {-1, -1};
    odla_BindToOutputById((const odla_value_id) "ArgMin", out_ArgMin, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_ArgMin = [";
    for (int i = 0; i < 2; i++) {
      std::cout << out_ArgMin[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }
    
    SUBCASE("MATH OPS CEIL TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);


    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Ceil = odla_Ceil(input, (const odla_value_id) "Ceil");
    odla_SetValueAsOutput(Ceil);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, 3.5, 10.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_Ceil[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Ceil", out_Ceil, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Ceil = [";
    for (int i = 0; i < 4; i++) {
      std::cout << out_Ceil[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

    SUBCASE("MATH OPS CLAMP TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);


    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {3, 3}}},
                            (const odla_value_id)("input"));

    float lo_data = 3.0;
    float hi_data = 5.0;

    auto Clamp =
        odla_Clamp(input, lo_data, hi_data, (const odla_value_id) "Clamp");
    odla_SetValueAsOutput(Clamp);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[3 * 3] = {2.0, 1.0, 3.5, 10.0, 4.3, 5.8, 9.0, 12.0, 100.3};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_Clamp[3 * 3] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Clamp", out_Clamp, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Clamp = [";
    for (int i = 0; i < 9; i++) {
        std::cout << out_Clamp[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }

    SUBCASE("MATH OPS EQUAL TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto lhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("lhs"));

    auto rhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("rhs"));

    auto Equal = odla_Equal(lhs, rhs, (const odla_value_id) "Equal");
    odla_SetValueAsOutput(Equal);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float lhs_data[2 * 2] = {2.0, 1.0, 3.5, 10.0};
    odla_BindToArgumentById((const odla_value_id) "lhs", lhs_data, ctx);

    float rhs_data[2 * 2] = {2.0, 1.0, 3.5, 9.0};
    odla_BindToArgumentById((const odla_value_id) "rhs", rhs_data, ctx);

    bool out_Equal[2 * 2] = {false, false, false, false};
    odla_BindToOutputById((const odla_value_id) "Equal", out_Equal, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Equal = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Equal[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
    
    SUBCASE("MATH OPS EXP TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Exp = odla_Exp(input, (const odla_value_id) "Exp");
    odla_SetValueAsOutput(Exp);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_Exp[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Exp", out_Exp, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Exp = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Exp[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
   
    SUBCASE("MATH OPS GREATER TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto lhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("lhs"));

    auto rhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("rhs"));

    auto Greater = odla_Greater(lhs, rhs, (const odla_value_id) "Greater");
    odla_SetValueAsOutput(Greater);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float lhs_data[2 * 2] = {2.3, 1.0, 3.5, 5.5};
    odla_BindToArgumentById((const odla_value_id) "lhs", lhs_data, ctx);

    float rhs_data[2 * 2] = {2.0, 1.5, 4.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "rhs", rhs_data, ctx);

    bool out_Greater[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Greater", out_Greater, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Greater = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Greater[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
   
    SUBCASE("MATH OPS LESS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto lhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("lhs"));

    auto rhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("rhs"));

    auto Less = odla_Less(lhs, rhs, (const odla_value_id) "Less");
    odla_SetValueAsOutput(Less);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float lhs_data[2 * 2] = {2.3, 1.0, 3.5, 5.5};
    odla_BindToArgumentById((const odla_value_id) "lhs", lhs_data, ctx);

    float rhs_data[2 * 2] = {2.0, 1.5, 4.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "rhs", rhs_data, ctx);

    bool out_Less[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Less", out_Less, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Less = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Less[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
   
    SUBCASE("MATH OPS LOG TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Log = odla_Log(input, (const odla_value_id) "Log");
    odla_SetValueAsOutput(Log);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_Log[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Log", out_Log, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Log = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Log[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }

    SUBCASE("MATH OPS MAX TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto lhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("lhs"));

    auto rhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("rhs"));

    auto Max = odla_Max(lhs, rhs, (const odla_value_id) "Max");
    odla_SetValueAsOutput(Max);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float lhs_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "lhs", lhs_data, ctx);

    float rhs_data[2 * 2] = {4.0, 0.9, -3.5, 588888.0};
    odla_BindToArgumentById((const odla_value_id) "rhs", rhs_data, ctx);

    float out_Max[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Max", out_Max, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Max = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Max[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }

    SUBCASE("MATH OPS MIN TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto lhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("lhs"));

    auto rhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("rhs"));

    auto Min = odla_Min(lhs, rhs, (const odla_value_id) "Min");
    odla_SetValueAsOutput(Min);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float lhs_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "lhs", lhs_data, ctx);

    float rhs_data[2 * 2] = {4.0, 0.9, -3.5, 588888.0};
    odla_BindToArgumentById((const odla_value_id) "rhs", rhs_data, ctx);

    float out_Min[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Min", out_Min, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Min = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Min[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }

    SUBCASE("MATH OPS MEAN TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto input_1 =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input_1"));

    auto input_2 =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input_2"));

    auto input_3 =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input_3"));

    //  std::vector<odla_value> input_vec{input_1, input_2, input_3};
    odla_values inputs{.size = 3, .values = {input_1, input_2, input_3}};
    auto Mean = odla_Mean(inputs, (const odla_value_id) "Mean");
    odla_SetValueAsOutput(Mean);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data_1 = {0.1, 0.2, 0.3, 0.4};
    odla_BindToArgumentById((const odla_value_id) "input_1",
                            input_data_1.data(), ctx);

    std::vector<float> input_data_2 = {0.5, 0.6, 0.7, 0.8};
    odla_BindToArgumentById((const odla_value_id) "input_2",
                            input_data_2.data(), ctx);

    std::vector<float> input_data_3 = {0.9, 1.0, 1.5, 1.8};
    odla_BindToArgumentById((const odla_value_id) "input_3",
                            input_data_3.data(), ctx);
    // float input_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    // odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_Mean[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Mean", out_Mean, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Mean = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Mean[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }

    SUBCASE("MATH OPS NEG TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Neg = odla_Neg(input, (const odla_value_id) "Neg");
    odla_SetValueAsOutput(Neg);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_Neg[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Neg", out_Neg, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Neg = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Neg[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }

    SUBCASE("MATH OPS NOT TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto input = odla_CreateArgument({ODLA_BOOL, {.size = 2, .dims = {2, 2}}},
                                        (const odla_value_id)("input"));

    auto Not = odla_Not(input, (const odla_value_id) "Not");
    odla_SetValueAsOutput(Not);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    bool input_data[2 * 2] = {true, false, true, false};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    bool out_Not[4] = {false, false, false, false};
    odla_BindToOutputById((const odla_value_id) "Not", out_Not, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Not = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Not[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }

    SUBCASE("MATH OPS POW TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto lhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("lhs"));

    auto rhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("rhs"));

    auto Pow = odla_Pow(lhs, rhs, (const odla_value_id) "Pow");
    odla_SetValueAsOutput(Pow);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float lhs_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "lhs", lhs_data, ctx);

    float rhs_data[2 * 2] = {4.0, 0.9, -2.0, 4.0};
    odla_BindToArgumentById((const odla_value_id) "rhs", rhs_data, ctx);

    float out_Pow[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Pow", out_Pow, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Pow = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Pow[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
   
    SUBCASE("MATH OPS RECIPROCAL TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Reciprocal =
        odla_Reciprocal(input, (const odla_value_id) "Reciprocal");
    odla_SetValueAsOutput(Reciprocal);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {-0.2, -4, 8, 9};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_Reciprocal[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Reciprocal", out_Reciprocal,
                            ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Reciprocal = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Reciprocal[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
   
    SUBCASE("MATH OPS REDUCEMAX TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    odla_size_t num_of_axes = 1;
    odla_bool keep_dims = 1;
    odla_uint32 axes[1] = {1};
    odla_value_shape output_dims;

    auto ReduceMax =
        odla_ReduceMax(input, num_of_axes, axes, keep_dims, output_dims,
                        (const odla_value_id) "ReduceMax");
    odla_SetValueAsOutput(ReduceMax);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {-0.2, -4, 8, 9};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_ReduceMax[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "ReduceMax", out_ReduceMax,
                            ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_ReduceMax = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_ReduceMax[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
   
    SUBCASE("MATH OPS REDUCEMIN TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    odla_size_t num_of_axes = 1;
    odla_bool keep_dims = 2;
    odla_uint32 axes[2] = {1, 0};
    odla_value_shape output_dims;

    auto ReduceMin =
        odla_ReduceMin(input, num_of_axes, axes, keep_dims, output_dims,
                        (const odla_value_id) "ReduceMin");
    odla_SetValueAsOutput(ReduceMin);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {-0.2, -4, 8, 9};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_ReduceMin[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "ReduceMin", out_ReduceMin,
                            ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_ReduceMin = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_ReduceMin[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
   
    SUBCASE("MATH OPS REDUCEPROD TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    odla_size_t num_of_axes = 1;
    odla_bool keep_dims = 2;
    odla_uint32 axes[1] = {0};
    odla_value_shape output_dims;

    auto ReduceProd =
        odla_ReduceProd(input, num_of_axes, axes, keep_dims, output_dims,
                        (const odla_value_id) "ReduceProd");
    odla_SetValueAsOutput(ReduceProd);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {-0.2, -4, 8, 9};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_ReduceProd[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "ReduceProd", out_ReduceProd,
                            ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_ReduceProd = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_ReduceProd[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
   
    SUBCASE("MATH OPS REDUCESUM TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    odla_size_t num_of_axes = 1;
    odla_bool keep_dims = 2;
    odla_uint32 axes[1] = {0};
    odla_value_shape output_dims;

    auto ReduceSum =
        odla_ReduceSum(input, num_of_axes, axes, keep_dims, output_dims,
                        (const odla_value_id) "ReduceSum");
    odla_SetValueAsOutput(ReduceSum);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {-0.2, -4, 8, 9};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_ReduceSum[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "ReduceSum", out_ReduceSum,
                            ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_ReduceSum = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_ReduceSum[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }

    SUBCASE("MATH OPS SIGN TEST") {
      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
       set_computationItem(comp, 1);

      auto input =
          odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                              (const odla_value_id)("input"));

      auto Sign = odla_Sign(input, (const odla_value_id) "Sign");
      odla_SetValueAsOutput(Sign);

      static odla_context ctx;
      odla_CreateContext(&ctx);

      float input_data[2 * 2] = {-0.2, -0.3, 1, 0.5};
      odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

      float out_Sign[4] = {0, 0, 0, 0};
      odla_BindToOutputById((const odla_value_id) "Sign", out_Sign, ctx);

      odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

      std::cout << "out_Sign = [";
      for (int i = 0; i < 4; i++) {
        std::cout << out_Sign[i] << ", ";
      }
      std::cout << "]" << std::endl;

      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);

    }

    SUBCASE("MATH OPS AND TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);

    auto lhs = odla_CreateArgument({ODLA_BOOL, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("lhs"));

    auto rhs = odla_CreateArgument({ODLA_BOOL, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("rhs"));

    auto And = odla_And(lhs, rhs, (const odla_value_id) "And");
    odla_SetValueAsOutput(And);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    bool lhs_data[2 * 2] = {false, false, true, true};
    odla_BindToArgumentById((const odla_value_id) "lhs", lhs_data, ctx);

    bool rhs_data[2 * 2] = {true, false, true, false};
    odla_BindToArgumentById((const odla_value_id) "rhs", rhs_data, ctx);
    
    bool gold[2 * 2] = {false, false, true, false};
    
    bool out_And[2 * 2] = {false, false, false, false};
    odla_BindToOutputById((const odla_value_id) "And", out_And, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_And = [";
    for (int i = 0; i < 4; i++) {
        CHECK_EQ(out_And[i], gold[i]);
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
    
  
      SUBCASE("MATH OPS OR TEST"){
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);

    auto lhs = odla_CreateArgument({ODLA_BOOL, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("lhs"));

    auto rhs = odla_CreateArgument({ODLA_BOOL, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("rhs"));

    auto Or = odla_Or(lhs, rhs, (const odla_value_id) "Or");
    odla_SetValueAsOutput(Or);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    bool lhs_data[2 * 2] = {false, false, true, true};
    odla_BindToArgumentById((const odla_value_id) "lhs", lhs_data, ctx);

    bool rhs_data[2 * 2] = {true, false, true, false};
    odla_BindToArgumentById((const odla_value_id) "rhs", rhs_data, ctx);

    bool gold[2 * 2] = {true, false, true, true};
      
    bool out_Or[2 * 2] = {false, false, false, false};
    odla_BindToOutputById((const odla_value_id) "Or", out_Or, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Or = [";
    for (int i = 0; i < 4; i++) {
        CHECK_EQ(out_Or[i], gold[i]);
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }

        SUBCASE("MATH OPS NOT EQUAL TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);

    auto lhs = odla_CreateArgument({ODLA_BOOL, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("lhs"));

    auto rhs = odla_CreateArgument({ODLA_BOOL, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("rhs"));

    auto NotEqual = odla_NotEqual(lhs, rhs, (const odla_value_id) "NotEqual");
    odla_SetValueAsOutput(NotEqual);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    bool lhs_data[2 * 2] = {false, false, true, true};
    odla_BindToArgumentById((const odla_value_id) "lhs", lhs_data, ctx);

    bool rhs_data[2 * 2] = {true, false, true, false};
    odla_BindToArgumentById((const odla_value_id) "rhs", rhs_data, ctx);
      
    bool gold[2 * 2] = {true, false, false, true};
      
    bool out_Equal[2 * 2] = {false, false, false, false};
    odla_BindToOutputById((const odla_value_id) "NotEqual", out_Equal, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Equal = [";
    for (int i = 0; i < 4; i++) {
        CHECK_EQ(out_Equal[i], gold[i]);
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
    
}


TEST_CASE("OPS TESTING") {
      SUBCASE("OPS Sub TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto lhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("lhs"));

    auto rhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("rhs"));

    auto Sub = odla_Add(lhs, rhs, (const odla_value_id) "Sub");
    odla_SetValueAsOutput(Sub);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float lhs_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "lhs", lhs_data, ctx);

    float rhs_data[2 * 2] = {4.0, 0.9, -2.0, 4.0};
    odla_BindToArgumentById((const odla_value_id) "rhs", rhs_data, ctx);

    float out_Sub[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Sub", out_Sub, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Sub = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Sub[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }
   
    SUBCASE("OPS Div TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto lhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("lhs"));

    auto rhs = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                                    (const odla_value_id)("rhs"));

    auto Sub = odla_Add(lhs, rhs, (const odla_value_id) "Sub");
    odla_SetValueAsOutput(Sub);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float lhs_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "lhs", lhs_data, ctx);

    float rhs_data[2 * 2] = {4.0, 0.9, -2.0, 4.0};
    odla_BindToArgumentById((const odla_value_id) "rhs", rhs_data, ctx);

    float out_Sub[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Sub", out_Sub, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Sub = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Sub[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }

    SUBCASE("OPS Floor TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Neg = odla_Floor(input, (const odla_value_id) "Neg");
    odla_SetValueAsOutput(Neg);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_Neg[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Neg", out_Neg, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Floor = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Neg[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }

    SUBCASE("OPS Sqrt TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Neg = odla_Sqrt(input, (const odla_value_id) "Neg");
    odla_SetValueAsOutput(Neg);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_Neg[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Neg", out_Neg, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Floor = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Neg[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }    

    SUBCASE("OPS Rsqrt TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Neg = odla_Rsqrt(input, (const odla_value_id) "Neg");
    odla_SetValueAsOutput(Neg);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_Neg[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Neg", out_Neg, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Floor = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Neg[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }

    SUBCASE("OPS Relu TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
     set_computationItem(comp, 1);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    auto Neg = odla_Relu(input, (const odla_value_id) "Neg");
    odla_SetValueAsOutput(Neg);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[2 * 2] = {2.0, 1.0, 3.5, 5.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_Neg[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Neg", out_Neg, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Floor = [";
    for (int i = 0; i < 4; i++) {
        std::cout << out_Neg[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
    }

   

}

TEST_CASE("NN OPS TESTING") {
  SUBCASE("AVERAGEPOOL OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 4, .dims = {1, 1, 4, 4}}},
                            (const odla_value_id)("input"));

    odla_memory_layout unused_layout;
    odla_uint32 dims[2] = {3, 3};
    odla_uint32 padding_front[2] = {0, 0};
    odla_uint32 padding_back[2] = {0, 0};
    odla_uint32 strides[2] = {1, 1};
    odla_value_shape output_dims;
    auto AveragePool = odla_AveragePool(
        input, unused_layout, dims, strides, padding_front, padding_back,
        output_dims, (const odla_value_id) "AveragePool");
    odla_SetValueAsOutput(AveragePool);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {1, 2,  3,  4,  5,  6,  7,  8,
                                     9, 10, 11, 12, 13, 14, 15, 16};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    float out_AveragePool[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "AveragePool", out_AveragePool,
                          ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_AveragePool = [";
    for (int i = 0; i < 4; i++) {
      std::cout << out_AveragePool[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("BATCHNORMALIZATION OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 4, .dims = {1, 2, 1, 3}}},
                            (const odla_value_id)("input"));

    auto scale = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {2}}},
                                     (const odla_value_id)("scale"));

    auto offset = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {2}}},
                                      (const odla_value_id)("offset"));

    auto mean = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {2}}},
                                    (const odla_value_id)("mean"));

    auto var = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {2}}},
                                   (const odla_value_id)("var"));

    odla_memory_layout unused_layout;
    odla_value_shape output_dims;
    float epsilon = 1e-5;
    float scalar_scale = 1;
    float scalar_offset = 1;
    auto BatchNormalization = odla_BatchNormalization(
        input, unused_layout, mean, var, epsilon, scale, offset, scalar_scale,
        scalar_offset, (const odla_value_id) "BatchNormalization");
    odla_SetValueAsOutput(BatchNormalization);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {-1, 0, 1, 2, 3, 4};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    std::vector<float> scale_data = {1.0, 1.5};
    odla_BindToArgumentById((const odla_value_id) "scale", scale_data.data(),
                            ctx);

    std::vector<float> offset_data = {0, 1};
    odla_BindToArgumentById((const odla_value_id) "offset", offset_data.data(),
                            ctx);

    std::vector<float> mean_data = {0, 3};
    odla_BindToArgumentById((const odla_value_id) "mean", mean_data.data(),
                            ctx);

    std::vector<float> var_data = {1, 1.5};
    odla_BindToArgumentById((const odla_value_id) "var", var_data.data(), ctx);

    float out_BatchNormalization[6] = {0, 0, 0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "BatchNormalization",
                          out_BatchNormalization, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_BatchNormalization = [";
    for (int i = 0; i < 6; i++) {
      std::cout << out_BatchNormalization[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("CONV OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 4, .dims = {1, 1, 5, 5}}},
                            (const odla_value_id)("input"));

    auto kernel =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 4, .dims = {1, 1, 3, 3}}},
                            (const odla_value_id)("kernel"));

    odla_memory_layout unused_layout;
    odla_uint32 padding_front[2] = {1, 1};
    odla_uint32 padding_back[2] = {1, 1};
    odla_uint32 strides[2] = {1, 1};
    odla_uint32 dilations[2] = {1, 1};
    odla_value_shape output_dims;
    auto Conv = odla_Conv(input, unused_layout,
                          1, // group
                          kernel, unused_layout, strides, dilations,
                          padding_front, padding_back,
                          NULL,        // bias, unused
                          output_dims, // unused
                          (const odla_value_id) "Conv");
    odla_SetValueAsOutput(Conv);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {0,  1,  2,  3,  4,  5,  6,  7,  8,
                                     9,  10, 11, 12, 13, 14, 15, 16, 17,
                                     18, 19, 20, 21, 22, 23, 24};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    std::vector<float> kernel_data = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    odla_BindToArgumentById((const odla_value_id) "kernel", kernel_data.data(),
                            ctx);

    float out_Conv[25] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Conv", out_Conv, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Conv = [";
    for (int i = 0; i < 25; i++) {
      std::cout << out_Conv[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("DECONV OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 4, .dims = {1, 1, 3, 3}}},
                            (const odla_value_id)("input"));

    auto kernel =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 4, .dims = {1, 2, 3, 3}}},
                            (const odla_value_id)("kernel"));

    odla_memory_layout unused_layout;
    odla_uint32 padding_front[2] = {0, 0};
    odla_uint32 padding_back[2] = {0, 0};
    odla_uint32 strides[2] = {1, 1};
    odla_uint32 dilations[2] = {1, 1};
    odla_value_shape output_dims;
    odla_uint32 group = 1;
    auto DeConv = odla_DeConv(input, unused_layout,
                              group, // group
                              kernel, unused_layout, strides, dilations,
                              padding_front, padding_back,
                              NULL,        // bias, unused
                              output_dims, // unused
                              (const odla_value_id) "DeConv");
    odla_SetValueAsOutput(DeConv);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    std::vector<float> kernel_data = {1, 1, 1, 1, 1, 1, 1, 1, 1,
                                      1, 1, 1, 1, 1, 1, 1, 1, 1};
    odla_BindToArgumentById((const odla_value_id) "kernel", kernel_data.data(),
                            ctx);

    float out_DeConv[60] = {0};
    odla_BindToOutputById((const odla_value_id) "DeConv", out_DeConv, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_DeConv = [";
    for (int i = 0; i < 50; i++) {
      std::cout << out_DeConv[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("ELU OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);

    auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {3}}},
                                     (const odla_value_id)("input"));

    float alpha = 2.0;
    auto Elu = odla_Elu(input, alpha, (const odla_value_id) "Elu");
    odla_SetValueAsOutput(Elu);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[3] = {-1, 0, 1};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_Elu[3] = {0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Elu", out_Elu, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Elu = [";
    for (int i = 0; i < 3; i++) {
      std::cout << out_Elu[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("HARDSIGMOID OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);

    auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {3}}},
                                     (const odla_value_id)("input"));

    float alpha = 0.5;
    float beta = 0.6;
    auto HardSigmoid = odla_HardSigmoid(input, alpha, beta,
                                        (const odla_value_id) "HardSigmoid");
    odla_SetValueAsOutput(HardSigmoid);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[3] = {-1, 0, 1};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_HardSigmoid[3] = {0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "HardSigmoid", out_HardSigmoid,
                          ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_HardSigmoid = [";
    for (int i = 0; i < 3; i++) {
      std::cout << out_HardSigmoid[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("INSTANCENORMALIZATION OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 4, .dims = {1, 2, 1, 3}}},
                            (const odla_value_id)("input"));

    auto scale = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {2}}},
                                     (const odla_value_id)("scale"));

    auto offset = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {2}}},
                                      (const odla_value_id)("offset"));

    auto mean = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {2}}},
                                    (const odla_value_id)("mean"));

    auto var = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {2}}},
                                   (const odla_value_id)("var"));

    odla_memory_layout unused_layout;
    odla_value_shape output_dims;
    float epsilon = 1e-5;
    float scalar_scale = 1;
    float scalar_offset = 1;
    auto InstanceNormalization = odla_InstanceNormalization(
        input, unused_layout, mean, var, epsilon, scale, offset, scalar_scale,
        scalar_offset, (const odla_value_id) "InstanceNormalization");
    odla_SetValueAsOutput(InstanceNormalization);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {-1, 0, 1, 2, 3, 4};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    std::vector<float> scale_data = {1.0, 1.5};
    odla_BindToArgumentById((const odla_value_id) "scale", scale_data.data(),
                            ctx);

    std::vector<float> offset_data = {0, 1};
    odla_BindToArgumentById((const odla_value_id) "offset", offset_data.data(),
                            ctx);

    // std::vector<float> mean_data = {0, 3};
    // odla_BindToArgumentById((const odla_value_id) "mean", mean_data.data(),
    // ctx);

    // std::vector<float> var_data = {1, 1.5};
    // odla_BindToArgumentById((const odla_value_id) "var", var_data.data(),
    // ctx);

    float out_InstanceNormalization[6] = {0, 0, 0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "InstanceNormalization",
                          out_InstanceNormalization, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_InstanceNormalization = [";
    for (int i = 0; i < 6; i++) {
      std::cout << out_InstanceNormalization[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("LEAKYRELU OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);

    auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {3}}},
                                     (const odla_value_id)("input"));

    float alpha = 0.1;
    auto LeakyRelu =
        odla_LeakyRelu(input, alpha, (const odla_value_id) "LeakyRelu");
    odla_SetValueAsOutput(LeakyRelu);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[3] = {-1, 0, 1};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_LeakyRelu[3] = {0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "LeakyRelu", out_LeakyRelu,
                          ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_LeakyRelu = [";
    for (int i = 0; i < 3; i++) {
      std::cout << out_LeakyRelu[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("LOGSOFTMAX OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {1, 3}}},
                            (const odla_value_id)("input"));

    int axis = 1;
    auto LogSoftmax =
        odla_LogSoftmax(input, axis, (const odla_value_id) "LogSoftmax");
    odla_SetValueAsOutput(LogSoftmax);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    float input_data[3] = {-1, 0, 1};
    odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

    float out_LogSoftmax[3] = {0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "LogSoftmax", out_LogSoftmax,
                          ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_LogSoftmax = [";
    for (int i = 0; i < 3; i++) {
      std::cout << out_LogSoftmax[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("LSTM OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 3, .dims = {1, 3, 2}}},
                            (const odla_value_id)("input"));

    auto W =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 3, .dims = {1, 12, 2}}},
                            (const odla_value_id)("W"));

    auto R =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 3, .dims = {1, 12, 3}}},
                            (const odla_value_id)("R"));

    auto B = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {1, 24}}},
                                 (const odla_value_id)("B"));
    int input_size = 2;
    int hidden_size = 3;
    float weight_scale = 0.1;
    int number_of_gates = 4;
    float seq_len = 100;
    odla_rnn_direction direction = ODLA_RNN_FORWARD;
    odla_rnn_outputs rnn_outputs = ODLA_RNN_NO_STATE;
    auto LSTM = odla_LSTM(
        input,
        {.size = 3, .dims = {1, number_of_gates * hidden_size, input_size}}, W,
        R, B, seq_len, hidden_size, direction, rnn_outputs,
        (const odla_value_id) "LSTM");

    odla_SetValueAsOutput(LSTM.values[0]);
    // odla_SetValuesAsOutput(LSTM);
    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    std::vector<float> W_data(24, 0.1);
    odla_BindToArgumentById((const odla_value_id) "W", W_data.data(), ctx);

    std::vector<float> R_data(36, 0.1);
    odla_BindToArgumentById((const odla_value_id) "R", R_data.data(), ctx);

    std::vector<float> B_data(24, 0);
    odla_BindToArgumentById((const odla_value_id) "B", B_data.data(), ctx);

    // std::vector<float> out_LSTM(75, 0);
    float out_LSTM[9] = {0};
    odla_BindToOutputById((const odla_value_id) "LSTM0", out_LSTM, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

      std::cout << "out_LSTM = [";
      for (int i = 0; i < 9; i++) {
        std::cout << out_LSTM[i] << ", ";
      }
      std::cout << "]" << std::endl;
      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);
    }
  
  SUBCASE("MAXPOOL OPS TEST") {
      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      set_computationItem(comp, 1);

      auto input =
          odla_CreateArgument({ODLA_FLOAT32, {.size = 4, .dims = {1, 1, 4, 4}}},
                              (const odla_value_id)("input"));

      odla_memory_layout unused_layout;
      odla_uint32 dims[2] = {3, 3};
      odla_uint32 padding_front[2] = {0, 0};
      odla_uint32 padding_back[2] = {0, 0};
      odla_uint32 strides[2] = {1, 1};
      odla_value_shape output_dims;
      auto MaxPool = odla_MaxPool(input, unused_layout, dims, strides,
                                  padding_front, padding_back, output_dims,
                                  (const odla_value_id) "MaxPool");
      odla_SetValueAsOutput(MaxPool);

      static odla_context ctx;
      odla_CreateContext(&ctx);

      std::vector<float> input_data = {1, 2,  3,  4,  5,  6,  7,  8,
                                       9, 10, 11, 12, 13, 14, 15, 16};
      odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                              ctx);

      float out_MaxPool[4] = {0, 0, 0, 0};
      odla_BindToOutputById((const odla_value_id) "MaxPool", out_MaxPool, ctx);

      odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

      std::cout << "out_MaxPool = [";
      for (int i = 0; i < 4; i++) {
        std::cout << out_MaxPool[i] << ", ";
      }
      std::cout << "]" << std::endl;
      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);
    }
  
  // SUBCASE("PRELU OPS TEST") {
  //     odla_computation comp;
  //     CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
  //     set_computationItem(comp, 1);

  //     auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {3}}},
  //                                      (const odla_value_id)("input"));
  //     float sloap = 1;
  //     auto PRelu = odla_PRelu(input, sloap, (const odla_value_id)("PRelu"));
  //     odla_SetValueAsOutput(PRelu);

  //     static odla_context ctx;
  //     odla_CreateContext(&ctx);

  //     float input_data[3] = {-1, 0, 1};
  //     odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

  //     float out_PRelu[3] = {0, 0, 0};
  //     odla_BindToOutputById((const odla_value_id) "PRelu", out_PRelu, ctx);

  //     odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

  //     std::cout << "out_PRelu = [";
  //     for (int i = 0; i < 3; i++) {
  //       std::cout << out_PRelu[i] << ", ";
  //     }
  //     std::cout << "]" << std::endl;
  //     odla_DestroyComputation(comp);
  //     odla_DestroyContext(ctx);
  //   }
  
  SUBCASE("SELU OPS TEST") {
      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      set_computationItem(comp, 1);

      auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {3}}},
                                       (const odla_value_id)("input"));
      float alpha = 2;
      float gamma = 3;
      auto Selu = odla_Selu(input, alpha, gamma, (const odla_value_id) "Selu");
      odla_SetValueAsOutput(Selu);

      static odla_context ctx;
      odla_CreateContext(&ctx);

      float input_data[3] = {-1, 0, 1};
      odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

      float out_Selu[3] = {0, 0, 0};
      odla_BindToOutputById((const odla_value_id) "Selu", out_Selu, ctx);

      odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

      std::cout << "out_Selu = [";
      for (int i = 0; i < 3; i++) {
        std::cout << out_Selu[i] << ", ";
      }
      std::cout << "]" << std::endl;
      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);
    }
  
  SUBCASE("SIGMOID OPS TEST") {
      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      set_computationItem(comp, 1);

      auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {3}}},
                                       (const odla_value_id)("input"));

      auto Sigmoid = odla_Sigmoid(input, (const odla_value_id) "Sigmoid");
      odla_SetValueAsOutput(Sigmoid);

      static odla_context ctx;
      odla_CreateContext(&ctx);

      float input_data[3] = {-1, 0, 1};
      odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

      float out_Sigmoid[3] = {0, 0, 0};
      odla_BindToOutputById((const odla_value_id) "Sigmoid", out_Sigmoid, ctx);

      odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

      std::cout << "out_Sigmoid = [";
      for (int i = 0; i < 3; i++) {
        std::cout << out_Sigmoid[i] << ", ";
      }
      std::cout << "]" << std::endl;
      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);
    }
  
  SUBCASE("TANH OPS TEST") {
      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      set_computationItem(comp, 1);

      auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {3}}},
                                       (const odla_value_id)("input"));

      auto Tanh = odla_Tanh(input, (const odla_value_id) "Tanh");
      odla_SetValueAsOutput(Tanh);

      static odla_context ctx;
      odla_CreateContext(&ctx);

      float input_data[3] = {-1, 0, 1};
      odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

      float out_Tanh[3] = {0, 0, 0};
      odla_BindToOutputById((const odla_value_id) "Tanh", out_Tanh, ctx);

      odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

      std::cout << "out_Tanh = [";
      for (int i = 0; i < 3; i++) {
        std::cout << out_Tanh[i] << ", ";
      }
      std::cout << "]" << std::endl;
      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);
    }
  
  // SUBCASE("TOPK OPS TEST") {
  //     odla_computation comp;
  //     CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
  //     set_computationItem(comp, 1);

  //     auto input =
  //         odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {3, 4}}},
  //                             (const odla_value_id)("input"));

  //     uint32_t axis = 1;
  //     uint32_t k = 1;
  //     odla_bool largest = true;
  //     odla_bool sorted = false;
  //     odla_value_type output_type;
  //     auto Topk = odla_TopK(input, k, largest, sorted, axis, output_type,
  //                           (const odla_value_id) "Topk");
  //     odla_SetValueAsOutput(Topk);

  //     static odla_context ctx;
  //     odla_CreateContext(&ctx);

  //     float input_data[3 * 4] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  //     odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

  //     float out_Topk[3 * 4] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  //     odla_BindToOutputById((const odla_value_id) "Topk", out_Topk, ctx);

  //     odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

  //     std::cout << "out_Topk = [";
  //     for (int i = 0; i < 3; i++) {
  //       std::cout << out_Topk[i] << ", ";
  //     }
  //     std::cout << "]" << std::endl;

  //     odla_DestroyComputation(comp);
  //     odla_DestroyContext(ctx);
  //   }
  
  SUBCASE("POSTPROCESS OPS TEST") {
      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      set_computationItem(comp, 1);

      auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {3}}},
                                       (const odla_value_id)("input"));

      auto Tanh = odla_Tanh(input, (const odla_value_id) "Tanh");
      odla_SetValueAsOutput(Tanh);

      static odla_context ctx;
      odla_CreateContext(&ctx);

      float input_data[3] = {-1, 0, 1};
      odla_BindToArgumentById((const odla_value_id) "input", input_data, ctx);

      float out_Tanh[3] = {0, 0, 0};
      odla_BindToOutputById((const odla_value_id) "Tanh", out_Tanh, ctx);

      odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

      std::cout << "out_Tanh = [";
      for (int i = 0; i < 3; i++) {
        std::cout << out_Tanh[i] << ", ";
      }
      std::cout << "]" << std::endl;
      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);
    }
   
  }

TEST_CASE("PROCESS OPS TESTING") {

   SUBCASE("CAST OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);


    auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {4}}},
                                     (const odla_value_id)("input"));

    odla_element_type element_type = ODLA_INT32;
    auto AveragePool =
        odla_Cast(input, element_type, (const odla_value_id) "Cast");
    odla_SetValueAsOutput(AveragePool);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {1.2, 2.3, 3.4, 4.5};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    int32_t out_Cast[4] = {0, 0, 0, 0};
    odla_BindToOutputById((const odla_value_id) "Cast", out_Cast, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Cast = [";
    for (int i = 0; i < 4; i++) {
      std::cout << out_Cast[i] << ", ";
    }
    std::cout << "]" << std::endl;

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }
   
   SUBCASE("CONCAT OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);


    auto input_1 =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 1}}},
                            (const odla_value_id)("input_1"));

    auto input_2 =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input_2"));

    auto input_3 =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 4}}},
                            (const odla_value_id)("input_3"));

    int axis = 1;
    odla_value_shape output_shape;
    auto Concat =
        odla_Concat({.size = 3, .values = {input_1, input_2, input_3}}, axis,
                    output_shape, (const odla_value_id) "Concat");
    odla_SetValueAsOutput(Concat);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data_1 = {5, 8};
    odla_BindToArgumentById((const odla_value_id) "input_1",
                            input_data_1.data(), ctx);

    std::vector<float> input_data_2 = {1, 3, 4, 7};
    odla_BindToArgumentById((const odla_value_id) "input_2",
                            input_data_2.data(), ctx);

    std::vector<float> input_data_3 = {1, 2, 3, 5, 7, 8, 9, 0};
    odla_BindToArgumentById((const odla_value_id) "input_3",
                            input_data_3.data(), ctx);

    float out_Concat[14] = {0};
    odla_BindToOutputById((const odla_value_id) "Concat", out_Concat, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Concat = [";
    for (int i = 0; i < 14; i++) {
      std::cout << out_Concat[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);

  }

   SUBCASE("EXPANDDIM OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);


    auto input = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                                     (const odla_value_id)("input"));

    int axis = 1;
    odla_value_shape output_shape{.size = 2, .dims = {1, 1}};
    //   odla_value_shape output_shape{.size=3, .dims={2, 1, 6}};
    auto ExpandDim = odla_ExpandDims(input, output_shape,
                                     (const odla_value_id) "ExpandDim");
    odla_SetValueAsOutput(ExpandDim);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {1.0};
    //   std::vector<float> input_data = {1, 2, 3};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    float out_ExpandDim[14] = {0};
    odla_BindToOutputById((const odla_value_id) "ExpandDim", out_ExpandDim,
                          ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    auto shape = comp->builder->getTensorShape(ExpandDim->tensor_id);
    std::cout << "result shape:[";
    for (int i = 0; i < shape.size(); ++i) {
      std::cout << shape[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);

    //   std::cout << "out_ExpandDim = [";
    //   for (int i = 0; i < 14; i++) {
    //     std::cout << out_ExpandDim[i] << ", ";
    //   }
    //   std::cout << "]" << std::endl;
  }
   
   SUBCASE("PAD OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);


    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 2}}},
                            (const odla_value_id)("input"));

    odla_uint32 padding_front[2] = {1, 1};
    odla_uint32 padding_back[2] = {1, 1};
    odla_value_shape output_dims;
    auto Pad = odla_Pad(input, padding_front, padding_back, output_dims,
                        (const odla_value_id) "Pad");
    odla_SetValueAsOutput(Pad);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {1, 2, 3, 4};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    float out_Pad[16] = {0};
    odla_BindToOutputById((const odla_value_id) "Pad", out_Pad, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Pad = [";
    for (int i = 0; i < 16; i++) {
      std::cout << out_Pad[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);

  }
   
   SUBCASE("RESIZE OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);


    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 4, .dims = {1, 1, 2, 4}}},
                            (const odla_value_id)("input"));
    odla_interpolation_mode interpolation = ODLA_NEAREST;
    odla_resize_coordinate_mode mode;
    odla_uint32 axes_mask;
    odla_value_shape output_dims{.size = 4, .dims = {1, 1, 4, 2}};
    auto Resize = odla_Resize(input, interpolation, mode, axes_mask,
                              output_dims, (const odla_value_id) "Resize");
    odla_SetValueAsOutput(Resize);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    float out_Resize[16] = {0};
    odla_BindToOutputById((const odla_value_id) "Resize", out_Resize, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    std::cout << "out_Resize = [";
    for (int i = 0; i < 16; i++) {
      std::cout << out_Resize[i] << ", ";
    }
    std::cout << "]" << std::endl;

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);

  }
   
   SUBCASE("SHAPE OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);

    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {2, 3}}},
                            (const odla_value_id)("input"));

    odla_value_shape output_dims = {.size = 2, .dims = {3, 2}};
    auto Reshape =
        odla_Reshape(input, output_dims, (const odla_value_id) "Reshape");

    odla_SetValueAsOutput(Reshape);
    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {1, 2, 3, 4, 5, 6};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    float out_Shape[16] = {0};
    odla_BindToOutputById((const odla_value_id) "Reshape", out_Shape, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);
    auto shape = comp->builder->getTensorShape(Reshape->tensor_id);

    // auto size = Reshape->tensor_id;
    std::cout << " shape:[";
    for (int i = 0; i < shape.size(); ++i) {
      std::cout << shape[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }
   
   SUBCASE("SQUEEZE OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);


    auto input =
        odla_CreateArgument({ODLA_FLOAT32, {.size = 3, .dims = {2, 1, 3}}},
                            (const odla_value_id)("input"));
    uint32_t axes_squeeze_num = 1;
    uint32_t axes_squeeze[1] = {1};
    odla_value_shape output_dims;
    auto Squeeze = odla_Squeeze(input, axes_squeeze_num, axes_squeeze,
                                output_dims, (const odla_value_id) "Squeeze");
    odla_SetValueAsOutput(Squeeze);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> input_data = {1, 2, 3, 4, 5, 6};
    odla_BindToArgumentById((const odla_value_id) "input", input_data.data(),
                            ctx);

    float out_Squeeze[16] = {0};
    odla_BindToOutputById((const odla_value_id) "Squeeze", out_Squeeze, ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);


    auto shape = comp->builder->getTensorShape(Squeeze->tensor_id);
    std::cout << "squeeze result shape:[";
    for (int i = 0; i < shape.size(); ++i) {
      std::cout << shape[i] << ", ";
    }
    std::cout << "]" << std::endl;
    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("SHAPE OPS TEST"){}
  
  SUBCASE("TILE OPS TEST"){}



}

TEST_CASE("GEMM OPS TESTING") {

  SUBCASE("GEMM OPS TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);

    auto a = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {4, 16}}},
                                 (const odla_value_id)("a"));
    auto b = odla_CreateArgument({ODLA_FLOAT32, {.size = 2, .dims = {16, 8}}},
                                 (const odla_value_id)("b"));

    auto result =
        odla_Gemm(a, 0, b, 0, 1, 0, nullptr, {.size = 2, .dims = {4, 8}},
                  (const odla_value_id) "Gemm");
    odla_SetValueAsOutput(result);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> a_data(4 * 16, 1.0);
    odla_BindToArgumentById((const odla_value_id) "a", a_data.data(), ctx);

    std::vector<float> b_data(16 * 8, 1.0);
    odla_BindToArgumentById((const odla_value_id) "b", b_data.data(), ctx);

    std::vector<float> result_data(4 * 8, 0);
    std::vector<float> expected(4 * 8, 16.0);
    odla_BindToOutputById((const odla_value_id) "Gemm", result_data.data(),
                          ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    // BOOST_TEST_MESSAGE("result_data = " << test::VecToStr(result_data));

    for (int i = 0; i < expected.size(); i++) {
      CHECK_EQ(result_data[i], expected[i]);
    }

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

  SUBCASE("GEMM TRANSPOSE TEST") {
    odla_computation comp;
    CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
    set_computationItem(comp, 1);

    auto a = odla_CreateArgument(
        {ODLA_FLOAT32, {.size = 4, .dims = {1, 12, 64, 64}}},
        (const odla_value_id)("a"));
    auto b = odla_CreateArgument(
        {ODLA_FLOAT32, {.size = 4, .dims = {1, 12, 64, 64}}},
        (const odla_value_id)("b"));

    auto result = odla_Gemm(a, 0, b, 1, 1, 0, nullptr, {.size = 0, .dims = {0}},
                            (const odla_value_id) "Gemm");

    odla_SetValueAsOutput(result);

    static odla_context ctx;
    odla_CreateContext(&ctx);

    std::vector<float> a_data(1 * 12 * 64 * 64, 1.0);
    odla_BindToArgumentById((const odla_value_id) "a", a_data.data(), ctx);

    std::vector<float> b_data(1 * 12 * 64 * 64, 1.0);
    odla_BindToArgumentById((const odla_value_id) "b", b_data.data(), ctx);

    std::vector<float> result_data(1 * 12 * 64 * 64, 0);

    odla_BindToOutputById((const odla_value_id) "Gemm", result_data.data(),
                          ctx);

    odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr);

    odla_DestroyComputation(comp);
    odla_DestroyContext(ctx);
  }

}


TEST_CASE("cache_file_test") 
{

  //  SUBCASE("test inject_error") {

  //     json _inject_error;
  //     _inject_error["POPLAR_ENGINE_OPTIONS"] = "{\"debug.simulateErrors\":\"MEMORY_ERROR@ALL:vertexName:popops__BroadcastScalar1DSupervisor___popops__expr__BinaryOpType__SUBTRACT_float\"}";
  //     std::ofstream file("/tmp/temp_error_injector.json");
  //     file << _inject_error;
  //     file.close();

  //     odla_computation comp;
  //     CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
  //     generate_model();
  //     set_computationItem(comp, 1);

  //     odla_context ctx;
  //     odla_CreateContext(&ctx);

  //     float in = 1.f, out = 0.f;
  //     odla_BindToArgumentById((const odla_value_id) "Input", &in, ctx);
  //     odla_BindToOutputById((const odla_value_id) "Sub", &out, ctx);
  //     CHECK_EQ(odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr), ODLA_RECOVERABLE_ERR);

  //     json _inject_error_void;
  //     std::ofstream file2("/tmp/temp_error_injector.json");
  //     file2 << _inject_error_void;
  //     file2.close();

  //     odla_DestroyComputation(comp);
  //     odla_DestroyContext(ctx);
  //  }

   SUBCASE("test exporting") {
      json _config_json = default_json();
      _config_json["amp"] = 0.123;
      std::ofstream file("./test.json");
      file << _config_json;
      file.close();

      std::string _path = "./test.popart";
      PopartConfig::instance()->set_cache_path(_path);
      PopartConfig::instance()->parse_from_json(_config_json);

      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      set_computationItem(comp, 1);

      generate_model();
      odla_context ctx;
      odla_CreateContext(&ctx);

      float in = 1.f;
      float out = 0.f;

      odla_BindToArgumentById((const odla_value_id) "Input", &in, ctx);
      odla_BindToOutputById((const odla_value_id) "Add", &out, ctx);

      comp->compile_and_export();

      odla_DestroyContext(ctx);
      odla_DestroyComputation(comp);
    }

   SUBCASE("test loading") {
     
      //cache file is not exist
      {
        std::string _path = "./wrong_path";
        PopartConfig::instance()->set_cache_path(_path);
        CHECK_EQ(PopartConfig::instance()->extract_config_from_cache(), ODLA_FAILURE) ;
      }

      //use cache (test.json)
      {
        std::string _path = "./test.popart";
        PopartConfig::instance()->set_cache_path(_path);
        CHECK_EQ(PopartConfig::instance()->extract_config_from_cache(), ODLA_SUCCESS);
        CHECK_EQ(PopartConfig::instance()->amp(), 0.123f);
      }
      //CHECK_EQ(PopartConfig::instance()->load_from_file("./test.popart"), ODLA_SUCCESS);

    }



    SUBCASE("osc"){
      bool use_ipu_model = 0;
      int ipu_num = 1;
      int batches_per_step = 1;
      char* cache_dir="./tmp";
      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));

      odla_SetComputationItem(comp, ODLA_CACHE_DIR, (odla_item_value)cache_dir);
      odla_SetComputationItem(comp, (odla_item_type) 1001, (odla_item_value)cache_dir);
      odla_SetComputationItem(comp, (odla_item_type) ODLA_ENABLE_ENGINE_CACHE, (odla_item_value)&ipu_num);


      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      odla_CreateContext(&ctx);
      odla_item_value _test;

      CHECK_EQ(odla_SetContextItem(ctx, ODLA_ASYNC_CALLBACK_ARG, (odla_item_value)&_test), ODLA_SUCCESS);
      CHECK_EQ(odla_SetContextItem(ctx, (odla_item_type)0, (odla_item_value)&_test), ODLA_UNSUPPORTED_DATATYPE);

      CHECK_EQ(odla_DestroyComputation(comp), ODLA_SUCCESS);
      CHECK_EQ(odla_DestroyContext(ctx), ODLA_SUCCESS);

    }


    SUBCASE("osc"){
      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      
      odla_context ctx;
      odla_CreateContext(&ctx);


      odla_executable exec;
      odla_CreateExecutable(&exec, ctx, comp );


      int a = 10;
      std::vector<int64_t> _shape(1, 1); 
      MakeNDArrayWrapper((odla_void*)&a, popart::DataType::FLOAT16, _shape);
      MakeNDArrayWrapper((odla_void*)&a, popart::DataType::UINT32, _shape);
      MakeNDArrayWrapper((odla_void*)&a, popart::DataType::BOOL, _shape);
      MakeNDArrayWrapper((odla_void*)&a, popart::DataType::INT64, _shape);
      MakeNDArrayWrapper((odla_void*)&a, popart::DataType::INT32, _shape);
      MakeNDArrayWrapper((odla_void*)&a, popart::DataType::FLOAT, _shape);

    }

    SUBCASE("test pipeline execute") 
    { 
      
      json _config_json = default_json2();
      _config_json["execution_mode"] = "pipeline_async";
      _config_json["ipu_num"] = 2;
      PopartConfig::instance()->parse_from_json(_config_json);

      odla_computation comp;
      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      model_helper();
      odla_CreateContext(&ctx);
      set_computationItem(comp, 2);

      float in = 1.f;
      float out = 0.f;
      odla_BindToArgumentById((const odla_value_id) "Input", &in, ctx);
      odla_BindToOutputById((const odla_value_id) "Add", &out, ctx);

      // CHECK_EQ(odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr), ODLA_SUCCESS);
      CHECK_EQ(odla_AsyncExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr), ODLA_SUCCESS);
      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);
    }

   SUBCASE("test config path") {
     
      //cache file is not exist
      {
        std::string _path = "./wrong_path";
        PopartConfig::instance()->set_cache_path(_path);
        CHECK_EQ(PopartConfig::instance()->extract_config_from_cache(), ODLA_FAILURE) ;
      }

      //use cache (test.json)
      {
        std::string _path = "./test.popart";
        PopartConfig::instance()->set_cache_path(_path);
        CHECK_EQ(PopartConfig::instance()->extract_config_from_cache(), ODLA_SUCCESS);
        CHECK_EQ(PopartConfig::instance()->amp(), 0.123f);
      }
      //CHECK_EQ(PopartConfig::instance()->load_from_file("./test.popart"), ODLA_SUCCESS);

    }

}


TEST_CASE("testing base interface") 
  {
    SUBCASE("test pipeline execute") 
    { 
      json _config_json = default_json2();
      _config_json["ipu_num"] = 2;
      _config_json["execution_mode"] = std::string("pipeline");

      PopartConfig::instance()->parse_from_json(_config_json);

      odla_computation comp;
      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      model_helper();
      odla_CreateContext(&ctx);
      set_computationItem(comp, 2);

      float in = 1.f;
      float out = 0.f;
      odla_BindToArgumentById((const odla_value_id) "Input", &in, ctx);
      odla_BindToOutputById((const odla_value_id) "Add", &out, ctx);

      CHECK_EQ(odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr), ODLA_SUCCESS);
      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);
    }

    SUBCASE("test execute function multithread") 
    {
      float in[3] = {1.f, 1.f, 1.f};
      float out[3] = {0.f};

      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      model_helper();
      set_computationItem(comp, 1);

      std::thread t1(execute_multithread, comp, &in[0], &out[0]);
      std::thread t2(execute_multithread, comp, &in[1], &out[1]);
      std::thread t3(execute_multithread, comp, &in[2], &out[2]);

      t1.join();
      t2.join();
      t3.join();

      CHECK_EQ(out[0], 5);
      CHECK_EQ(out[1], 5);
      CHECK_EQ(out[2], 5);

      odla_DestroyComputation(comp);
    }

    SUBCASE("test base function") 
    {
      odla_computation comp;
      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      odla_CreateContext(&ctx);
      set_computationItem(comp, 1);

      int wrong_addr;
      CHECK_EQ(odla_CreateComputation(&comp), ODLA_SUCCESS);
      // CHECK_EQ(odla_DestroyComputation(nullptr), ODLA_FAILURE); //todo 1: nullptr wrong addr protest 
      // CHECK_EQ(odla_DestroyComputation((odla_computation)&wrong_addr), ODLA_FAILURE);

      CHECK_EQ(odla_CreateContext(&ctx), ODLA_SUCCESS);
      // CHECK_EQ(odla_DestroyContext(nullptr), ODLA_FAILURE); //todo 1
      // CHECK_EQ(odla_DestroyContext((odla_context)&wrong_addr), ODLA_FAILURE);

      odla_item_value _test;
      // CHECK_EQ(odla_SetComputationItem(nullptr, ODLA_USE_SIM_MODE, (odla_item_value)&_test), ODLA_FAILURE); // todo: unvaild value, should be recognized failure
      // CHECK_EQ(odla_SetComputationItem((odla_computation)&wrong_addr, ODLA_USE_SIM_MODE, (odla_item_value)&_test), ODLA_FAILURE); //todo 1
      CHECK_EQ(odla_SetComputationItem(comp, ODLA_USE_SIM_MODE, (odla_item_value)&_test), ODLA_SUCCESS);
      CHECK_EQ(odla_SetComputationItem(comp, ODLA_LOAD_ENGINE_MODE, (odla_item_value)&_test), ODLA_UNSUPPORTED_DATATYPE);

      // CHECK_EQ(odla_SetContextItem(nullptr, ODLA_ASYNC_CALLBACK_FUNC, (odla_item_value)&_test), ODLA_INVALID_PARAM); //todo 1
      // CHECK_EQ(odla_SetContextItem((odla_context)&wrong_addr, ODLA_ASYNC_CALLBACK_FUNC, (odla_item_value)&_test), ODLA_INVALID_PARAM);
      CHECK_EQ(odla_SetContextItem(ctx, ODLA_ASYNC_CALLBACK_FUNC, (odla_item_value)&_test), ODLA_SUCCESS);

      CHECK_EQ(odla_DestroyComputation(comp), ODLA_SUCCESS);
      CHECK_EQ(odla_DestroyContext(ctx), ODLA_SUCCESS);

    }

    SUBCASE("test arg function") 
    {    
      
      odla_computation comp;
      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      odla_CreateContext(&ctx);
      set_computationItem(comp, 1);
      
      int wrong_addr;
      int data[5] = {0};
      odla_uint32 _num, _id;
      odla_value _ov;

      auto _input1 = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                          (const odla_value_id)("_input1"));
      auto _input2 = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                          (const odla_value_id)("_input2"));

      auto _constance = odla_CreateConstant({ODLA_FLOAT32, {.size = 1, .dims = {1}}}, &data,
                          (const odla_value_id) "_constance");

      auto _constance1 = odla_CreateConstant({ODLA_FLOAT32, {.size = 8, .dims = {1}}},
                          (int*)0x11, (const odla_value_id) "_constance");

      CHECK_EQ(odla_GetNumOfArgsFromComputation(comp, &_num), ODLA_SUCCESS);
      // CHECK_EQ(odla_GetNumOfArgsFromComputation(nullptr, &_num), ODLA_FAILURE); // todo 1
      // CHECK_EQ(odla_GetNumOfArgsFromComputation((odla_computation)&wrong_addr, &_num), ODLA_FAILURE);
      CHECK_EQ(_num, 2);

      CHECK_EQ(odla_GetArgFromComputationByIdx(comp, 0, &_ov), ODLA_SUCCESS);
      CHECK_EQ(odla_GetArgFromComputationByIdx(comp, 2, &_ov), ODLA_INVALID_PARAM);
      // CHECK_EQ(odla_GetArgFromComputationByIdx(nullptr, 0, &_ov), ODLA_FAILURE); //todo 1
      // CHECK_EQ(odla_GetArgFromComputationByIdx((odla_computation)&wrong_addr, 0, &_ov), ODLA_FAILURE);

      CHECK_EQ(odla_SetValueAsOutput(_input1), ODLA_SUCCESS);
      // CHECK_EQ(odla_SetValueAsOutput(_input1), ODLA_FAILURE); //todo: double set, should be failed
      CHECK_EQ(odla_SetValueAsOutput(_input2), ODLA_SUCCESS);

      odla_values _ovs = {2, {_input1, _input2}};
      // CHECK_EQ(odla_SetValuesAsOutput(_ovs), ODLA_FAILURE); //todo: duplicate set, should be failed
      CHECK_EQ(odla_GetNumOfOutputsFromComputation(comp, &_num), ODLA_SUCCESS);
      // CHECK_EQ(_num, 2); //todo: duplicate set, should be failed

      CHECK_EQ(odla_GetOutputFromComputationByIdx(comp, 0, &_ov), ODLA_SUCCESS);
      CHECK_EQ(odla_GetOutputFromComputationByIdx(comp, 2, &_ov), ODLA_INVALID_PARAM);
      // CHECK_EQ(odla_GetOutputFromComputationByIdx(nullptr, 0, &_ov), ODLA_FAILURE); //todo 1
      // CHECK_EQ(odla_GetOutputFromComputationByIdx((odla_computation)&wrong_addr, 0, &_ov), ODLA_FAILURE);

      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);

    }

    SUBCASE("test bind funtion") 
    {    
      odla_computation comp;
      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      model_helper();
      odla_CreateContext(&ctx);
      set_computationItem(comp, 1);

      int wrong_addr;
      float in = 1.f;
      float out = 1.f;

      // CHECK_EQ(odla_BindToArgumentById((const odla_value_id) "Input", &in, nullptr), ODLA_FAILURE); //todo 1
      // CHECK_EQ(odla_BindToArgumentById((const odla_value_id) "Input", &in, (odla_context)&wrong_addr), ODLA_FAILURE);
      CHECK_EQ(odla_BindToArgumentById((const odla_value_id) "Input", &in, ctx), ODLA_SUCCESS);
      // CHECK_EQ(odla_BindToArgumentById((const odla_value_id) "Input", &in, ctx), ODLA_FAILURE); // todo duplicate bind, should be recognized failure
      // CHECK_EQ(odla_BindToArgumentById((const odla_value_id) "Input", nullptr, ctx), ODLA_FAILURE); //todo 1

      // CHECK_EQ(odla_BindToOutputById((const odla_value_id) "Add", &out, nullptr), ODLA_FAILURE);
      // CHECK_EQ(odla_BindToOutputById((const odla_value_id) "Add", &out, (odla_context)&wrong_addr), ODLA_FAILURE); //todo 1
      CHECK_EQ(odla_BindToOutputById((const odla_value_id) "Add", &out, ctx), ODLA_SUCCESS);
      // CHECK_EQ(odla_BindToOutputById((const odla_value_id) "Add", &out, ctx), ODLA_FAILURE); //todo duplicate bind, should be recognized failure
      // CHECK_EQ(odla_BindToOutputById((const odla_value_id) "Add", nullptr, ctx), ODLA_FAILURE);

      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);
    }
    
    SUBCASE("test bind funtion multithread") 
    {
      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      model_helper();
      set_computationItem(comp, 1);

      std::thread threads[5];
      float in[5], out[5];

      for (int i = 0; i < 5; i++) {
        threads[i] = std::thread(test_bind_funciton_multithread, &in[i], &out[i]);
      }
      for (auto& t : threads) {
        t.join();
      }
      CHECK_EQ(ODLA_SUCCESS, odla_DestroyComputation(comp));
    }

    SUBCASE("test get type function") 
    {
      odla_computation comp;
      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      model_helper();
      odla_CreateContext(&ctx);
      set_computationItem(comp, 1);

      auto _ov1 = odla_CreateArgument({ODLA_FLOAT16, {.size = 1, .dims = {1}}},
                                      (const odla_value_id)("_ov1"));
      odla_value _ov2;
      odla_value_type _ovt;
      CHECK_EQ(odla_GetValueType(_ov1, &_ovt), ODLA_SUCCESS);
      CHECK_EQ(_ovt.element_type, ODLA_FLOAT32);
      CHECK_EQ(_ovt.shape.size, 1);
      auto GetOdlaType(popart::DataType::FLOAT16);
      // CHECK_EQ(odla_GetValueType(_ov2, &_ovt), ODLA_FAILURE); // todo unvaild value 

      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);

    }


    SUBCASE("test get type function") 
    {
      odla_computation comp;
      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      model_helper();
      odla_CreateContext(&ctx);
      set_computationItem(comp, 1);

      auto _ov1 = odla_CreateArgument({ODLA_INT32, {.size = 1, .dims = {1}}},
                                      (const odla_value_id)("_ov1"));
      odla_value _ov2;
      odla_value_type _ovt;
      CHECK_EQ(odla_GetValueType(_ov1, &_ovt), ODLA_SUCCESS);
      CHECK_EQ(_ovt.element_type, ODLA_FLOAT32);
      CHECK_EQ(_ovt.shape.size, 1);
      auto GetOdlaType(popart::DataType::INT32);
      // CHECK_EQ(odla_GetValueType(_ov2, &_ovt), ODLA_FAILURE); // todo unvaild value 

      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);

    }
    SUBCASE("test get type function") 
    {
      odla_computation comp;
      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      model_helper();
      odla_CreateContext(&ctx);
      set_computationItem(comp, 1);

      auto _ov1 = odla_CreateArgument({ODLA_INT64, {.size = 1, .dims = {1}}},
                                      (const odla_value_id)("_ov1"));
      odla_value _ov2;
      odla_value_type _ovt;
      CHECK_EQ(odla_GetValueType(_ov1, &_ovt), ODLA_SUCCESS);
      CHECK_EQ(_ovt.element_type, ODLA_FLOAT32);
      CHECK_EQ(_ovt.shape.size, 1);
      auto GetOdlaType(popart::DataType::INT64);
     // CHECK_EQ(odla_GetValueType(_ov2, &_ovt), ODLA_FAILURE); // todo unvaild value 

      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);

    }

    SUBCASE("test get type function") 
    {
      odla_computation comp;
      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      model_helper();
      odla_CreateContext(&ctx);
      set_computationItem(comp, 1);

      auto _ov1 = odla_CreateArgument({ODLA_BOOL, {.size = 1, .dims = {1}}},
                                      (const odla_value_id)("_ov1"));
      odla_value _ov2;
      odla_value_type _ovt;
      CHECK_EQ(odla_GetValueType(_ov1, &_ovt), ODLA_SUCCESS);
      CHECK_EQ(_ovt.element_type, ODLA_FLOAT32);
      CHECK_EQ(_ovt.shape.size, 1);
      auto GetOdlaType(popart::DataType::BOOL);
      // CHECK_EQ(odla_GetValueType(_ov2, &_ovt), ODLA_FAILURE); // todo unvaild value 

      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);

    }
    SUBCASE("test get type function") 
    {
      odla_computation comp;
      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      model_helper();
      odla_CreateContext(&ctx);
      set_computationItem(comp, 1);

      auto _ov1 = odla_CreateArgument({ODLA_UINT32, {.size = 1, .dims = {1}}},
                                      (const odla_value_id)("_ov1"));
      odla_value _ov2;
      odla_value_type _ovt;
      CHECK_EQ(odla_GetValueType(_ov1, &_ovt), ODLA_SUCCESS);
      CHECK_EQ(_ovt.element_type, ODLA_FLOAT32);
      CHECK_EQ(_ovt.shape.size, 1);
      auto GetOdlaType(popart::DataType::UINT32);
      // CHECK_EQ(odla_GetValueType(_ov2, &_ovt), ODLA_FAILURE); // todo unvaild value 

      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);

    }

    SUBCASE("test get type function") 
    {
      odla_computation comp;
      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      model_helper();
      odla_CreateContext(&ctx);
      set_computationItem(comp, 1);

      auto _ov1 = odla_CreateArgument({ODLA_UINT64, {.size = 1, .dims = {1}}},
                                      (const odla_value_id)("_ov1"));
      odla_value _ov2;
      odla_value_type _ovt;
      CHECK_EQ(odla_GetValueType(_ov1, &_ovt), ODLA_SUCCESS);
      CHECK_EQ(_ovt.element_type, ODLA_FLOAT32);
      CHECK_EQ(_ovt.shape.size, 1);
      auto GetOdlaType(popart::DataType::UINT64);
      // CHECK_EQ(odla_GetValueType(_ov2, &_ovt), ODLA_FAILURE); // todo unvaild value 

      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);

    }


    SUBCASE("test get type function") 
    {
      odla_computation comp;
      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      model_helper();
      odla_CreateContext(&ctx);
      set_computationItem(comp, 1);

      auto _ov1 = odla_CreateArgument({ODLA_UINT64, {.size = 1, .dims = {1}}},
                                      (const odla_value_id)("_ov1"));
      odla_value _ov2;
      odla_value_type _ovt;
      CHECK_EQ(odla_GetValueType(_ov1, &_ovt), ODLA_SUCCESS);
      CHECK_EQ(_ovt.element_type, ODLA_FLOAT32);
      CHECK_EQ(_ovt.shape.size, 1);
      auto GetOdlaType(popart::DataType::UINT64);
      // CHECK_EQ(odla_GetValueType(_ov2, &_ovt), ODLA_FAILURE); // todo unvaild value 

      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);

    }




}


TEST_CASE("testing base interface") 
  {
    SUBCASE("test pipeline execute") 
    { 
      json _config_json = default_json2();
      _config_json["ipu_num"] = 2;
      _config_json["execution_mode"] = std::string("pipeline");

      PopartConfig::instance()->parse_from_json(_config_json);

      odla_computation comp;
      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      model_helper();
      odla_CreateContext(&ctx);
      set_computationItem(comp, 2);

      float in = 1.f;
      float out = 0.f;
      odla_BindToArgumentById((const odla_value_id) "Input", &in, ctx);
      odla_BindToOutputById((const odla_value_id) "Add", &out, ctx);

      CHECK_EQ(odla_ExecuteComputation(comp, ctx, ODLA_COMPUTE_INFERENCE, nullptr), ODLA_SUCCESS);
      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);
    }

    SUBCASE("test execute function multithread") 
    {
      float in[3] = {1.f, 1.f, 1.f};
      float out[3] = {0.f};

      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      model_helper();
      set_computationItem(comp, 1);

      std::thread t1(execute_multithread, comp, &in[0], &out[0]);
      std::thread t2(execute_multithread, comp, &in[1], &out[1]);
      std::thread t3(execute_multithread, comp, &in[2], &out[2]);

      t1.join();
      t2.join();
      t3.join();

      CHECK_EQ(out[0], 5);
      CHECK_EQ(out[1], 5);
      CHECK_EQ(out[2], 5);

      odla_DestroyComputation(comp);
    }

    SUBCASE("test base function") 
    {
      odla_computation comp;
      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      odla_CreateContext(&ctx);
      set_computationItem(comp, 1);

      int wrong_addr;
      CHECK_EQ(odla_CreateComputation(&comp), ODLA_SUCCESS);
      // CHECK_EQ(odla_DestroyComputation(nullptr), ODLA_FAILURE); //todo 1: nullptr wrong addr protest 
      // CHECK_EQ(odla_DestroyComputation((odla_computation)&wrong_addr), ODLA_FAILURE);

      CHECK_EQ(odla_CreateContext(&ctx), ODLA_SUCCESS);
      // CHECK_EQ(odla_DestroyContext(nullptr), ODLA_FAILURE); //todo 1
      // CHECK_EQ(odla_DestroyContext((odla_context)&wrong_addr), ODLA_FAILURE);

      odla_item_value _test;
      // CHECK_EQ(odla_SetComputationItem(nullptr, ODLA_USE_SIM_MODE, (odla_item_value)&_test), ODLA_FAILURE); // todo: unvaild value, should be recognized failure
      // CHECK_EQ(odla_SetComputationItem((odla_computation)&wrong_addr, ODLA_USE_SIM_MODE, (odla_item_value)&_test), ODLA_FAILURE); //todo 1
      CHECK_EQ(odla_SetComputationItem(comp, ODLA_USE_SIM_MODE, (odla_item_value)&_test), ODLA_SUCCESS);
      CHECK_EQ(odla_SetComputationItem(comp, ODLA_LOAD_ENGINE_MODE, (odla_item_value)&_test), ODLA_UNSUPPORTED_DATATYPE);

      // CHECK_EQ(odla_SetContextItem(nullptr, ODLA_ASYNC_CALLBACK_FUNC, (odla_item_value)&_test), ODLA_INVALID_PARAM); //todo 1
      // CHECK_EQ(odla_SetContextItem((odla_context)&wrong_addr, ODLA_ASYNC_CALLBACK_FUNC, (odla_item_value)&_test), ODLA_INVALID_PARAM);
      CHECK_EQ(odla_SetContextItem(ctx, ODLA_ASYNC_CALLBACK_FUNC, (odla_item_value)&_test), ODLA_SUCCESS);

      CHECK_EQ(odla_DestroyComputation(comp), ODLA_SUCCESS);
      CHECK_EQ(odla_DestroyContext(ctx), ODLA_SUCCESS);

    }

    SUBCASE("test arg function") 
    {    
      
      odla_computation comp;
      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      odla_CreateContext(&ctx);
      set_computationItem(comp, 1);
      
      int wrong_addr;
      int data[5] = {0};
      odla_uint32 _num, _id;
      odla_value _ov;

      auto _input1 = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                          (const odla_value_id)("_input1"));
      auto _input2 = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                          (const odla_value_id)("_input2"));

      auto _constance = odla_CreateConstant({ODLA_FLOAT32, {.size = 1, .dims = {1}}}, &data,
                          (const odla_value_id) "_constance");

      auto _constance1 = odla_CreateConstant({ODLA_FLOAT32, {.size = 8, .dims = {1}}},
                          (int*)0x11, (const odla_value_id) "_constance");

      CHECK_EQ(odla_GetNumOfArgsFromComputation(comp, &_num), ODLA_SUCCESS);
      // CHECK_EQ(odla_GetNumOfArgsFromComputation(nullptr, &_num), ODLA_FAILURE); // todo 1
      // CHECK_EQ(odla_GetNumOfArgsFromComputation((odla_computation)&wrong_addr, &_num), ODLA_FAILURE);
      CHECK_EQ(_num, 2);

      CHECK_EQ(odla_GetArgFromComputationByIdx(comp, 0, &_ov), ODLA_SUCCESS);
      CHECK_EQ(odla_GetArgFromComputationByIdx(comp, 2, &_ov), ODLA_INVALID_PARAM);
      // CHECK_EQ(odla_GetArgFromComputationByIdx(nullptr, 0, &_ov), ODLA_FAILURE); //todo 1
      // CHECK_EQ(odla_GetArgFromComputationByIdx((odla_computation)&wrong_addr, 0, &_ov), ODLA_FAILURE);

      CHECK_EQ(odla_SetValueAsOutput(_input1), ODLA_SUCCESS);
      // CHECK_EQ(odla_SetValueAsOutput(_input1), ODLA_FAILURE); //todo: double set, should be failed
      CHECK_EQ(odla_SetValueAsOutput(_input2), ODLA_SUCCESS);

      odla_values _ovs = {2, {_input1, _input2}};
      // CHECK_EQ(odla_SetValuesAsOutput(_ovs), ODLA_FAILURE); //todo: duplicate set, should be failed
      CHECK_EQ(odla_GetNumOfOutputsFromComputation(comp, &_num), ODLA_SUCCESS);
      // CHECK_EQ(_num, 2); //todo: duplicate set, should be failed

      CHECK_EQ(odla_GetOutputFromComputationByIdx(comp, 0, &_ov), ODLA_SUCCESS);
      CHECK_EQ(odla_GetOutputFromComputationByIdx(comp, 2, &_ov), ODLA_INVALID_PARAM);
      // CHECK_EQ(odla_GetOutputFromComputationByIdx(nullptr, 0, &_ov), ODLA_FAILURE); //todo 1
      // CHECK_EQ(odla_GetOutputFromComputationByIdx((odla_computation)&wrong_addr, 0, &_ov), ODLA_FAILURE);

      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);

    }

    SUBCASE("test bind funtion") 
    {    
      odla_computation comp;
      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      model_helper();
      odla_CreateContext(&ctx);
      set_computationItem(comp, 1);

      int wrong_addr;
      float in = 1.f;
      float out = 1.f;

      // CHECK_EQ(odla_BindToArgumentById((const odla_value_id) "Input", &in, nullptr), ODLA_FAILURE); //todo 1
      // CHECK_EQ(odla_BindToArgumentById((const odla_value_id) "Input", &in, (odla_context)&wrong_addr), ODLA_FAILURE);
      CHECK_EQ(odla_BindToArgumentById((const odla_value_id) "Input", &in, ctx), ODLA_SUCCESS);
      // CHECK_EQ(odla_BindToArgumentById((const odla_value_id) "Input", &in, ctx), ODLA_FAILURE); // todo duplicate bind, should be recognized failure
      // CHECK_EQ(odla_BindToArgumentById((const odla_value_id) "Input", nullptr, ctx), ODLA_FAILURE); //todo 1

      // CHECK_EQ(odla_BindToOutputById((const odla_value_id) "Add", &out, nullptr), ODLA_FAILURE);
      // CHECK_EQ(odla_BindToOutputById((const odla_value_id) "Add", &out, (odla_context)&wrong_addr), ODLA_FAILURE); //todo 1
      CHECK_EQ(odla_BindToOutputById((const odla_value_id) "Add", &out, ctx), ODLA_SUCCESS);
      // CHECK_EQ(odla_BindToOutputById((const odla_value_id) "Add", &out, ctx), ODLA_FAILURE); //todo duplicate bind, should be recognized failure
      // CHECK_EQ(odla_BindToOutputById((const odla_value_id) "Add", nullptr, ctx), ODLA_FAILURE);

      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);
    }
    
    SUBCASE("test bind funtion multithread") 
    {
      odla_computation comp;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      model_helper();
      set_computationItem(comp, 1);

      std::thread threads[5];
      float in[5], out[5];

      for (int i = 0; i < 5; i++) {
        threads[i] = std::thread(test_bind_funciton_multithread, &in[i], &out[i]);
      }
      for (auto& t : threads) {
        t.join();
      }
      CHECK_EQ(ODLA_SUCCESS, odla_DestroyComputation(comp));
    }

    SUBCASE("test get type function") 
    {
      odla_computation comp;
      odla_context ctx;
      CHECK_EQ(ODLA_SUCCESS, odla_CreateComputation(&comp));
      model_helper();
      odla_CreateContext(&ctx);
      set_computationItem(comp, 1);

      auto _ov1 = odla_CreateArgument({ODLA_FLOAT32, {.size = 1, .dims = {1}}},
                                      (const odla_value_id)("_ov1"));
      odla_value _ov2;
      odla_value_type _ovt;
      CHECK_EQ(odla_GetValueType(_ov1, &_ovt), ODLA_SUCCESS);
      CHECK_EQ(_ovt.element_type, ODLA_FLOAT32);
      CHECK_EQ(_ovt.shape.size, 1);

      // CHECK_EQ(odla_GetValueType(_ov2, &_ovt), ODLA_FAILURE); // todo unvaild value 

      odla_DestroyComputation(comp);
      odla_DestroyContext(ctx);

    }

}

