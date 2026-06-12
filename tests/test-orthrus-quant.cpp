#include "../src/llama-ext.h"
#include "ggml-cpp.h"
#include "llama.h"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

static const char * ftype_name(llama_ftype ftype) {
    switch (ftype) {
        case LLAMA_FTYPE_MOSTLY_Q2_K:     return "Q2_K";
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS:  return "IQ3_XXS";
        default:                          return "unknown";
    }
}

static bool check_ftype(quantize_state_impl * qs, const std::vector<ggml_tensor *> & tensors, llama_ftype ftype) {
    std::vector<ggml_type> result_types(tensors.size());
    llama_quant_compute_types(qs, ftype, const_cast<ggml_tensor **>(tensors.data()), result_types.data(), tensors.size());

    std::map<std::string, ggml_type> by_name;
    for (size_t i = 0; i < tensors.size(); i++) {
        by_name[ggml_get_name(tensors[i])] = result_types[i];
    }

    bool ok = true;
    auto expect_eq = [&](const char * lhs, const char * rhs) {
        if (by_name[lhs] != by_name[rhs]) {
            printf("FAIL [%s] %s is %s, expected same as %s (%s)\n",
                    ftype_name(ftype), lhs, ggml_type_name(by_name[lhs]), rhs, ggml_type_name(by_name[rhs]));
            ok = false;
        }
    };
    auto expect_ne = [&](const char * lhs, const char * rhs) {
        if (by_name[lhs] == by_name[rhs]) {
            printf("FAIL [%s] %s unexpectedly matches %s as %s\n",
                    ftype_name(ftype), lhs, rhs, ggml_type_name(by_name[lhs]));
            ok = false;
        }
    };

    expect_eq("blk.0.attn_q_diff.weight",      "blk.0.attn_q.weight");
    expect_eq("blk.0.attn_k_diff.weight",      "blk.0.attn_k.weight");
    expect_eq("blk.0.attn_v_diff.weight",      "blk.0.attn_v.weight");
    expect_eq("blk.0.attn_output_diff.weight", "blk.0.attn_output.weight");

    expect_ne("blk.0.attn_q_diff.weight", "blk.0.attn_kv_b.weight");
    expect_ne("blk.0.attn_q_diff.weight", "blk.0.attn_v_diff.weight");

    return ok;
}

int main() {
    llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);

    llama_quant_model_desc desc = {};
    desc.architecture           = "orthrus";
    desc.n_embd                 = 64;
    desc.n_ff                   = 128;
    desc.n_layer                = 8;
    desc.n_head                 = 8;
    desc.n_head_kv              = 2;
    desc.n_embd_head_k          = 8;
    desc.n_embd_head_v          = 8;

    llama_model * model = llama_quant_model_from_metadata(&desc);
    if (model == nullptr) {
        printf("FAIL could not create mock Orthrus model\n");
        return 1;
    }

    llama_model_quantize_params qparams = llama_model_quantize_default_params();
    quantize_state_impl *       qs      = llama_quant_init(model, &qparams);

    static const char * names[] = {
        "blk.0.attn_q.weight",
        "blk.0.attn_q_diff.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_k_diff.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_v_diff.weight",
        "blk.0.attn_output.weight",
        "blk.0.attn_output_diff.weight",
        "blk.0.attn_kv_b.weight",
    };
    const size_t n_names = sizeof(names) / sizeof(names[0]);

    struct ggml_init_params params = { n_names * ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx(ggml_init(params));

    std::vector<ggml_tensor *> tensors;
    tensors.reserve(n_names);

    for (const char * name : names) {
        ggml_tensor * tensor = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 64, 64);
        ggml_set_name(tensor, name);
        tensors.push_back(tensor);
    }

    bool ok = true;
    ok &= check_ftype(qs, tensors, LLAMA_FTYPE_MOSTLY_Q2_K);
    ok &= check_ftype(qs, tensors, LLAMA_FTYPE_MOSTLY_IQ3_XXS);

    llama_quant_free(qs);
    llama_model_free(model);

    if (!ok) {
        return 1;
    }

    printf("PASS Orthrus diffusion tensor quant category checks\n");
    return 0;
}
