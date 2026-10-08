#include "server-task.h"

#include "build-info.h"
#include "server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "server-common.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#if defined(__linux__)
#include <fcntl.h>
#include <sched.h>
#include <unistd.h>
#endif

//
// task_params
//

json task_params::format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

json task_params::to_json(bool only_metrics) const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto & sampler : sampling.samplers) {
        samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (auto & it : this->lora) {
        lora.push_back({{"id", it.first}, {"scale", it.second}});
    }

    if (only_metrics) {
        return json {
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"adaptive_target",           sampling.adaptive_target},
            {"adaptive_decay",            sampling.adaptive_decay},
            {"max_tokens",                n_predict},
            {"n_predict",                 n_predict}, // TODO: deduplicate?
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"chat_format",               common_chat_format_name(chat_parser_params.format)},
            {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
            {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
            {"generation_prompt",         chat_parser_params.generation_prompt.text},
            {"samplers",                  samplers},
            {"speculative.types",         common_speculative_type_name_str(speculative.types)},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"backend_sampling",          sampling.backend_sampling},
            {"lora",                      lora},
        };
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }

    return json {
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"top_n_sigma",               sampling.top_n_sigma},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typ_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"adaptive_target",           sampling.adaptive_target},
        {"adaptive_decay",            sampling.adaptive_decay},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict},
        {"n_predict",                 n_predict}, // TODO: deduplicate?
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   common_grammar_value(sampling.grammar)},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(chat_parser_params.format)},
        {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
        {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
        {"generation_prompt",         chat_parser_params.generation_prompt.text},
        {"samplers",                  samplers},
        {"speculative.types",         common_speculative_type_name_str(speculative.types)},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"backend_sampling",          sampling.backend_sampling},
        {"lora",                      lora},
    };
}

//
// task_result_state
//
task_result_state::task_result_state(const common_chat_parser_params & chat_parser_params)
    : chat_parser_params(chat_parser_params)
    , oai_resp_id("resp_" + random_string())
    , oai_resp_reasoning_id("rs_" + random_string())
    , oai_resp_message_id("msg_" + random_string()) {
    if (chat_parser_params.is_continuation && !chat_parser_params.echo) {
        // initialize chat_msg to avoid emitting a delta containing the assistant prefill
        chat_msg = common_chat_parse(generated_input, true, chat_parser_params);
    }
}

common_chat_msg task_result_state::update_chat_msg(
        const common_chat_input & added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls) {
    generated_input.append(added);
    auto msg_prv_copy = chat_msg;
    //SRV_DBG("Parsing chat message: %s\n", generated_input.text.c_str());
    auto new_msg = common_chat_parse(
        generated_input,
        is_partial,
        chat_parser_params);
    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id   = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
    return json {
        {"total",     total},
        {"cache",     cache},
        {"processed", processed},
        {"time_ms",   time_ms},
    };
}

static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

//
// completion_token_output
//

json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto & p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json {
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
        });
    }
    return probs_for_token;
}

json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto & p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        out.push_back(json {
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
        });
    }
    return out;
}

float completion_token_output::logarithm(float x) {
    // the JSON library converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string & str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
        bytes.push_back(c);
    }
    return bytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_final::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             content.text},
        {"tokens",              tokens},
        {"id_slot",             id_slot},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             stats.to_json()},
    };
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json {
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}

json server_task_result_cmpl_final::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content.text},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content.text;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", index},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto & diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", index},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion.chunk"},
    });

    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
        });
    }

    if (stats.is_set()) {
        deltas.back()["timings"] = stats.to_json();
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content.text;
    }

    std::vector<json> output;

    if (msg.reasoning_content != "") {
        output.push_back(json {
            {"id",      "rs_" + random_string()},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (msg.content != "") {
        output.push_back(json {
            {"content", json::array({ json {
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;

    if (oaicompat_msg.reasoning_content != "") {
        const json output_item = json {
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
        server_sent_events.push_back(json {
            {"event", "response.output_text.done"},
            {"data", json {
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"text",    oaicompat_msg.content}
            }}
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content}
        };

        server_sent_events.push_back(json {
            {"event", "response.content_part.done"},
            {"data", json {
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"part",    content_part}
            }}
        });
        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const json output_item = {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name}
        };
        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    server_sent_events.push_back(json {
        {"event", "response.completed"},
        {"data", json {
            {"type", "response.completed"},
            {"response", json {
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }}
    });

    if (stats.is_set()) {
        server_sent_events.back().at("data")["timings"] = stats.to_json();
    }

    return server_sent_events;
}

json server_task_result_cmpl_final::to_json_oaicompat_asr() {
    json event = json {
        {"type",  "transcript.text.done"},
        {"text",  oaicompat_msg.content},
        {"usage", json {
            {"type",         "tokens"},
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };
    return event;
}

json server_task_result_cmpl_final::to_json_anthropic() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content.text;
    }

    // thinking block comes first (Anthropic extended thinking format)
    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}  // empty signature for local models (no cryptographic verification)
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
        });
    }

    for (const auto & tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        } catch (const std::exception &) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_thinking = !oaicompat_msg.reasoning_content.empty();
    bool has_text     = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    size_t text_block_index     = has_thinking ? 1 : 0;

    bool thinking_block_started = false;
    bool text_block_started     = false;
    std::unordered_set<size_t> tool_calls_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto & full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    // close content blocks in order
    if (has_thinking) {
        // Anthropic API requires a signature_delta before closing thinking blocks
        // We use an empty signature since we can't generate a cryptographic signature for local models
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
        });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
        });
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
        });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
        });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
    });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
    });

    return events;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    if (is_begin) {
        return; // begin marker only flushes headers, skip parsing
    }
    state.update_chat_msg(content, true, oaicompat_msg_diffs);

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_created       = state.oai_resp_created;
    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    if (res_type == TASK_RESPONSE_TYPE_OAI_RESP && !state.oai_resp_created && (is_progress || n_decoded == 1)) {
        state.oai_resp_created = true;
    }

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id = diff.tool_call_delta.id;
        }
    }
}

json server_task_result_cmpl_partial::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    if (is_begin) {
        return nullptr; // simply signal to HTTP handler to send the headers and status code
    }
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_partial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content.text},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content.text},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json & delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    };
    // We have to send an initial update to conform to openai behavior
    if (first || is_progress) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
        });
    }

    for (const auto & diff : oaicompat_msg_diffs) {
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        auto & last_json = deltas[deltas.size() - 1];
        GGML_ASSERT(last_json.at("choices").size() >= 1);

        if (prob_output.probs.size() > 0) {
            last_json.at("choices").at(0)["logprobs"] = json {
                {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
            };
        }

        if (stats.is_set()) {
            last_json["timings"] = stats.to_json();
        }
        if (is_progress) {
            last_json["prompt_progress"] = progress.to_json();
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (!oai_resp_created) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type", "response.created"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    } else if (is_progress) {
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                thinking_block_started = true;
            }
            events.push_back(json {
                {"event", "response.reasoning_text.delta"},
                {"data", json {
                    {"type",    "response.reasoning_text.delta"},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"part", json {
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                text_block_started = true;
            }
            events.push_back(json {
                {"event", "response.output_text.delta"},
                {"data", json {
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type",  "response.output_item.added"},
                    {"item", json {
                        {"id",        "fc_" + diff.tool_call_delta.id},
                        {"arguments", ""},
                        {"call_id",   "call_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                }},
            });
        }
    }

    if (!events.empty()) {
        json & data = events.back().at("data");
        if (stats.is_set()) {
            data["timings"] = stats.to_json();
        }
        if (is_progress) {
            data["prompt_progress"] = progress.to_json();
        }
    }

    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    json event = json {
        {"type", "transcript.text.delta"},
        {"delta", content.text},
    };
    return event;
}

json server_task_result_cmpl_partial::to_json_anthropic() {
    json events = json::array();
    bool first = (n_decoded == 1);
    // use member variables to track block state across streaming calls
    // (anthropic_thinking_block_started, anthropic_text_block_started)

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
        });
    }

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
    size_t text_block_index     = anthropic_has_reasoning ? 1 : 0;

    // use local copies of streaming state (copied from task_result_state in update())
    // these reflect the state BEFORE this chunk was processed
    bool thinking_started = thinking_block_started;
    bool text_started     = text_block_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            // use anthropic_has_reasoning for thinking block count (persists across calls)
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    return events;
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
    return res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? to_json_oaicompat()
        : to_json_non_oaicompat();
}

json server_task_result_embd::to_json_non_oaicompat() {
    return json {
        {"index",     index},
        {"embedding", embedding},
    };
}

json server_task_result_embd::to_json_oaicompat() {
    return json {
        {"index",            index},
        {"embedding",        embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
    return json {
        {"index",            index},
        {"score",            score},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_decision
//
json server_task_result_decision::to_json() {
    return json {
        {"index",            index},
        {"scores",           scores},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}

//
// server_task_result_metrics
//
json server_task_result_slots::to_json() {
    return slots_data;
}

json server_task_result_metrics::to_json() {
    // not used, /metrics renders prometheus text via to_metrics()
    return json{};
}

// metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
std::string server_task_result_metrics::to_metrics() {
    const std::vector<metric_item> counters = {
        {
            "prompt_tokens_total",
            "Number of prompt tokens processed, excluding cached tokens",
            (double) metrics.prompt.count
        }, {
            "prompt_tokens_cached_total",
            "Number of prompt tokens reused from the cache",
            (double) metrics.n_prompt_cached
        }, {
            "prompt_seconds_total",
            "Total time spent processing prompts",
            metrics.prompt.time / 1.e6
        }, {
            "tokens_predicted_total",
            "Number of generation tokens processed",
            (double) metrics.predict.count
        }, {
            "tokens_predicted_seconds_total",
            "Total time spent generating tokens",
            metrics.predict.time / 1.e6
        }, {
            "n_decode_total",
            "Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding",
            (double) metrics.n_decode
        }, {
            "n_tokens_max",
            "Largest observed sequence length (prompt + generation)",
            (double) metrics.n_tokens_max
        }, {
            "spec_decode_num_draft_tokens_total",
            "Speculative: Total draft tokens generated",
            (double) metrics.n_draft_tokens
        }, {
            "spec_decode_num_accepted_tokens_total",
            "Speculative: Total draft tokens accepted by the target model",
            (double) metrics.n_draft_accepted
        }, {
            "spec_decode_num_drafts_total",
            "Speculative: Total speculative decoding verification steps",
            (double) metrics.n_draft_verif_steps
        },
    };

    const std::vector<metric_item> gauges = {
        {
            "prompt_tokens_seconds",
            "Average prompt throughput in tokens/s",
            metrics.prompt_bucket.n_per_second()
        }, {
            "predicted_tokens_seconds",
            "Average generation throughput in tokens/s",
            metrics.predict_bucket.n_per_second()
        }, {
            "requests_processing",
            "Number of requests processing",
            (double) n_processing_slots
        }, {
            "requests_deferred",
            "Number of requests deferred",
            (double) n_tasks_deferred
        }, {
            "n_busy_slots_per_decode",
            "Average number of busy slots per llama_decode() call",
            (double) metrics.n_busy_slots / std::max((double) metrics.n_decode, 1.0)
        }, {
            "prompt_cache_bytes",
            "Host RAM held by the prompt cache: serialized KV state plus context checkpoints",
            (double) prompt_cache_bytes
        }, {
            "prompt_cache_tokens",
            "Number of tokens whose state the prompt cache is holding",
            (double) prompt_cache_tokens
        }, {
            "prompt_cache_limit_bytes",
            "Prompt cache size limit from --cache-ram; 0 when unlimited or disabled",
            (double) prompt_cache_limit_bytes
        },
    };

    std::stringstream prometheus;

    auto add_items = [&prometheus](const char * type, const std::vector<metric_item> & items) {
        for (const auto & item : items) {
            prometheus << "# HELP llamacpp:" << item.name << " " << item.description << "\n"
                       << "# TYPE llamacpp:" << item.name << " " << type             << "\n"
                       << "llamacpp:"        << item.name << " " << item.value       << "\n";
        }
    };

    add_items("counter", counters);
    add_items("gauge",   gauges);

    // the MoE expert cache lives in libllama, not in the slot metrics, and its
    // own logging is demoted to TRACE and filtered at the default verbosity
    llama_moe_cache_stats moe_cache;
    if (llama_moe_cache_get_stats(&moe_cache)) {
        add_items("counter", {
            { "moe_cache_hits_total",     "MoE expert cache: routed expert ids found resident",      (double) moe_cache.n_hit    },
            { "moe_cache_misses_total",   "MoE expert cache: routed expert ids not resident",        (double) moe_cache.n_miss   },
            { "moe_cache_inserts_total",  "MoE expert cache: expert uploads scheduled",              (double) moe_cache.n_insert },
            { "moe_cache_evictions_total","MoE expert cache: resident experts displaced",            (double) moe_cache.n_evict  },
        });
        add_items("gauge", {
            { "moe_cache_layers",         "MoE expert cache: cached layers",                         (double) moe_cache.n_layers },
            { "moe_cache_slots_per_layer","MoE expert cache: slots per cached layer",                (double) moe_cache.n_slots  },
        });
        if (moe_cache.n_host_slots > 0) {
            add_items("counter", {
                { "moe_cache_host_hits_total",         "MoE host tier: routed experts found in the host pool",      (double) moe_cache.n_host_hit          },
                { "moe_cache_host_misses_total",       "MoE host tier: routed experts copied into the host pool",   (double) moe_cache.n_host_miss         },
                { "moe_cache_host_read_bytes_total",   "MoE host tier: bytes copied into the host pool",            (double) moe_cache.host_bytes_read     },
                { "moe_cache_host_read_seconds_total", "MoE host tier: time spent copying into the host pool",      moe_cache.host_read_us / 1.e6          },
            });
            add_items("gauge", {
                { "moe_cache_host_slots_per_layer",    "MoE host tier: pool slots per cached layer",                (double) moe_cache.n_host_slots        },
            });
        }
    }

    // labeled counter: one time series per draft position
    if (!metrics.n_accepted_per_pos.empty()) {
        prometheus << "# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
                      " Accepted tokens per draft position\n"
                   << "# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter\n";
        for (size_t i = 0; i < metrics.n_accepted_per_pos.size(); i++) {
            prometheus << "llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\""
                       << i << "\"} " << metrics.n_accepted_per_pos[i] << "\n";
        }
    }

    return prometheus.str();
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
    if (is_save) {
        return json {
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   n_tokens },
            { "n_written", n_bytes },
            { "timings", {
                { "save_ms", t_ms }
            }},
        };
    }

    return json {
        { "id_slot",    id_slot },
        { "filename",   filename },
        { "n_restored", n_tokens },
        { "n_read",     n_bytes },
        { "timings", {
            { "restore_ms", t_ms }
        }},
    };
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
    return json {
        { "id_slot",  id_slot },
        { "n_erased", n_erased },
    };
}

//
// server_task_result_get_lora
//

json server_task_result_get_lora::to_json() {
    json result = json::array();
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        json entry = {
            {"id",            i},
            {"path",          lora.info.path},
            {"scale",         lora.info.scale},
            {"task_name",     lora.info.task_name},
            {"prompt_prefix", lora.info.prompt_prefix},
        };
        if (!lora.alora_invocation_tokens.empty()) {
            entry["alora_invocation_string"] = lora.alora_invocation_string;
            entry["alora_invocation_tokens"] = lora.alora_invocation_tokens;
        }
        result.push_back(std::move(entry));
    }
    return result;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() {
    return json {{ "success", true }};
}

//
// server_prompt_cache_disk
//

namespace {

constexpr uint32_t PCACHE_DISK_MAGIC   = 0x4443504c; // "LPCD"
constexpr uint32_t PCACHE_DISK_VERSION = 3;

constexpr size_t PCACHE_DISK_CKPT_HEADER_SIZE = 8 + 3*4 + 3*8;

// evicted entries wait in RAM for the writer, this bounds how much
constexpr size_t PCACHE_DISK_PENDING_MAX = 4ull*1024*1024*1024;

constexpr uint64_t PCACHE_HASH_SEED  = 0xcbf29ce484222325ULL;
constexpr uint64_t PCACHE_HASH_PRIME = 0x100000001b3ULL;

// word-wise so that it runs at memory speed on multi-GiB blobs
uint64_t pcache_hash(uint64_t h, const void * data, size_t size) {
    const uint8_t * p = (const uint8_t *) data;

    for (; size >= sizeof(uint64_t); size -= sizeof(uint64_t), p += sizeof(uint64_t)) {
        uint64_t word;
        memcpy(&word, p, sizeof(word));
        h = (h ^ word) * PCACHE_HASH_PRIME;
    }

    for (; size > 0; --size, ++p) {
        h = (h ^ *p) * PCACHE_HASH_PRIME;
    }

    return h;
}

struct pcache_writer {
    explicit pcache_writer(FILE * file) : file(file) {}

    FILE *      file;
    uint64_t    hash    = PCACHE_HASH_SEED;
    size_t      n_bytes = 0;
    std::string err;

    void put(const void * data, size_t size) {
        hash     = pcache_hash(hash, data, size);
        n_bytes += size;

        if (err.empty() && size > 0 && fwrite(data, 1, size, file) != size) {
            err = strerror(errno);
        }
    }

    template <typename T>
    void put(const T & value) {
        static_assert(std::is_arithmetic<T>::value, "put() is for scalar fields");
        put(&value, sizeof(value));
    }

    void put(const std::vector<uint8_t> & blob) {
        put(blob.data(), blob.size());
    }
};

struct pcache_reader {
    FILE *   file;
    uint64_t remaining;
    uint64_t hash = PCACHE_HASH_SEED;
    bool     ok   = true;

    void get(void * data, size_t size) {
        if (!ok || size > remaining || (size > 0 && fread(data, 1, size, file) != size)) {
            ok = false;
            return;
        }

        remaining -= size;
        hash       = pcache_hash(hash, data, size);
    }

    template <typename T>
    T get() {
        static_assert(std::is_arithmetic<T>::value, "get() is for scalar fields");
        T value{};
        get(&value, sizeof(value));
        return value;
    }

    bool fits(uint64_t a, uint64_t b = 0, uint64_t c = 0) const {
        return ok && a <= remaining && b <= remaining - a && c <= remaining - a - b;
    }

    void get(std::vector<uint8_t> & blob, uint64_t size) {
        if (!fits(size)) {
            ok = false;
            return;
        }

        blob.resize(size);
        get(blob.data(), size);
    }
};

// the size and mtime of a file, a missing file hashes as such
uint64_t pcache_hash_file(uint64_t h, const std::string & path) {
    std::error_code ec;
    const uint64_t size  = std::filesystem::file_size(path, ec);
    const int64_t  mtime = static_cast<int64_t>(std::filesystem::last_write_time(path, ec).time_since_epoch().count());

    h = pcache_hash(h, path.data(), path.size());
    h = pcache_hash(h, &size, sizeof(size));

    return pcache_hash(h, &mtime, sizeof(mtime));
}

static_assert(sizeof(llama_token) == sizeof(int32_t), "the file stores the keys as int32");

// the media chunk that starts at idx, nullptr if none does
const mtmd_input_chunk * pcache_chunk_at(const server_tokens & tokens, size_t idx) {
    try {
        return tokens.find_chunk(idx).get();
    } catch (const std::exception &) {
        return nullptr;
    }
}

// a key in [INT32_MIN + 1, -2]: neither a token id nor LLAMA_TOKEN_NULL
llama_token pcache_media_key(uint64_t chunk_hash, uint64_t idx_in_chunk) {
    const uint64_t h = pcache_hash(chunk_hash, &idx_in_chunk, sizeof(idx_in_chunk));

    return -2 - (llama_token) ((uint32_t) (h ^ (h >> 32)) % 0x7ffffffeu);
}

// the file and the index hold keys: a text position keeps its token id, a media position gets a negative key
// LLAMA_TOKEN_NULL marks a position without key (media chunk without id)
llama_tokens pcache_keys(const server_tokens & tokens) {
    llama_tokens keys(tokens.size(), LLAMA_TOKEN_NULL);

    size_t n_run = 1;
    for (size_t i = 0; i < keys.size(); i += n_run) {
        n_run = 1;

        if (tokens[i] >= 0) {
            keys[i] = tokens[i];

            continue;
        }

        const mtmd_input_chunk * chunk = pcache_chunk_at(tokens, i);
        if (chunk == nullptr) {
            continue;
        }

        const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
        if (n_tokens == 0 || n_tokens > keys.size() - i) {
            continue;
        }

        n_run = n_tokens;

        const char * id = mtmd_input_chunk_get_id(chunk);
        if (id == nullptr || id[0] == '\0') {
            continue;
        }

        const uint64_t shape[3] = { strlen(id), n_tokens, (uint64_t) mtmd_input_chunk_get_n_pos(chunk) };

        uint64_t chunk_hash = pcache_hash(PCACHE_HASH_SEED, shape, sizeof(shape));
        chunk_hash = pcache_hash(chunk_hash, id, shape[0]);

        for (size_t j = 0; j < n_tokens; ++j) {
            keys[i + j] = pcache_media_key(chunk_hash, j);
        }
    }

    return keys;
}

bool pcache_has_unkeyable(const llama_tokens & keys) {
    return std::find(keys.begin(), keys.end(), LLAMA_TOKEN_NULL) != keys.end();
}

bool pcache_is_media_key(llama_token key) {
    return key < 0;
}

size_t pcache_key_prefix(const llama_tokens & a, const llama_tokens & b) {
    return std::mismatch(a.begin(), a.end(), b.begin(), b.end()).first - a.begin();
}

// the longer entry serves every request of the shorter one only if it has no media after the shorter prefix
bool pcache_covers(const llama_tokens & longer, const llama_tokens & shorter) {
    return pcache_key_prefix(longer, shorter) == shorter.size() && std::none_of(longer.begin() + shorter.size(), longer.end(), pcache_is_media_key);
}

// a prefix that ends inside a media chunk of the request moves back to the start of the chunk
size_t pcache_round_down_to_chunk_start(const server_tokens & tokens, size_t n) {
    if (n == 0 || n >= tokens.size() || tokens[n - 1] != LLAMA_TOKEN_NULL || tokens[n] != LLAMA_TOKEN_NULL) {
        return n;
    }

    while (n > 0 && pcache_chunk_at(tokens, n) == nullptr) {
        n--;
    }

    return n;
}

// the prefix of the request that an entry restores, 0 = no match
// 0 also if media is left in the entry after the prefix: the restored tokens take the chunks from the request
size_t pcache_usable_prefix(const llama_tokens & entry_keys, const llama_tokens & keys_new, const server_tokens & tokens_new) {
    const size_t n_match = pcache_round_down_to_chunk_start(tokens_new, pcache_key_prefix(entry_keys, keys_new));

    return std::any_of(entry_keys.begin() + n_match, entry_keys.end(), pcache_is_media_key) ? 0 : n_match;
}

// the matched prefix comes from the request, the rest is the text of the entry
// the chunks are placeholders like in a slot prompt: their KV is restored, so their data is not needed
server_tokens pcache_restored_tokens(const llama_tokens & entry_keys, size_t n_match, const server_tokens & tokens_new) {
    if (std::none_of(entry_keys.begin(), entry_keys.end(), pcache_is_media_key)) {
        return server_tokens(entry_keys, tokens_new.has_mtmd);
    }

    server_tokens res;
    res.has_mtmd = tokens_new.has_mtmd;

    for (size_t i = 0; i < n_match;) {
        if (tokens_new[i] != LLAMA_TOKEN_NULL) {
            res.push_back(tokens_new[i++]);

            continue;
        }

        const mtmd_input_chunk * chunk = pcache_chunk_at(tokens_new, i);
        const size_t n_chunk = chunk ? mtmd_input_chunk_get_n_tokens(chunk) : 0;
        if (n_chunk == 0 || n_chunk > n_match - i) {
            throw std::runtime_error("prefix does not end at a media chunk boundary");
        }

        res.push_back_placeholder(chunk);
        i += n_chunk;
    }

    res.insert(llama_tokens(entry_keys.begin() + n_match, entry_keys.end()));

    return res;
}

std::string pcache_disk_path(const std::string & dir, const std::string & prefix, uint64_t id) {
    return (std::filesystem::path(dir) / (prefix + std::to_string(id) + ".pcache")).string();
}

// the digits keep the files of a model with a longer stem out
bool pcache_disk_is_own_file(const std::string & name, const std::string & prefix) {
    if (name.compare(0, prefix.size(), prefix) != 0) {
        return false;
    }

    const size_t pos_digits_end = name.find_first_not_of("0123456789", prefix.size());
    if (pos_digits_end == prefix.size() || pos_digits_end == std::string::npos) {
        return false;
    }

    const std::string suffix = name.substr(pos_digits_end);

    return suffix == ".pcache" || suffix == ".pcache.tmp";
}

// returns the reason on failure, empty on success
std::string pcache_disk_write(const std::string & path, uint64_t id, uint64_t fingerprint, const llama_tokens & keys, const server_prompt_cache_state & state, size_t & n_bytes) {
    std::unique_ptr<FILE, decltype(&fclose)> file(fopen(path.c_str(), "wb"), fclose);
    if (!file) {
        return strerror(errno);
    }

    pcache_writer w(file.get());

    w.put(PCACHE_DISK_MAGIC);
    w.put(PCACHE_DISK_VERSION);
    w.put(fingerprint);
    w.put(id);
    w.put((uint64_t) keys.size());
    w.put((uint64_t) state.data.main.size());
    w.put((uint64_t) state.data.drft.size());
    w.put((uint32_t) state.prompt.checkpoints.size());

    w.put(keys.data(), keys.size() * sizeof(llama_token));

    for (const auto & ckpt : state.prompt.checkpoints) {
        w.put(ckpt.n_tokens);
        w.put((int32_t) ckpt.id_task);
        w.put((int32_t) ckpt.pos_min);
        w.put((int32_t) ckpt.pos_max);
        w.put((uint64_t) ckpt.data_tgt.size());
        w.put((uint64_t) ckpt.data_dft.size());
        w.put((uint64_t) ckpt.data_spec.size());
        w.put(ckpt.data_tgt);
        w.put(ckpt.data_dft);
        w.put(ckpt.data_spec);
    }

    w.put(state.data.main);
    w.put(state.data.drft);

    const uint64_t checksum = w.hash;
    if (w.err.empty() && fwrite(&checksum, 1, sizeof(checksum), file.get()) != sizeof(checksum)) {
        w.err = strerror(errno);
    }
    w.n_bytes += sizeof(checksum);

    if (w.err.empty() && fflush(file.get()) != 0) {
        w.err = strerror(errno);
    }

#if defined(__linux__)
    if (w.err.empty()) {
        const int fd = fileno(file.get());

        if (fdatasync(fd) != 0) {
            w.err = strerror(errno);
        }

        // do not let a multi-GiB file sit in the page cache
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    }
#endif

    if (fclose(file.release()) != 0 && w.err.empty()) {
        w.err = strerror(errno);
    }

    n_bytes = w.n_bytes;

    return w.err;
}

// writes to a temporary file and renames it, so that a reader never sees a partial file
std::string pcache_disk_store(const std::string & path, uint64_t id, uint64_t fingerprint, const llama_tokens & keys, const server_prompt_cache_state & state, size_t & n_bytes) {
    const std::string path_tmp = path + ".tmp";

    std::string err;
    try {
        err = pcache_disk_write(path_tmp, id, fingerprint, keys, state, n_bytes);
    } catch (const std::exception & e) {
        err = e.what();
    }

    std::error_code ec;
    if (err.empty()) {
        std::filesystem::rename(path_tmp, path, ec);
        if (ec) {
            err = ec.message();
        }
    }

    if (!err.empty()) {
        std::filesystem::remove(path_tmp, ec);
    }

    return err;
}

// reads the header and the keys of a file left by a previous run, the payload stays unread
std::string pcache_disk_peek(const std::string & path, uint64_t fingerprint, uint64_t & id, llama_tokens & keys, size_t & n_bytes) {
    std::error_code ec;
    const uint64_t file_size = std::filesystem::file_size(path, ec);
    if (ec) {
        return ec.message();
    }

    std::unique_ptr<FILE, decltype(&fclose)> file(fopen(path.c_str(), "rb"), fclose);
    if (!file) {
        return strerror(errno);
    }

    try {
        pcache_reader r { file.get(), file_size };

        const uint32_t magic            = r.get<uint32_t>();
        const uint32_t version          = r.get<uint32_t>();
        const uint64_t file_fingerprint = r.get<uint64_t>();
        const uint64_t file_id          = r.get<uint64_t>();
        const uint64_t n_tokens         = r.get<uint64_t>();
        const uint64_t size_main        = r.get<uint64_t>();
        const uint64_t size_drft        = r.get<uint64_t>();
        const uint32_t n_checkpoints    = r.get<uint32_t>();

        if (!r.ok) {
            return "truncated header";
        }

        if (magic != PCACHE_DISK_MAGIC || version != PCACHE_DISK_VERSION) {
            return "bad magic or version";
        }

        if (file_fingerprint != fingerprint) {
            return "another model file or server build";
        }

        // lower bound only, the checkpoints are not walked
        const uint64_t size_tokens = n_tokens * sizeof(llama_token);
        const uint64_t size_rest   = n_checkpoints * PCACHE_DISK_CKPT_HEADER_SIZE + sizeof(uint64_t);

        if (n_tokens == 0 || n_tokens > r.remaining / sizeof(llama_token) || !r.fits(size_tokens + size_rest, size_main, size_drft)) {
            return "sizes in the header exceed the file size";
        }

        keys.resize(n_tokens);
        r.get(keys.data(), size_tokens);

        if (!r.ok) {
            return "truncated file";
        }

        if (pcache_has_unkeyable(keys)) {
            return "has media without key";
        }

        id      = file_id;
        n_bytes = file_size;
    } catch (const std::exception & e) {
        return e.what();
    }

    return "";
}

std::string pcache_disk_read(const std::string & path, uint64_t id, uint64_t fingerprint, const llama_tokens & keys, server_prompt_cache_state & out, size_t & n_bytes) {
    std::error_code ec;
    const uint64_t file_size = std::filesystem::file_size(path, ec);
    if (ec) {
        return ec.message();
    }

    std::unique_ptr<FILE, decltype(&fclose)> file(fopen(path.c_str(), "rb"), fclose);
    if (!file) {
        return strerror(errno);
    }

    try {
        pcache_reader r { file.get(), file_size };

        const uint32_t magic            = r.get<uint32_t>();
        const uint32_t version          = r.get<uint32_t>();
        const uint64_t file_fingerprint = r.get<uint64_t>();
        const uint64_t file_id          = r.get<uint64_t>();
        const uint64_t file_n_tokens    = r.get<uint64_t>();
        const uint64_t size_main        = r.get<uint64_t>();
        const uint64_t size_drft        = r.get<uint64_t>();
        const uint32_t n_checkpoints    = r.get<uint32_t>();

        if (!r.ok) {
            return "truncated header";
        }

        if (magic != PCACHE_DISK_MAGIC || version != PCACHE_DISK_VERSION || file_fingerprint != fingerprint) {
            return "bad magic, version or fingerprint";
        }

        if (file_id != id || file_n_tokens != keys.size()) {
            return "header does not match the index";
        }

        llama_tokens file_keys(keys.size());
        r.get(file_keys.data(), file_keys.size() * sizeof(llama_token));

        if (!r.ok) {
            return "truncated keys";
        }

        if (file_keys != keys) {
            return "keys do not match the index";
        }

        if (!r.fits(size_main, size_drft) || n_checkpoints > r.remaining / PCACHE_DISK_CKPT_HEADER_SIZE) {
            return "sizes in the header exceed the file size";
        }

        for (uint32_t i = 0; i < n_checkpoints; ++i) {
            common_prompt_checkpoint ckpt;

            ckpt.n_tokens = r.get<int64_t>();
            ckpt.id_task  = r.get<int32_t>();
            ckpt.pos_min  = r.get<int32_t>();
            ckpt.pos_max  = r.get<int32_t>();

            const uint64_t size_tgt  = r.get<uint64_t>();
            const uint64_t size_dft  = r.get<uint64_t>();
            const uint64_t size_spec = r.get<uint64_t>();

            if (!r.fits(size_tgt, size_dft, size_spec)) {
                return "checkpoint sizes exceed the file size";
            }

            r.get(ckpt.data_tgt,  size_tgt);
            r.get(ckpt.data_dft,  size_dft);
            r.get(ckpt.data_spec, size_spec);

            out.prompt.checkpoints.push_back(std::move(ckpt));
        }

        r.get(out.data.main, size_main);
        r.get(out.data.drft, size_drft);

        const uint64_t hash = r.hash;

        uint64_t checksum = 0;
        if (!r.ok || fread(&checksum, 1, sizeof(checksum), file.get()) != sizeof(checksum)) {
            return "truncated file";
        }

        if (checksum != hash) {
            return "checksum mismatch";
        }
    } catch (const std::exception & e) {
        return e.what();
    }

    n_bytes = file_size;

    return "";
}

} // namespace

uint64_t server_prompt_cache_disk::make_fingerprint(const std::string & model_path, const std::string & mmproj_path) {
    uint64_t h = pcache_hash_file(PCACHE_HASH_SEED, model_path);

    // the KV of an image depends on the projector
    if (!mmproj_path.empty()) {
        h = pcache_hash_file(h, mmproj_path);
    }

    const std::string commit = llama_commit();
    h = pcache_hash(h, commit.data(), commit.size());

#if defined(__linux__)
    // the commit is "unknown" in some container builds, the binary tells the builds apart
    h = pcache_hash_file(h, "/proc/self/exe");
#endif

    return h;
}

server_prompt_cache_disk::server_prompt_cache_disk(const std::string & dir, const std::string & prefix, size_t limit_size, uint64_t fingerprint)
    : dir(dir), prefix(prefix), limit_size(limit_size), fingerprint(fingerprint) {
    size_t n_removed = 0;

    std::error_code ec;
    std::filesystem::directory_iterator it(dir, ec);
    for (; !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
        const std::filesystem::path path = it->path();
        if (!pcache_disk_is_own_file(path.filename().string(), prefix)) {
            continue;
        }

        uint64_t     id = 0;
        llama_tokens keys;
        size_t       n_bytes = 0;

        std::string err = path.extension() == ".pcache" ? pcache_disk_peek(path.string(), fingerprint, id, keys, n_bytes) : "unfinished write";

        if (err.empty() && path.filename().string() != prefix + std::to_string(id) + ".pcache") {
            err = "file name does not match the header";
        }

        if (err.empty()) {
            index.push_back({ std::move(keys), path.string(), n_bytes, false, id });

            continue;
        }

        SRV_DBG("prompt cache disk: removing %s: %s\n", path.string().c_str(), err.c_str());

        std::error_code ec_remove;
        if (std::filesystem::remove(path, ec_remove)) {
            n_removed++;
        }
    }
    if (ec) {
        throw std::runtime_error("cannot list " + dir + ": " + ec.message());
    }

    index.sort([](const entry & a, const entry & b) { return a.id < b.id; });

    // the worker has not started yet
    next_id = index.empty() ? 0 : index.back().id + 1;

    enforce_limit();

    SRV_INF("prompt cache disk: dir %s, limit %.0f MiB, indexed %zu entries, %.1f MiB from a previous run, removed %zu stale files\n",
            dir.c_str(), limit_size / (1024.0 * 1024.0), index.size(), indexed_size() / (1024.0 * 1024.0), n_removed);

    worker = std::thread([this] { worker_loop(); });
}

server_prompt_cache_disk::~server_prompt_cache_disk() {
    stop_worker();
}

void server_prompt_cache_disk::stop_worker() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        stop = true;
    }
    cv.notify_all();

    if (worker.joinable()) {
        worker.join();
    }

    // the write that was running at stop lands in done
    receive();
}

void server_prompt_cache_disk::worker_loop() {
#if defined(__linux__)
    // the main thread is pinned to one core and this thread inherits that
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    for (unsigned cpu = 0; cpu < std::min<unsigned>(std::thread::hardware_concurrency(), CPU_SETSIZE); ++cpu) {
        CPU_SET(cpu, &cpus);
    }
    sched_setaffinity(0, sizeof(cpus), &cpus);
#endif

    while (true) {
        pending_state item;

        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this] { return stop || !pending.empty(); });

            if (stop) {
                return;
            }

            item = std::move(pending.front());
            pending.pop_front();
            pending_bytes -= item.state.size();
        }

        const uint64_t id       = next_id++;
        const size_t   n_tokens = item.keys.size();
        const std::string path  = pcache_disk_path(dir, prefix, id);

        const int64_t t_start = ggml_time_us();

        size_t n_bytes = 0;
        const std::string err = pcache_disk_store(path, id, fingerprint, item.keys, item.state, n_bytes);

        if (!err.empty()) {
            SRV_WRN("prompt cache disk: failed to write %zu-token entry: %s\n", n_tokens, err.c_str());

            continue;
        }

        const double t_ms = std::max((ggml_time_us() - t_start) / 1000.0, 1e-3);

        SRV_INF("prompt cache disk: wrote %zu-token entry, %.1f MiB in %.1f ms (%.0f MB/s)\n",
                n_tokens, n_bytes / (1024.0 * 1024.0), t_ms, n_bytes / 1e3 / t_ms);

        std::lock_guard<std::mutex> lock(mtx);
        done.push_back({ std::move(item.keys), path, n_bytes, false, id });
    }
}

void server_prompt_cache_disk::push(server_prompt_cache_state && state) {
    collect();

    // the keys are made here, the writer never sees the media chunks
    pending_state item;
    item.keys = pcache_keys(state.prompt.tokens);

    if (pcache_has_unkeyable(item.keys)) {
        SRV_INF("prompt cache disk: not keeping evicted %zu-token entry, it has media without id\n", item.keys.size());

        return;
    }

    state.prompt.tokens.clear();
    item.state = std::move(state);

    const size_t n_bytes = item.state.size();

    size_t n_waiting = 0;
    bool   accepted  = false;

    {
        std::lock_guard<std::mutex> lock(mtx);

        n_waiting = pending_bytes;
        accepted  = pending.empty() || pending_bytes + n_bytes <= PCACHE_DISK_PENDING_MAX;

        if (accepted) {
            pending_bytes += n_bytes;
            pending.push_back(std::move(item));
        }
    }

    if (accepted) {
        cv.notify_one();
    } else {
        SRV_WRN("prompt cache disk: dropping evicted %zu-token entry, %.1f MiB, %.1f MiB already waiting\n",
                item.keys.size(), n_bytes / (1024.0 * 1024.0), n_waiting / (1024.0 * 1024.0));
    }
}

bool server_prompt_cache_disk::take(const server_tokens & tokens_new, float & f_keep_best, float & f_sim_best, server_prompt_cache_state & out, int32_t id_slot) {
    collect();

    const llama_tokens keys_new = pcache_keys(tokens_new);

    float f_keep_win = f_keep_best;
    float f_sim_win  = f_sim_best;

    auto   it_best = index.end();
    size_t lcp_win = 0;

    for (auto it = index.begin(); it != index.end(); ++it) {
        const size_t lcp_cur = pcache_usable_prefix(it->keys, keys_new, tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->keys.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (f_keep_win < f_keep_cur && f_sim_win < f_sim_cur) {
            f_keep_win = f_keep_cur;
            f_sim_win  = f_sim_cur;

            it_best = it;
            lcp_win = lcp_cur;
        }
    }

    if (it_best == index.end()) {
        return false;
    }

    const size_t n_tokens = it_best->keys.size();

    const int64_t t_start = ggml_time_us();

    size_t n_bytes = 0;
    std::string err = pcache_disk_read(it_best->path, it_best->id, fingerprint, it_best->keys, out, n_bytes);

    if (err.empty()) {
        try {
            out.prompt.tokens = pcache_restored_tokens(it_best->keys, lcp_win, tokens_new);
        } catch (const std::exception & e) {
            err = e.what();
        }
    }

    // consumed by a restore, even a failed one
    remove_entry(it_best);

    if (!err.empty()) {
        SRV_WRN("prompt cache disk: slot %d failed to read %zu-token entry: %s\n", id_slot, n_tokens, err.c_str());

        return false;
    }

    const double t_ms = std::max((ggml_time_us() - t_start) / 1000.0, 1e-3);

    SRV_INF("prompt cache disk: slot %d read %zu-token entry, %.1f MiB in %.1f ms (%.0f MB/s)\n",
            id_slot, n_tokens, n_bytes / (1024.0 * 1024.0), t_ms, n_bytes / 1e3 / t_ms);

    f_keep_best = f_keep_win;
    f_sim_best  = f_sim_win;

    return true;
}

void server_prompt_cache_disk::pin(const server_tokens & tokens_next) {
    // pin before the size limit runs, it could remove the entry otherwise
    receive();

    const llama_tokens keys_next = pcache_keys(tokens_next);

    auto   it_best  = index.end();
    size_t lcp_best = 0;

    for (auto it = index.begin(); it != index.end(); ++it) {
        it->pinned = false;

        const size_t lcp_cur = pcache_usable_prefix(it->keys, keys_next, tokens_next);

        if (lcp_cur > lcp_best && float(lcp_cur) / it->keys.size() >= 0.25f) {
            lcp_best = lcp_cur;
            it_best  = it;
        }
    }

    if (it_best != index.end()) {
        it_best->pinned = true;
    }

    enforce_limit();
}

void server_prompt_cache_disk::unpin() {
    for (auto & e : index) {
        e.pinned = false;
    }
}

void server_prompt_cache_disk::remove_contained(const server_tokens & tokens) {
    remove_contained(pcache_keys(tokens));
}

void server_prompt_cache_disk::remove_contained(const llama_tokens & keys) {
    collect();

    for (auto it = index.begin(); it != index.end();) {
        const auto it_cur = it++;

        if (pcache_covers(keys, it_cur->keys)) {
            SRV_TRC(" - removing obsolete disk entry with length %zu\n", it_cur->keys.size());

            remove_entry(it_cur);
        }
    }
}

void server_prompt_cache_disk::flush(std::list<server_prompt_cache_state> && states, int64_t deadline_ms) {
    stop_worker();

    // what the worker did not reach was evicted before everything that is in the RAM cache
    std::list<pending_state> queue;
    queue.swap(pending);
    pending_bytes = 0;

    for (auto & state : states) {
        queue.push_back({ pcache_keys(state.prompt.tokens), std::move(state) });
    }

    states.clear();

    const int64_t t_start = ggml_time_us();

    // ids grow with the age of the state, the newest is written first
    const uint64_t id_first = next_id;
    next_id += queue.size();

    size_t n_written = 0;
    size_t n_bytes_written = 0;
    size_t n_failed = 0;
    size_t n_unkeyable = 0;
    size_t n_contained = 0;
    size_t n_late = 0;

    const auto is_in_longer_state = [&queue](const llama_tokens & keys) {
        return std::any_of(queue.begin(), queue.end(), [&keys](const pending_state & other) {
            return other.keys.size() > keys.size() && pcache_covers(other.keys, keys) && !pcache_has_unkeyable(other.keys);
        });
    };

    while (!queue.empty()) {
        const uint64_t id = id_first + queue.size() - 1;

        pending_state item = std::move(queue.back());
        queue.pop_back();

        if (pcache_has_unkeyable(item.keys)) {
            n_unkeyable++;

            continue;
        }

        if (is_contained(item.keys) || is_in_longer_state(item.keys)) {
            n_contained++;

            continue;
        }

        // assume 1 GB/s: a write that runs past the deadline can be force-killed by the router
        if (ggml_time_ms() + (int64_t) (item.state.size() >> 20) > deadline_ms) {
            n_late++;

            continue;
        }

        const std::string path = pcache_disk_path(dir, prefix, id);

        size_t n_bytes = 0;
        const std::string err = pcache_disk_store(path, id, fingerprint, item.keys, item.state, n_bytes);

        if (!err.empty()) {
            SRV_WRN("prompt cache disk: failed to write %zu-token entry: %s\n", item.keys.size(), err.c_str());

            n_failed++;

            continue;
        }

        // the index stays sorted by id
        remove_contained(item.keys);
        index.insert(std::find_if(index.begin(), index.end(), [id](const entry & e) { return e.id > id; }),
                entry{ std::move(item.keys), path, n_bytes, false, id });

        n_written++;
        n_bytes_written += n_bytes;
    }

    enforce_limit();

    SRV_INF("prompt cache disk: exit flush wrote %zu entries, %.1f MiB in %.0f ms, skipped %zu with media without id, %zu contained, %zu past the deadline, %zu failed\n",
            n_written, n_bytes_written / (1024.0 * 1024.0), (ggml_time_us() - t_start) / 1000.0, n_unkeyable, n_contained, n_late, n_failed);
}

size_t server_prompt_cache_disk::n_entries() {
    collect();

    return index.size();
}

size_t server_prompt_cache_disk::size() {
    collect();

    return indexed_size();
}

void server_prompt_cache_disk::collect() {
    receive();
    enforce_limit();
}

void server_prompt_cache_disk::receive() {
    std::lock_guard<std::mutex> lock(mtx);

    index.splice(index.end(), done);
}

void server_prompt_cache_disk::enforce_limit() {
    while (indexed_size() > limit_size) {
        const auto it_oldest = std::find_if(index.begin(), index.end(), [](const entry & e) { return !e.pinned; });
        if (it_oldest == index.end()) {
            return;
        }

        SRV_INF("prompt cache disk: removing oldest entry (%zu tokens, %.1f MiB)\n",
                it_oldest->keys.size(), it_oldest->size / (1024.0 * 1024.0));

        remove_entry(it_oldest);
    }
}

bool server_prompt_cache_disk::is_contained(const llama_tokens & keys) const {
    return std::any_of(index.begin(), index.end(), [&keys](const entry & e) {
        return pcache_covers(e.keys, keys);
    });
}

size_t server_prompt_cache_disk::indexed_size() const {
    size_t res = 0;

    for (const auto & e : index) {
        res += e.size;
    }

    return res;
}

void server_prompt_cache_disk::remove_entry(std::list<entry>::iterator it) {
    std::error_code ec;
    std::filesystem::remove(it->path, ec);

    index.erase(it);
}

//
// server_prompt_cache
//
size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

void server_prompt_cache::reserve(const server_tokens & tokens_next) {
    release();

    auto it_best = states.end();
    int lcp_best = 0;

    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_next);

        // same rule as load(): an entry that would lose most of its context is not a candidate
        if (lcp_cur > lcp_best && float(lcp_cur) / it->prompt.tokens.size() >= 0.25f) {
            lcp_best = lcp_cur;
            it_best  = it;
        }
    }

    if (it_best != states.end()) {
        reserved.splice(reserved.end(), states, it_best);
    }

    if (disk) {
        disk->pin(tokens_next);
    }
}

void server_prompt_cache::release() {
    states.splice(states.end(), reserved);

    if (disk) {
        disk->unpin();
    }
}

void server_prompt_cache::evict_oldest() {
    if (disk) {
        disk->push(std::move(states.front()));
    }

    states.pop_front();
}

server_prompt_cache_state * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size()) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // keep only the newest checkpoints: on recurrent models each one is a full state copy, and a
    // restore seldom rewinds past the last few turns
    const size_t n_ckpt_keep = std::min<size_t>(prompt.checkpoints.size(), 4);
    const auto ckpt_first = std::prev(prompt.checkpoints.end(), n_ckpt_keep);

    // calculate checkpoints size to see if it will fit with the prompt
    size_t checkpoints_size = 0;
    for (auto it = ckpt_first; it != prompt.checkpoints.end(); ++it) {
        checkpoints_size += it->size();
    }

    const size_t state_size_new = state_size_tgt + state_size_dft + checkpoints_size;

    // skip over-limit entries to avoid disturbing the cache
    if (limit_size > 0 && state_size_new > limit_size) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return nullptr;
    }

    // remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (len == (int) it->prompt.tokens.size()) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);

            it = states.erase(it);
        } else {
            ++it;
        }
    }

    if (disk) {
        disk->remove_contained(prompt.tokens);
    }

    if (limit_size > 0) {
        // make room before allocating the new vectors to avoid breaching the limit
        while (!states.empty() && size() + state_size_new > limit_size) {
            SRV_WRN(" - making room for prompt cache entry, removing oldest entry (%zu tokens, %.3f MiB)\n",
                    states.front().prompt.tokens.size(), states.front().size() / (1024.0 * 1024.0));

            evict_oldest();
        }
    }

    std::vector<uint8_t> state_data_tgt;
    std::vector<uint8_t> state_data_dft;

    // check if we can allocate enough memory for the new state
    try {
        state_data_tgt.resize(state_size_tgt);
        state_data_dft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4*size());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    states.push_back({
        /*.prompt =*/ {
            /*.tokens      =*/ prompt.tokens.clone(),
            /*.checkpoints =*/ std::list<common_prompt_checkpoint>(ckpt_first, prompt.checkpoints.end()),
        },
        /*.data   =*/ {
            /*.main =*/ std::move(state_data_tgt),
            /*.drft =*/ std::move(state_data_dft),
        },
    });

    return &states.back();
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    // a reserved entry rejoins as the newest, so the update() after this load spares it too
    // the disk pin stays until the disk is searched
    states.splice(states.end(), reserved);

    const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float f_sim_best  = float(lcp_best) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    auto it_best = states.end();

    // the longest prefix any entry shares with the prompt, whether or not it qualifies
    int lcp_any = 0;

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n", it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        lcp_any = std::max(lcp_any, lcp_cur);

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;

            it_best = it;
        }
    }

    if (disk) {
        server_prompt_cache_state st;

        if (disk->take(tokens_new, f_keep_best, f_sim_best, st, id_slot)) {
            states.push_back(std::move(st));

            it_best = std::prev(states.end());
        }

        disk->unpin();
    }

    // a miss is silent otherwise, and under several agents "it keeps prefilling" is
    // indistinguishable from an eviction, an admission failure or a bad match without this
    if (it_best == states.end()) {
        SRV_INF("prompt cache: slot %d keeps its own %zu tokens (lcp %d of %zu); no better entry among %zu (+%zu on disk), best lcp %d\n",
                id_slot, prompt.tokens.size(), lcp_best, tokens_new.size(), states.size(), disk_n_entries(), lcp_any);
    }

    if (it_best != states.end()) {
        SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

        const int64_t t_start = ggml_time_us();

        {
            auto & data = it_best->data.main;

            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_ERR("failed to restore state with size %zu\n", size);

                return false;
            }

            data.clear();
            data.shrink_to_fit();
        }

        {
            auto & data = it_best->data.drft;

            if (!data.empty()) {
                GGML_ASSERT(ctx_dft);

                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore state with size %zu\n", size);

                    return false;
                }

                data.clear();
                data.shrink_to_fit();
            }
        }

        SRV_INF("prompt cache: slot %d restored %zu-token entry (f_keep = %.3f, f_sim = %.3f) for a %zu-token prompt in %.1f ms\n",
                id_slot, it_best->prompt.tokens.size(), f_keep_best, f_sim_best, tokens_new.size(), (ggml_time_us() - t_start) / 1000.0);

        prompt = std::move(it_best->prompt);

        states.erase(it_best);
    }

    return true;
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        while (!states.empty() && size() > limit_size) {
            SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", states.front().size() / (1024.0 * 1024.0));

            evict_oldest();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front().size() / (1024.0 * 1024.0));

            evict_oldest();
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.prompt.n_tokens(), state.prompt.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}
