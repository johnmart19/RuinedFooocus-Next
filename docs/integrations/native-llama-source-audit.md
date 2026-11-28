# Exact native llama source reviews

`model_capabilities.py` retains each artifact's original b12110 source record.
The additive `resources/model_metadata/native_source_audits.json` records the
separate b12327 review at `81f21f4b27fc8252f11404a006744045aac5ea6e`, checked
2026-11-12. This exact build can inherit only the reviewed architecture/task
contracts from the original b12110 record. Unknown builds and architectures stay
unknown; artifact restrictions, matching projector identities and unavailable
image tasks stay authoritative. Neither version ordering nor a familiar filename
grants source support.

The additive b12366 review at `31953bcf74ba489a8bbb0c936c5fcf24fcf9485c`,
checked 2026-11-13, preserves the same artifact-specific task restrictions.
Its 39-commit delta from b12327 was reviewed separately. Twenty inspected files
are byte-identical, including LFM3, GPT-OSS, Qwen3/Qwen2-VL/Qwen3.5-MoE,
the chat/template entry point, mtmd public API and native fit/log/model-loader
sources. Eight inspected files changed. Qwen3.5 adds fused tensor paths with
the original separate-tensor fallbacks; mtmd tightens projector tensor/merge
validation and corrects MiniCPM dimensions. Server media restrictions retain RF's
inline image data URLs, while tool/CORS changes do not enable RF server tools.
The sidecar records exact file hashes for both comparisons, preserving b12110
as the original artifact source contract rather than rewriting its evidence.

The reviewed [b12366 delta](https://github.com/ggml-org/llama.cpp/compare/81f21f4b27fc8252f11404a006744045aac5ea6e...31953bcf74ba489a8bbb0c936c5fcf24fcf9485c)
also preserves the inspected mandatory LFM3/GPT-OSS input placement and JSON log
fields. Managed b12366 children can therefore use the same bounded log parser
and input-buffer attribution. Unknown/custom children still cannot inherit it.
This is source/transport support, without new model, device or quality evidence.

The review checks the native architecture factory and each retained family's
hparams, GGUF tensor loader and graph entry points, plus the existing multimodal
and server APIs. The reviewed text families are bailingmoe3, gemma3, gemma4,
gpt-oss, granite, lfm2, lfm3, llama, mistral3, qwen2, qwen2vl, qwen3, qwen35 and
qwen35moe. Existing image-capable artifacts also retain their reviewed mtmd routes
for gemma3, gemma4, granite, lfm3, llama, mistral3, qwen2vl, qwen35 and qwen35moe.
This does not add image support to text-only artifacts.

The sidecar preserves candidate and base SHA-256 identities for every inspected
source file. All files in the original b12327 inspection changed from b12110;
that review does not claim unchanged implementation or demonstrated artifact
compatibility. Binary compatibility, device execution, model behavior, quality,
speed and stability require separate qualification. Existing b12110 runtime
qualification remains attached to its original source/artifacts.

Primary source owners at the exact reviewed revision:

- [Architecture factory and input placement](https://github.com/ggml-org/llama.cpp/blob/81f21f4b27fc8252f11404a006744045aac5ea6e/src/llama-model.cpp).
- [Native model implementations](https://github.com/ggml-org/llama.cpp/tree/81f21f4b27fc8252f11404a006744045aac5ea6e/src/models),
  including [LFM3](https://github.com/ggml-org/llama.cpp/blob/81f21f4b27fc8252f11404a006744045aac5ea6e/src/models/lfm3.cpp)
  and [GPT-OSS/OpenAI MoE](https://github.com/ggml-org/llama.cpp/blob/81f21f4b27fc8252f11404a006744045aac5ea6e/src/models/openai-moe.cpp).
- [mtmd model/projector loading](https://github.com/ggml-org/llama.cpp/blob/81f21f4b27fc8252f11404a006744045aac5ea6e/tools/mtmd/mtmd-model.cpp)
  and [mtmd public API](https://github.com/ggml-org/llama.cpp/blob/81f21f4b27fc8252f11404a006744045aac5ea6e/tools/mtmd/mtmd.h).
- [Server chat endpoint](https://github.com/ggml-org/llama.cpp/blob/81f21f4b27fc8252f11404a006744045aac5ea6e/tools/server/server.cpp)
  and [tool/reasoning request handling](https://github.com/ggml-org/llama.cpp/blob/81f21f4b27fc8252f11404a006744045aac5ea6e/tools/server/server-common.cpp).

```sh
python tools/check_cpu.py test_model_capabilities.py test_model_registry.py
```

These passive checks cover exact build/architecture/task gating and preservation
of old records. They execute no native binary or model. Native acquisition and
rollback remain described in [the SDK contract](native-llama-sdk.md).
