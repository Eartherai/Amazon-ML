# Model license audit

Verified25September2026IST using publisher model metadata and repository license endpoints. No deep model is part of the current selected solution. Revision pins below are candidates for a future measured rescue experiment; licenses do not establish effectiveness.

| Model | Weight revision | Weight license | Publisher code license | Final status |
|---|---|---|---|---|
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3/blob/5617a9f61b028005a4858fdac845db406aefb181/README.md) | `5617a9f61b028005a4858fdac845db406aefb181` | mit | MIT | Not selected; benchmark pending |
| [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small/blob/614241f622f53c4eeff9890bdc4f31cfecc418b3/README.md) | `614241f622f53c4eeff9890bdc4f31cfecc418b3` | mit | MIT | Not selected; benchmark pending |
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/README.md) | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | apache-2.0 | Publisher LICENSE lookup returned404; unresolved | Not selected; benchmark pending |

[FlagEmbedding code license](https://github.com/FlagOpen/FlagEmbedding/blob/master/LICENSE): MIT, license blob360931513aa6c02f933a403202afa99ac2c5bc88. [E5 publisher code license](https://github.com/microsoft/unilm/blob/master/LICENSE): MIT, license blobae241a567f3de21656c00c7b341a6f8701e405b3. Publisher APIs identified these licenses; pin the actual installed runtime revision and include its license when selecting an implementation. Qwen publisher code verification is unresolved; exclude that implementation from final use until independently verified.

HuggingFace metadata reports117,653,760F32parameters plus512I64buffer values for multilingual-e5-small and595,776,512BF16parameters for Qwen3Embedding0.6B. E5-small is the first small-model candidate. BGE-M3 metadata did not return a safetensors parameter total; verify instantiated parameter count before approval. All final selected models must satisfy the official8B limit and license rule.

Benchmark only against competition-record lexical misses and full distractors. Compare native/light/transliterated representations without external identity lookup. No external identity data, remote enrichment, or test-label inference. Prefer bounded miss/hard-negative evaluation before full-target embedding cost. GPU launch remains blocked by zero active GPU quota; requests are open, not approved.
