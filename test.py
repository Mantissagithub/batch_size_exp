from vllm import LLMEngine, AsyncEngineArgs, SamplingParams

engine_args = AsyncEngineArgs.from_cli_args(["--model", model_name])
engine = LLMEngine.from_engine_args(engine_args, log_stats=True)

req_id = "long_prompt"
prompt = "The quick brown fox jumps over the lazy dog. " * 500
params = SamplingParams(max_tokens=100)
engine.add_request(req_id, prompt, params)

peak_usage = 0.0
while not engine.has_unfinished_requests():
    step_outputs = engine.step()
    stats = engine._get_stats()
    peak_usage = max(peak_usage, stats.gpu_cache_usage)
    print(f"Step KV usage: {stats.gpu_cache_usage:.2%}, blocks: {stats.num_gpu_total_blocks_used}")

print(f"Peak KV usage: {peak_usage:.2%}")
