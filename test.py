from vllm import LLM, SamplingParams

if __name__ == "__main__":
    model_name = "internlm/internlm-7b"

    llm = LLM(model=model_name, tensor_parallel_size=1, trust_remote_code=True)

    sampling_params = SamplingParams()

    output = llm.generate(
        ["Explain the theory of relativity in simple terms."],
        sampling_params
    )

    print(output)