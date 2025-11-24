# this file is gonna contain the throghput(tokens/sec) vs batch size scaling for the model internlm/internlm2-chat-7b with vllm
from vllm import LLM, SamplingParams
import time
import torch

model_name = "internlm/internlm2-chat-7b"

prompts = [
    "Tell me a joke about cats.",
    "Explain the theory of relativity in simple terms.",
    "What are the health benefits of regular exercise?",
    "How does photosynthesis work in plants?",
]

batch_sizes = [1, 2, 4, 8, 16, 32, 64]

def measure_throughput(model, prompts, batch_size, max_new_tokens=50):
    # preparing the batch of prompts
    print(f"Measuring throughput for batch size: {batch_size}")
    batch_prompts = prompts * (batch_size // len(prompts)) + prompts[: batch_size % len(prompts)]

    start_time = time.time()
    outputs = model.generate(
        batch_prompts,
        sampling_params=SamplingParams(max_tokens=max_new_tokens),
    )
    end_time = time.time()

    total_tokens = sum(len(output.sequences[0]) for output in outputs)

    elapsed_time = end_time - start_time
    throughput = total_tokens / elapsed_time if elapsed_time > 0 else 0

    return throughput

if __name__ == "__main__":
    model = LLM(model_name, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True)

    results = {}

    for batch_size in batch_sizes:
        throughput = measure_throughput(model, prompts, batch_size)
        print(f"Batch Size: {batch_size}, Throughput: {throughput:.2f} tokens/sec")

        results[batch_size] = throughput

    # plotting the results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.title('Throughput vs Batch Size for internlm2-chat-7b')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (tokens/sec)')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.savefig('throughput_vs_batch_size_internlm2_chat_7b.png')
    plt.close()
