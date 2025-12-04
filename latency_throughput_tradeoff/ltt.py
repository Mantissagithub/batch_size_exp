# this file is gonna contain the latency vs throughput tradeoff for the model internlm/internlm2-chat-7b with vllm
# latency as in time-to-first-token and total latency across same batch size

import time
from vllm import LLM, SamplingParams
import torch

model_name = "internlm/internlm2-chat-7b"

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

def measure_latency_throughput(model_name, batch_size, prompt, max_new_tokens=50):
    print(f"Measuring for batch size: {batch_size}")
    llm = LLM(model=model_name, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True)

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
    )

    inputs = [prompt] * batch_size

    output = llm.generate(inputs, sampling_params, batch_size=batch_size)

    print(f"Output for the batch size {batch_size}: {output}")
    metrics = output[0].metrics

    ttft = metrics.fist_token_time - metrics.first_scheduled_time

    total_latency = metrics.finished_time - metrics.first_scheduled_time

    return {
        "batch_size": batch_size,
        "time_to_first_token": ttft,
        "total_latency": total_latency,
        "throughput": batch_size / total_latency,
    }

if __name__ == "__main__":
    prompt = "Explain the theory of relativity in simple terms."

    results = []
    for bs in batch_sizes:
        res = measure_latency_throughput(model_name, bs, prompt)
        results.append(res)

    # need to plot a graph now
    import matplotlib.pyplot as plt

    print(results)

    # Extract data for plotting
    batch_sizes_results = [r["batch_size"] for r in results]
    ttft_values = [r["time_to_first_token"] for r in results]
    total_latency_values = [r["total_latency"] for r in results]
    throughput_values = [r["throughput"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(batch_sizes_results, ttft_values, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time to First Token (s)', fontsize=12, fontweight='bold')
    ax1.set_title('TTFT vs Batch Size\ninternlm2-chat-7b', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xscale('log', base=2)

    ax2.plot(batch_sizes_results, total_latency_values, marker='s', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Latency (s)', fontsize=12, fontweight='bold')
    ax2.set_title('Total Latency vs Batch Size\ninternlm2-chat-7b', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig('vllm_latency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nGraph saved as 'vllm_latency_analysis.png'")


