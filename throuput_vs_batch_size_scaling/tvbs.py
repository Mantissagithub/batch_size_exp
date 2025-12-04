# this file is gonna contain the throghput(tokens/sec) vs batch size scaling for the model internlm/internlm2-chat-7b with vllm
from vllm import LLM, SamplingParams
import time
import torch
import tiktoken

model_name = "internlm/internlm2-chat-7b"

prompts = [
    "Tell me a joke about cats.",
    "Explain the theory of relativity in simple terms.",
    "What are the health benefits of regular exercise?",
    "How does photosynthesis work in plants?",
]

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

def count_tokens(text, model_name=model_name):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)

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

    # total_tokens = sum(len(outputs[0].outputs[0].text) for output in outputs)
    # counting the tokens using the method count_token()
    model_name = "cl100k_base"  # or any other model name compatible with tiktoken
    total_tokens = sum(count_tokens(output.outputs[0].text, model_name) for output in outputs)

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
    import numpy as np

    # Enhanced plotting with more comprehensive features
    plt.style.use('seaborn-v0_8-darkgrid')  # Professional styling
    fig, ax = plt.subplots(figsize=(12, 7))

    batch_sizes = list(results.keys())
    throughputs = list(results.values())

    # Main plot with enhanced styling
    line = ax.plot(batch_sizes, throughputs,
                  marker='o', markersize=8,
                  linewidth=2.5, color='#2E86AB',
                  markerfacecolor='#A23B72',
                  markeredgecolor='white',
                  markeredgewidth=1.5,
                  label='Measured Throughput')

    # Add data point annotations
    for bs, tp in zip(batch_sizes, throughputs):
        ax.annotate(f'{tp:.1f}',
                    xy=(bs, tp),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=8,
                    alpha=0.8)

    # Titles and labels with enhanced formatting
    ax.set_title('LLM Inference Throughput vs Batch Size\nInternLM2-Chat-7B Performance Analysis',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Batch Size (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (tokens/sec, log scale)', fontsize=12, fontweight='bold')

    # Logarithmic scales
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    # Enhanced grid
    ax.grid(True, which="major", ls="-", alpha=0.4, linewidth=1.2)
    ax.grid(True, which="minor", ls=":", alpha=0.2, linewidth=0.8)

    # Add reference lines for key batch sizes
    if batch_sizes:
        optimal_idx = np.argmax(throughputs)
        optimal_bs = batch_sizes[optimal_idx]
        optimal_tp = throughputs[optimal_idx]

        ax.axvline(optimal_bs, color='green', linestyle='--',
                  alpha=0.6, linewidth=1.5,
                  label=f'Peak: BS={optimal_bs}')
        ax.axhline(optimal_tp, color='green', linestyle='--',
                  alpha=0.3, linewidth=1)

    # Add statistics text box
    stats_text = f'Peak Throughput: {max(throughputs):.2f} tok/s\n'
    stats_text += f'Optimal Batch Size: {batch_sizes[np.argmax(throughputs)]}\n'
    stats_text += f'Batch Size Range: {min(batch_sizes)}-{max(batch_sizes)}'

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Legend
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    # Customize spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Tight layout and save
    plt.tight_layout()
    plt.savefig('throughput_vs_batch_size_internlm2_chat_7b.png',
                dpi=300, bbox_inches='tight')
    plt.close()

