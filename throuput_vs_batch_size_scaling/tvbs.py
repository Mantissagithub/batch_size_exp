# this file is gonna contain the throghput(tokens/sec) vs batch size scaling for the model internlm/internlm2-chat-7b with vllm
from vllm import LLM, SamplingParams
import time
import torch

model_name = "internlm/internlm2-chat-7b"

prompts = [
    # Original: "Tell me a joke about cats."
    "You are a stand-up comedian known for witty animal puns and observational humor. Craft an original, family-friendly joke about cats that highlights their mischievous habit of knocking objects off tables at 3 AM, incorporating a clever pun on physics or gravity. Deliver it with perfect comedic timing, as if performing live to a giggling audience.",

    # Original: "Explain the theory of relativity in simple terms."
    "Act as a patient physics teacher explaining Einstein's theory of special and general relativity to a curious high school student who's never studied physics. Break it down into simple analogies—like comparing time dilation to a GPS satellite clock running slower in space, or gravity as curved spacetime like a trampoline with a bowling ball. Use everyday examples, avoid equations, and include a thought experiment to illustrate why nothing exceeds light speed.",

    # Original: "What are the health benefits of regular exercise?"
    "You are a certified fitness coach with a background in sports science. Provide a comprehensive overview of the top 10 evidence-based health benefits of regular aerobic and strength training exercise (at least 150 minutes per week), tailored for a busy adult in their 30s. Structure it with short-term vs. long-term effects, backed by simple stats from studies (e.g., WHO or CDC), and include practical tips like 'start with 10-minute walks' to build habits.",

    # Original: "How does photosynthesis work in plants?"
    "Pretend you're a biology professor leading a classroom demo on plant biology for middle schoolers. Explain the full process of photosynthesis step-by-step—from light absorption by chlorophyll in chloroplasts, to the light-dependent reactions splitting water and producing ATP/O2, then the Calvin cycle fixing CO2 into glucose. Use a flowchart analogy (e.g., sunlight as the 'power plant'), real-world examples like why leaves are green, and a simple equation in words: 6 CO2 + 6 H2O + light → C6H12O6 + 6 O2.",

    # New: Tech/Programming
    "You are a senior software engineer at a FAANG company mentoring a junior dev. Walk through optimizing a slow Python loop that processes 1 million rows of CSV data for duplicates, using pandas and NumPy. Include before/after code snippets, explain vectorization vs. iteration trade-offs, memory profiling tips with %timeit, and edge cases like large files (>10GB) on a laptop with 16GB RAM. Benchmark on realistic data shapes.",

    # New: History/Creativity
    "As a historical novelist like Hilary Mantel, write a 300-word immersive scene from the perspective of Cleopatra during her first meeting with Julius Caesar in 48 BCE. Focus on sensory details (Nile scents, Roman armor clinks), her strategic mindset calculating alliances, internal monologue blending ambition and vulnerability, and subtle foreshadowing of Egypt's fate. Use vivid, archaic-tinged language without modern anachronisms.",

    # New: Cooking/Practical
    "You're a Michelin-starred chef simplifying gourmet recipes for home cooks. Provide a foolproof, step-by-step recipe for vegan mushroom risotto for 4 servings, including ingredient substitutions for allergies (e.g., nut-free), exact timings/temps (e.g., Arborio rice simmered 18 mins), pro tips like 'stir clockwise for even creaminess,' nutritional breakdown per serving, and pairing suggestions with wine or sides.",

    # New: Philosophy/Ethics
    "Embody a Socratic philosopher debating modern AI ethics with a tech CEO. Pose 5 probing questions on the trolley problem applied to self-driving cars (e.g., 'Sacrifice one passenger to save five pedestrians?'), explore utilitarianism vs. deontology with real Tesla Autopilot examples, counterarguments from both sides, and conclude with actionable guidelines for responsible AI deployment.",

    # New: Space/Science
    "Channel Neil deGrasse Tyson explaining black holes to a sci-fi fan. Describe formation from massive star collapse, event horizon math via simple analogy (inescapable point like a cosmic one-way door), Hawking radiation evaporation over eons, and evidence from LIGO mergers or Event Horizon Telescope images. Debunk myths (e.g., 'not time machines') and speculate on wormholes with caveats.",

    # New: Productivity/Self-Help
    "You are a productivity expert like David Allen (Getting Things Done). Design a personalized weekly planner template for a freelancer juggling 3 clients, incorporating Eisenhower Matrix prioritization, Pomodoro sessions (25/5), zero-inbox email rituals, and reflection prompts. Format as a markdown table with columns for tasks, urgency/importance, time blocks, and wins/obstacles; include tools like Notion or Google Calendar integrations."
]

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

def measure_throughput(prompts, batch_size, max_new_tokens=50):

    model = LLM(model_name, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True)

    # preparing the batch of prompts
    print(f"Measuring throughput for batch size: {batch_size}")
    batch_prompts = prompts * (batch_size // len(prompts)) + prompts[: batch_size % len(prompts)]

    start_time = time.time()
    outputs = model.generate(
        batch_prompts,
        sampling_params=SamplingParams(max_tokens=max_new_tokens),
    )
    end_time = time.time()

    total_tokens = sum(len(outputs[0].outputs[0].text) for output in outputs)

    elapsed_time = end_time - start_time
    throughput = total_tokens / elapsed_time if elapsed_time > 0 else 0

    return throughput

if __name__ == "__main__":
    # model = LLM(model_name, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True)
#
    results = {}

    for batch_size in batch_sizes:
        throughput = measure_throughput(prompts, batch_size)
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

