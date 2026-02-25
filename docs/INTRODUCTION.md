# Strands Robots Sim - Technical Introduction

**LLM-controlled robot manipulation in simulation with Isaac-GR00T and Libero**

## What It Is

A Python framework that enables AI agents to control robots in simulation environments through tool-based interfaces. Built on [Strands Agents SDK](https://github.com/strands-agents/sdk-python), it provides two execution modes:

1. **SimEnv**: Full episode execution - agent specifies task, policy runs to completion
2. **SteppedSimEnv**: Iterative control - agent observes and adapts every N steps with visual feedback

**Key Benefit**: Enables rapid prototyping and algorithm development in a safe, simulated environment without requiring physical robotic hardware. Perfect for iterating on agent strategies, testing VLA policies, and validating approaches before real-world deployment.

## System 1 vs System 2 Thinking in Robot Control

This framework implements a **dual-system architecture** inspired by cognitive science (Kahneman's System 1 and System 2 thinking):

```
┌─────────────────────────────────────────────────────────────────┐
│                     SYSTEM 2: Deliberate Planning               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Strands Agent (Claude LLM)                               │  │
│  │  • High-level task reasoning and planning                 │  │
│  │  • Natural language understanding                         │  │
│  │  • Task decomposition and strategy adaptation             │  │
│  │  • Error detection and recovery planning                  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                  │
│                   Language Instructions                         │
│                   "move gripper to block"                       │
│                              ↓                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   SYSTEM 1: Fast Action Execution               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  GR00T VLA Policy (Vision-Language-Action Model)          │  │
│  │  • Sensorimotor control and fast reactions                │  │
│  │  • Vision + Language → Robot Actions                      │  │
│  │  • Low-level trajectory execution                         │  │
│  │  • Real-time feedback processing                          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                  │
│                     Robot Actions                               │
│                  [joint positions, gripper state]               │
│                              ↓                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Libero Simulation Environment                            │  │
│  │  • Physics simulation and rendering                       │  │
│  │  • State updates and reward computation                   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Matters

**System 2 (Strands Agent)** provides:
- **Deliberate reasoning**: Can break down complex tasks into subtasks
- **Contextual understanding**: Interprets high-level goals and environmental context
- **Adaptive planning**: Adjusts strategy based on visual feedback and task progress
- **Error recovery**: Detects failures and generates alternative approaches

**System 1 (GR00T VLA)** provides:
- **Fast execution**: 40-160ms inference latency for real-time control
- **Sensorimotor skills**: Learned visuomotor policies for manipulation
- **Generalization**: Pre-trained on diverse tasks, adapts to new instructions
- **Reactive control**: Handles low-level motor commands without explicit planning

### Two Modes of Collaboration

1. **SimEnv Mode (Single System 2 Call)**
   - System 2 generates complete task description once
   - System 1 executes entire episode autonomously
   - Best for: Well-defined tasks, benchmarking

2. **SteppedSimEnv Mode (Iterative System 2 Guidance)**
   - System 2 observes progress and provides guidance every N steps
   - System 1 executes short action sequences
   - System 2 adapts instructions based on visual feedback
   - Best for: Complex tasks, error-prone scenarios, research

## Core Capabilities

**Mode 1: SimEnv - Direct Task Execution**
```python
from strands import Agent
from strands_robots_sim import SimEnv, gr00t_inference

# Agent specifies task, policy executes to completion
agent = Agent(
    model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    tools=[SimEnv(env_type="libero"), gr00t_inference]
)

# Start GR00T inference service
agent.tool.gr00t_inference(
    action="start",
    checkpoint_path="/data/checkpoints/gr00t-libero-90",
    port=8000
)

# Execute task - policy runs full episode
agent.tool.my_sim(
    action="execute",
    instruction="pick up the red block and place it in the drawer",
    policy_port=8000,
    max_episodes=5,
    record_video=True
)
# Returns: success rate, videos saved to ./rollouts/
```

**Mode 2: SteppedSimEnv - Iterative Agent Control**
```python
from strands import Agent
from strands_robots_sim import SteppedSimEnv, gr00t_inference

# Agent controls robot through step-by-step planning with visual feedback
agent = Agent(
    model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    tools=[SteppedSimEnv(env_type="libero"), gr00t_inference]
)

# Agent decomposes "pick up block and place in drawer" into:
# 1. Locate block → execute 10 steps → observe camera feedback
# 2. Grasp block → execute 10 steps → check success
# 3. Move to drawer → execute 10 steps → adapt if needed
# 4. Place block → execute 10 steps → verify reward=1.0
```

## Technical Architecture

**SimEnv Architecture (Direct Execution):**
```
┌─────────────────────────────────────────────────────────────┐
│  LLM Agent (Claude Sonnet 4.5)                              │
│  • Task specification                                       │
│  • Natural language instructions                            │
└────────────┬────────────────────────────────────────────────┘
             │ One-shot Tool Call
             │ "execute task X"
             │
┌────────────▼───────────────┐      ┌─────────────────────────┐
│  SimEnv                    │      │  GR00T Inference        │
│  • Run full episode        │──────│  • Vision-Language-     │
│  • Record video            │ ZMQ  │    Action (VLA) policy  │
│  • Return final results    │      │  • Flow matching        │
└────────────┬───────────────┘      │  • Diffusion denoising  │
             │                      └─────────────────────────┘
             │
┌────────────▼───────────────┐
│  Libero Simulation         │
│  • Robot manipulation      │
│  • 90+ benchmark tasks     │
│  • Multi-camera rendering  │
└────────────────────────────┘
```

**SteppedSimEnv Architecture (Iterative Control):**
```
┌─────────────────────────────────────────────────────────────┐
│  LLM Agent (Claude Sonnet 4.5)                              │
│  • Task decomposition & planning                            │
│  • Visual observation analysis                              │
│  • Adaptive instruction generation                          │
└────────────┬────────────────────────────────────────────────┘
             │ Iterative Tool Calls
             │ (observe → plan → execute loop)
             │
┌────────────▼───────────────┐      ┌─────────────────────────┐
│  SteppedSimEnv             │      │  GR00T Inference        │
│  • Execute N steps         │──────│  • Vision-Language-     │
│  • Return camera images    │ ZMQ  │    Action (VLA) policy  │
│  • State & reward feedback │      │  • Flow matching        │
└────────────┬───────────────┘      │  • Diffusion denoising  │
             │                      └─────────────────────────┘
             │
┌────────────▼───────────────┐
│  Libero Simulation         │
│  • Robot manipulation      │
│  • 90+ benchmark tasks     │
│  • Multi-camera rendering  │
└────────────────────────────┘
```

## Key Differentiators

| Feature | SimEnv | SteppedSimEnv |
|---------|--------|---------------|
| **Control** | Agent specifies task once | Agent observes & adapts every N steps |
| **Feedback** | Final reward only | Camera images + state + reward per batch |
| **Use Case** | Known tasks, direct execution | Complex tasks, hierarchical planning |
| **Agent Role** | Task specification | Active control loop with visual grounding |

**Why SteppedSimEnv?**
- **Visual grounding**: Agent sees camera feeds and can verify progress
- **Error recovery**: Can detect failures and retry with different instructions
- **Hierarchical planning**: Natural decomposition of complex tasks into subtasks
- **Interpretability**: Full trace of agent decisions and observations

## Quick Start

**Prerequisites:**
```bash
# 1. Setup Isaac-GR00T Docker container with GPU
bash scripts/setup-gr00t-gpu.sh

# 2. Install environment
conda env create -f environment.yml
```

**Run Stepped Example:**
```bash
python examples/libero_stepped_example.py
# Agent decomposes tasks and controls robot step-by-step
```

**Run Standard Example:**
```bash
python examples/libero_example.py
# Direct task execution without agent control loop
```

## Supported Environments

**Libero Task Suites:**
- **libero_spatial**: Spatial reasoning tasks (10 tasks)
- **libero_object**: Object-centric tasks (10 tasks)
- **libero_goal**: Goal-conditioned manipulation (10 tasks)
- **libero_10**: Standard 10-task benchmark
- **libero_90**: Extended 90-task benchmark (comprehensive evaluation)

**GR00T Policy Integration:**
- Isaac-GR00T N1.5-3B foundation model
- Available checkpoints: spatial, 90-task, and other task-specific variants
- Communication via ZMQ protocol

**Coming Soon:** IsaacLab...

## Performance Tuning

**GR00T Inference:**
- `denoising_steps=4`: Fast (40ms), recommended for real-time
- `denoising_steps=8`: Balanced (80ms), default
- `denoising_steps=16`: High quality (160ms), benchmarking only

**Agent Models:**
- Claude Sonnet 4.5: Best balance (speed + reasoning)
- Claude Opus 4.5: Maximum reasoning (slower)
- Claude Haiku: Fast but limited planning

**Execution Overhead:**

**SimEnv Mode (Single Episode):**

When using natural language with `agent("Run the task...")`, expect overhead on top of actual simulation time:

| Component | Overhead | Description |
|-----------|----------|-------------|
| LLM Call | 3-5s | Agent processes natural language, decides tool to call, generates parameters |
| Environment Reset | 0.5-1s | Loading initial state, initializing physics engine, setting up rendering |
| Video Encoding | 0.5-1s | Encoding frames to MP4 and writing to disk |
| Policy Communication | 0.2-0.5s | ZMQ communication setup with GR00T server |
| Python Setup | 0.5-1s | Import overhead and agent initialization (first run) |
| **Total Overhead** | **~5-8s** | Fixed overhead per episode |

**Example:** A 9-second simulation will take ~14-17 seconds total execution time.

**SteppedSimEnv Mode (Iterative Control):**

SteppedSimEnv operates on **1 episode per run** but with iterative agent control. It has **significantly higher overhead** due to multiple LLM calls for iterative planning:

| Component | Overhead | Description |
|-----------|----------|-------------|
| Base Setup | 5-10s | Initial environment setup, policy initialization |
| LLM Call per Iteration | 3-5s | Agent observes, reasons, decides next action (repeated N times) |
| Execution per Iteration | Variable | Depends on steps_per_call (e.g., 10 steps) |
| **Total Overhead** | **~5-10s + (3-5s × iterations)** | Scales with task complexity |

**Example:** Task requiring 5 agent iterations = ~5-10s base + (3-5s × 5) = **~20-35s overhead** + actual simulation time.

**Notes:**
- **SimEnv**: One LLM call, lower overhead, best for known tasks
- **SteppedSimEnv**: Multiple LLM calls, higher overhead, enables adaptive planning and error recovery
- The LLM calls are the largest bottleneck in both modes (~50-60% of overhead)
- Direct tool calls (`agent.tool.my_sim(...)`) bypass LLM processing, saving 3-5 seconds per call
- For SimEnv, overhead is relatively fixed; for SteppedSimEnv, it scales with task complexity
- SteppedSimEnv's overhead trade-off: higher time cost for better success rates on complex tasks
- For batch experiments, consider the overhead when estimating total runtime

## Use Cases

1. **Quick prototyping without hardware**: Develop and test robot control algorithms in a simulated, controlled environment without needing physical robotic arms or hardware setup - ideal for rapid iteration and experimentation
2. **Robotic task planning research**: Study how LLMs decompose complex manipulation tasks
3. **VLA policy evaluation**: Test vision-language-action models on standardized benchmarks
4. **Agent-in-the-loop simulation**: Adaptive robot control with human-like reasoning
5. **Benchmark reproduction**: Libero-90 with full video recording and metrics

## Technical Stack

- **Agent Framework**: [Strands SDK](https://github.com/strands-agents/sdk-python)
- **VLA Policy**: [Isaac-GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T) (only policy currently implemented)
- **Simulation**: [Libero](https://github.com/Lifelong-Robot-Learning/LIBERO)
- **LLM**: Claude 4.x via AWS Bedrock
- **Communication**: ZMQ for GR00T policy inference

## Repository Structure

```
strands-robots-sim/
├── strands_robots_sim/
│   ├── __init__.py             # Package exports
│   ├── sim_env.py              # Full episode execution
│   ├── stepped_sim_env.py      # Iterative agent control
│   ├── envs/                   # Environment implementations
│   │   ├── __init__.py         # Environment factory
│   │   ├── base.py             # Base environment interface
│   │   └── env_libero.py       # Libero integration
│   ├── policies/               # Policy implementations
│   │   ├── __init__.py         # Policy base + factory
│   │   └── groot/              # GR00T implementation (only VLA implemented)
│   │       ├── __init__.py
│   │       ├── client.py       # ZMQ client
│   │       └── data_config.py  # Embodiment configs
│   └── tools/
│       ├── __init__.py
│       └── gr00t_inference.py  # Docker service management
├── examples/
│   ├── libero_example.py           # Standard SimEnv example
│   └── libero_stepped_example.py   # SteppedSimEnv example
├── scripts/
│   ├── setup-gr00t-gpu.sh          # GR00T setup + checkpoints
│   └── run_libero_stepped_example.sh  # Batch execution
└── tests/                      # Mock, fast, and integration tests
```

**Policy Implementations:**
- ✅ **GR00T** - Fully implemented Isaac-GR00T VLA policy
- ✅ **Mock** - Testing policy (returns random actions)
- ⏳ **ACT, SmolVLA** - Architecture supports extensibility, not yet implemented

## Performance Metrics

**Typical Episode (Libero-90):**
- Task: "Pick up object and place in drawer"
- Steps: ~150-200 (depending on task complexity)
- Time: 3-5 minutes (with Claude Sonnet 4.5)
- Success Rate: ~60-80% (varies by task and checkpoint)
- Video Output: Side-by-side dual camera view @ 30 FPS

## Citation

If you use this framework in your research:

```bibtex
@software{strands_robots_sim,
  title = {Strands Robots Sim: LLM-Controlled Robot Manipulation},
  author = {AWS WWSO Prototyping},
  year = {2025},
  url = {https://github.com/strands-labs/robots-sim}
}
```

## Related Projects

- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T): Foundation model for robot control
- [Libero](https://github.com/Lifelong-Robot-Learning/LIBERO): Lifelong robot learning benchmark
- [Strands SDK](https://github.com/strands-agents/sdk-python): AI agent framework
- [π0 (Pi0)](https://github.com/physical-intelligence/pi0): Vision-language-action flow model

---

**Quick Links:**
- 📖 [Full Documentation](README.md)
- 🐛 [Issues](https://gitlab.aws.dev/aws-wwso-prototyping/strands-robots-sim/-/issues)
- 🎥 [Demo Videos](rollouts/)
