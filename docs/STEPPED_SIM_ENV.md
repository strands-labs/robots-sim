# SteppedSimEnv: Agent-Driven Planning for Robot Simulation

## Overview

`SteppedSimEnv` is a specialized simulation environment control tool that enables **agent-driven hierarchical planning**. Unlike the standard `SimEnv` which executes complete episodes, `SteppedSimEnv` allows an AI agent to act as a "brain" that:

1. **Plans in steps**: Execute a limited number of steps (e.g., 10 steps)
2. **Observes outcomes**: Receive camera images and state after each execution
3. **Adapts dynamically**: Change instructions based on observations
4. **Decomposes tasks**: Transform complex instructions into simpler sub-tasks

This approach is ideal for scenarios where tasks require adaptive planning and reasoning.

## Key Features

- ✅ **Stepped Execution**: Execute N steps per tool call (configurable)
- ✅ **Visual Feedback**: Return camera images (front/wrist) after each call
- ✅ **Rich State Info**: Episode progress, steps, rewards, success rate
- ✅ **Stateful**: Maintains state across multiple tool calls
- ✅ **Flexible Actions**: execute_steps, reset_episode, get_state
- ✅ **Agent-Friendly**: Designed for LLM-based agent frameworks

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AI Agent (Brain)                      │
│  • Analyzes camera images                                    │
│  • Plans next instruction                                    │
│  • Decomposes complex tasks                                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ Tool Call: execute_steps
                        │ instruction="move to object"
                        │ num_steps=10
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                     SteppedSimEnv Tool                       │
│  • Executes 10 simulation steps                             │
│  • Captures camera images                                    │
│  • Records state (steps, reward, done)                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ Returns:
                        │ • Camera images (base64 encoded)
                        │ • State text (episode, steps, reward)
                        │ • Episode status (running/done)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                        AI Agent (Brain)                      │
│  • Observes: "gripper moved closer to object"               │
│  • Decides: "continue with 'grasp object' instruction"      │
└─────────────────────────────────────────────────────────────┘
```

## Comparison: SimEnv vs SteppedSimEnv

| Feature | SimEnv | SteppedSimEnv |
|---------|--------|---------------|
| **Execution Mode** | Complete episodes | Stepped (N steps at a time) |
| **Agent Control** | Start/Stop only | Full control per step batch |
| **Observations** | None (runs to completion) | Camera images + state after each call |
| **Instruction Changes** | Fixed per episode | Can change between calls |
| **Use Case** | Autonomous execution | Agent-driven planning |
| **Typical Steps** | 500 (full episode) | 10-50 (per call) |

## Installation

```bash
# Install the package
pip install -e .

# Ensure dependencies are available
pip install numpy pillow
```

## Quick Start

### 1. Create SteppedSimEnv Instance

```python
from strands_robots_sim import SteppedSimEnv

# Initialize with steps_per_call configuration
stepped_env = SteppedSimEnv(
    tool_name="libero_stepped",
    env_type="libero",
    task_suite="libero_spatial",
    steps_per_call=10,  # Execute 10 steps per call
    max_steps_per_episode=500,
)
```

### 2. Use as Agent Tool

The tool supports three actions:

#### Action 1: `reset_episode` - Start New Episode

```python
tool_use = {
    "toolUseId": "use_001",
    "input": {
        "action": "reset_episode",
        "task_name": None,  # Random task, or specify task name
    }
}
```

**Returns:**
- State text (episode info, steps, reward)
- Initial camera images (front/wrist views)

#### Action 2: `execute_steps` - Run N Steps

```python
tool_use = {
    "toolUseId": "use_002",
    "input": {
        "action": "execute_steps",
        "instruction": "move gripper to the target object",
        "policy_port": 8000,
        "policy_host": "localhost",
        "policy_provider": "groot",
        "num_steps": 10,  # Optional, defaults to steps_per_call
    }
}
```

**Returns:**
- State text with updated progress
- Camera images after execution
- Episode status (running/done)

#### Action 3: `get_state` - Query Current State

```python
tool_use = {
    "toolUseId": "use_003",
    "input": {
        "action": "get_state",
    }
}
```

**Returns:**
- Current state text
- Last observed camera images

## Agent Workflow Example

Here's how an AI agent might use SteppedSimEnv:

```python
# Agent receives task: "Pick up the red block and place it in the drawer"

# Step 1: Agent decomposes task
subtasks = [
    "locate red block",
    "move gripper to red block",
    "grasp red block",
    "lift red block",
    "locate drawer",
    "move to drawer",
    "place block in drawer",
]

# Step 2: Agent executes each subtask iteratively
for subtask in subtasks:
    # Execute steps with current instruction
    result = execute_steps(
        instruction=subtask,
        num_steps=10
    )
    
    # Agent analyzes returned images and state
    if "episode_done" in result:
        break
    
    # Agent can adapt based on observations
    if "gripper_empty" in observation:
        # Adjust instruction if needed
        subtask = "retry grasping object"
```

## State Information

Each tool call returns detailed state information:

```markdown
## Simulation State

**Environment**: libero (libero_spatial)
**Task**: LIVING_ROOM_SCENE5_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket

**Current Instruction**: move gripper to the target object

**Episode Progress**:
- Episode: 1
- Episode Steps: 45 / 500
- Episode Reward: 0.125
- Status: running

**Overall Progress**:
- Total Steps: 45
- Total Reward: 0.125
- Success Count: 0
- Success Rate: 0.0%
```

## Camera Images

The tool returns base64-encoded JPEG images:

- **Front Camera**: `agentview_image`, `front_camera`, `video.webcam`
- **Wrist Camera**: `robot0_eye_in_hand_image`, `wrist_camera`

Images are automatically extracted from observations and encoded for agent consumption.

## Advanced Usage

### Custom Steps Per Call

```python
stepped_env = SteppedSimEnv(
    tool_name="libero_stepped",
    env_type="libero",
    task_suite="libero_spatial",
    steps_per_call=20,  # Execute 20 steps per call
)
```

### Variable Steps Per Execution

```python
# Override default in tool call
tool_use = {
    "input": {
        "action": "execute_steps",
        "instruction": "careful grasping motion",
        "num_steps": 5,  # Slower, more careful execution
    }
}
```

### Task-Specific Episodes

```python
tool_use = {
    "input": {
        "action": "reset_episode",
        "task_name": "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
    }
}
```

## Complete Example

See `examples/stepped_sim_example.py` for a full example:

```bash
# Run the example (requires GR00T server)
python examples/stepped_sim_example.py
```

## Integration with Agent Frameworks

### Langchain Integration

```python
from langchain.agents import Tool

stepped_tool = Tool(
    name="simulation_environment",
    func=stepped_env.stream,
    description=stepped_env.tool_spec["description"],
)
```

### Custom Agent Loop

```python
async def agent_planning_loop(task_description):
    """Agent-driven planning loop."""
    
    # Reset episode
    await execute_tool_action("reset_episode")
    
    # Agent generates initial instruction from task
    instruction = llm.generate_instruction(task_description)
    
    while not episode_done:
        # Execute steps
        result = await execute_tool_action(
            "execute_steps",
            instruction=instruction
        )
        
        # Agent analyzes images and state
        analysis = llm.analyze_observation(
            images=result["images"],
            state=result["state_text"]
        )
        
        # Agent decides next instruction
        instruction = llm.generate_next_instruction(analysis)
        
        episode_done = result.get("episode_done", False)
```

## Benefits Over Standard SimEnv

1. **Visual Feedback**: Agent sees what's happening in the environment
2. **Adaptive Planning**: Change strategy based on observations
3. **Error Recovery**: Detect failures early and adapt
4. **Task Decomposition**: Break complex tasks into manageable pieces
5. **Learning**: Agent can learn which strategies work better
6. **Debugging**: Easier to understand and debug agent behavior

## Limitations

- **Increased Latency**: More tool calls vs single episode execution
- **State Management**: Agent must track episode state across calls
- **Complexity**: Requires more sophisticated agent logic

## Best Practices

1. **Start Simple**: Begin with 10-20 steps per call
2. **Monitor Progress**: Check episode_steps and reward regularly
3. **Handle Episode Done**: Always check if episode completed
4. **Clear Instructions**: Provide specific, actionable instructions
5. **Error Handling**: Implement retry logic for failed executions

## Troubleshooting

### Issue: "No active episode"

**Solution**: Call `reset_episode` before `execute_steps`

### Issue: No camera images returned

**Solution**: Check that environment has camera observations:
- Libero: `agentview_image`, `robot0_eye_in_hand_image`
- Check observation keys with `get_state`

### Issue: Episode keeps running

**Solution**: 
- Check `max_steps_per_episode` configuration
- Monitor `episode_steps` in state
- Episode auto-completes at max_steps or success

## API Reference

### Constructor Parameters

- `tool_name` (str): Name for the tool
- `env_type` (str): Environment type ("libero", etc.)
- `task_suite` (str): Task suite name
- `steps_per_call` (int): Default steps per execution (default: 10)
- `max_steps_per_episode` (int): Maximum episode steps (default: 500)
- `action_horizon` (int): Actions per inference (default: 8)
- `data_config` (Any): Policy data configuration

### Tool Actions

| Action | Required Parameters | Returns |
|--------|-------------------|---------|
| `reset_episode` | task_name (optional) | State + Images |
| `execute_steps` | instruction, policy_port | State + Images |
| `get_state` | None | State + Images |

### State Properties

Access via `stepped_env._state`:

- `status`: Current execution status
- `current_instruction`: Active instruction
- `current_episode`: Episode number
- `episode_steps`: Steps in current episode
- `total_steps`: Total steps across all episodes
- `episode_reward`: Cumulative reward for episode
- `success_count`: Number of successful episodes

## Future Enhancements

- [ ] Support for custom reward functions
- [ ] Action logging and replay
- [ ] Multi-modal observations (audio, force sensors)
- [ ] Trajectory visualization
- [ ] Agent learning from experience

## Contributing

Contributions are welcome! Please see the main README for contribution guidelines.

## License

See LICENSE file in the project root.
