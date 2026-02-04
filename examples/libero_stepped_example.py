#!/usr/bin/env python3
"""
Example: Using SteppedSimEnv with Libero Environment and GR00T Policy

This example demonstrates how to use the SteppedSimEnv class for agent-driven
planning in Libero environments. Unlike SimEnv which runs complete episodes,
SteppedSimEnv executes in batches (e.g., 10 steps), returns camera images and
state, allowing an agent to observe progress and adapt instructions.

The agent can attempt multiple episodes (via --max-episodes) to complete a task,
resetting and trying different strategies if earlier attempts fail.

Usage:
    python examples/libero_stepped_example.py --max-episodes 1  # Single attempt
    python examples/libero_stepped_example.py --max-episodes 3  # Up to 3 attempts

Requirements:
- Libero installed: pip install strands-robots[sim]
- GR00T inference service running on port 8000
- Docker with isaac-gr00t containers available

For testing without these requirements, see:
- tests/test_mock.py - Mock environment tests
- tests/test_libero_fast.py - Fast Libero tests with zero actions
"""

import random
import asyncio
import argparse
from strands import Agent
from strands_robots_sim import SteppedSimEnv, gr00t_inference


def main(max_episodes=1):
    """Example showing stepped execution for agent-driven planning.

    Args:
        max_episodes: Maximum number of episodes to attempt (default: 1).
                     Agent can reset and retry if task fails, up to this limit.
    """
    print("🎮 Creating stepped simulation environment...")
    
    # Create stepped simulation environment
    # Key difference: steps_per_call defines how many steps to execute per tool call
    stepped_sim_env = SteppedSimEnv(
        tool_name="my_libero_stepped_sim",
        env_type="libero",
        task_suite="libero_10",
        data_config="libero_10",
        steps_per_call=10,  # Execute 10 steps per call (can be adjusted)
        max_steps_per_episode=500,
    )
    
    # Create agent with both tools
    # Use Claude Sonnet 4.5 for excellent balance of speed and reasoning
    agent = Agent(
        model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",  # Bedrock inference profile
        tools=[stepped_sim_env, gr00t_inference],
    )
    
    print("\n💡 Note: This example requires:")
    print("   1. Libero installed: pip install strands-robots[sim]")
    print("   2. GR00T inference service running on port 8000")
    print("   3. Docker with isaac-gr00t containers available")
    print("\n🚀 Starting stepped execution example...\n")
    
    try:
        # 1. Start GR00T inference service
        print("1. Starting GR00T inference service...")
        result = agent.tool.gr00t_inference(
            action="start",
            #checkpoint_path="/data/checkpoints/gr00t-n1.5-libero-90-posttrain",
            checkpoint_path="/data/checkpoints/gr00t-n1.5-libero-long-posttrain",
            port=8000,
            data_config="examples.Libero.custom_data_config:LiberoDataConfig"
        )
        print(f"   Result: {result}")
        
        # 2. Check if service is running
        print("\n2. Checking GR00T service status...")
        status = agent.tool.gr00t_inference(action="status", port=8000)
        print(f"   Status: {status}")
        
        # 3. Initialize sim_env to get available tasks
        print("\n3. Initializing simulation environment to get available tasks...")
        async def init_sim_env():
            return await stepped_sim_env.sim_env.initialize()
        
        if not asyncio.run(init_sim_env()):
            raise RuntimeError("❌ Failed to initialize simulation environment")
        
        available_tasks = stepped_sim_env.sim_env.available_tasks
        if not available_tasks:
            raise RuntimeError("❌ No available tasks found in the simulation environment")
        
        # 4. Randomly select a task (random.choice is acceptable here - not security-sensitive)
        selected_task = random.choice(available_tasks)  # nosec B311
        print(f"   🎲 Randomly selected task: {selected_task}")
        
        # Extract a simpler description from the task name for instructions
        # Example: "KITCHEN_SCENE1_open_the_top_drawer" -> "open the top drawer"
        task_parts = selected_task.split('_')
        if len(task_parts) > 2:
            complex_instruction = ' '.join(task_parts[2:]).replace('_', ' ')
        else:
            complex_instruction = selected_task
        
        print(f"   📝 Instruction: {complex_instruction}")
        
        # 5. Let the ReAct agent control the execution
        print(f"\n4. Letting the agent execute the task with stepped control (max_episodes={max_episodes})...")
        print("   The agent will use SteppedSimEnv to:")
        print("   - Reset the episode")
        print("   - Execute steps iteratively (10 steps at a time)")
        print("   - Observe camera images and state after each batch")
        print("   - Decide when to continue or stop")
        print("   - Adapt instructions based on observations")
        print(f"   - Try up to {max_episodes} episode(s) to achieve reward = 1.0\n")
        
        # Use agent's natural language interface
        # The agent will decide how many iterations to run based on the task
        prompt = f"""
Task: {complex_instruction}
Max Episodes: {max_episodes}

You are a robot task planner. Your goal is to complete the complex task by breaking it down into simpler subtasks and executing them step-by-step using the SteppedSimEnv tool.

You have a maximum of {max_episodes} episode(s) to complete this task.

STEP 1: DECOMPOSE THE TASK
First, analyze the complex task "{complex_instruction}" and decompose it into a sequence of simpler subtasks. 
For example, if the task is "pick up the block and place it in the drawer", you might decompose it into:
1. Locate and move gripper toward the block
2. Grasp the block
3. Lift the block
4. Move toward the drawer
5. Place the block in the drawer

Think about what subtasks are needed for "{complex_instruction}" and create a similar list.

STEP 2: EXECUTE THE PLAN
Once you have your subtask list, execute it using the my_libero_stepped_sim tool:

1. First, reset the episode:
   - action="reset_episode"
   - task_name="{selected_task}"

2. Then, for each subtask in your decomposed plan:
   a. Call execute_steps with the current subtask as the instruction:
      - action="execute_steps"
      - instruction="[current subtask]"
      - policy_port=8000
      - num_steps=10
   
   b. After each execute_steps call:
      - Observe the camera images and state in the response
      - Check if the episode is done (status: episode_done)
      - Check the reward and episode steps
      - If the subtask seems stuck or failing, you can retry with a rephrased instruction or move to the next subtask
   
   c. Continue to the next subtask in your plan

3. Continue executing subtasks until the episode completes or reward reaches 1.0:
   - Continue executing steps while the episode is not done (episode_done = False)
   - Try to maximize the reward by executing appropriate subtask instructions
   - If the episode status shows "episode_done":
     * Check the final reward
     * If reward = 1.0: SUCCESS! Stop - task completed.
     * If reward < 1.0 and you have remaining episodes (current episode < {max_episodes}):
       - The task failed in this episode
       - Reset and try again: call action="reset_episode" with the same task_name
       - Try a different approach or strategy in the next episode
     * If reward < 1.0 and no episodes remaining: FAILED. Stop and report.

4. Finally, use action="get_state" to check the final status and success rate.

IMPORTANT NOTES:
- Each execute_steps call only runs 10 steps, so each subtask may require multiple calls
- Observe the state feedback to know when to move to the next subtask
- The reward should generally increase as you make progress
- Be adaptive - if a subtask isn't working, try rephrasing it or adjust your approach
- You have a maximum of {max_episodes} episode(s) to achieve reward = 1.0
- Track which episode you're on and only reset if you have remaining attempts
- **CRITICAL: Stop when reward = 1.0 OR when you've used all {max_episodes} episode(s).**

Begin by decomposing the task, then execute your plan!
"""
        
        result = agent(prompt)
        print(f"\n   Agent execution completed")
        print(f"   Result: {result}")
        
        print("\n✅ Stepped execution example completed!")
        print("\n📊 Key Differences from SimEnv:")
        print("   • Stepped execution (10 steps at a time)")
        print("   • Agent receives camera images after each batch")
        print("   • Agent can adapt instructions based on observations")
        print("   • Enables hierarchical planning and task decomposition")
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   - Install simulation dependencies: pip install strands-robots[sim]")
        print("   - Ensure Docker is running and isaac-gr00t containers are available")
        print("   - Check that the GR00T inference service can start properly")
        import traceback
        traceback.print_exc()
        
    finally:
        # Always try to cleanup
        print("\n7. Cleaning up...")
        try:
            # Stop GR00T inference service
            agent.tool.gr00t_inference(action="stop", port=8000)
            
            # Cleanup stepped sim env
            asyncio.run(stepped_sim_env.cleanup())
            
            print("   ✅ Cleanup completed")
        except Exception as e:
            print(f"   ⚠️ Cleanup warning: {e}")


def demonstrate_agent_planning():
    """
    Advanced example showing how an agent would use SteppedSimEnv for
    hierarchical task decomposition and adaptive planning.
    
    This is a conceptual example showing the pattern - in practice,
    you would use an LLM-based agent to generate instructions dynamically.
    """
    print("\n" + "="*60)
    print("🧠 AGENT-DRIVEN PLANNING DEMONSTRATION")
    print("="*60)
    print("\nThis demonstrates how an AI agent would use SteppedSimEnv:")
    print("\n1. Complex Task: 'Pick up red block and place in drawer'")
    print("\n2. Agent Decomposes into subtasks:")
    print("   - locate red block")
    print("   - move gripper to red block")
    print("   - grasp red block")
    print("   - lift red block")
    print("   - locate drawer")
    print("   - move to drawer")
    print("   - place block in drawer")
    print("\n3. Agent executes each subtask with SteppedSimEnv:")
    print("   For each subtask:")
    print("     a. Execute 10 steps with current instruction")
    print("     b. Receive camera images + state")
    print("     c. Analyze progress")
    print("     d. Decide: continue / change instruction / retry")
    print("\n4. Benefits:")
    print("   ✓ Visual feedback enables better decision-making")
    print("   ✓ Can recover from errors by adapting strategy")
    print("   ✓ Natural fit for hierarchical planning")
    print("   ✓ Interpretable execution trace")
    print("\n" + "="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Libero stepped simulation with agent-driven control')
    parser.add_argument('--max-episodes', type=int, default=1,
                        help='Maximum number of episodes to attempt (default: 1)')
    args = parser.parse_args()

    # Run main example
    main(max_episodes=args.max_episodes)

    # Show conceptual demonstration (commented out by default)
    # print("\n")
    # demonstrate_agent_planning()

    print("\n💡 Next Steps:")
    print("   - Try with different task_suites (libero_spatial, libero_10)")
    print("   - Adjust steps_per_call for finer/coarser control")
    print("   - Integrate with LLM-based agent for dynamic planning")
    print("   - Analyze returned camera images for better decisions")
