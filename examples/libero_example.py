#!/usr/bin/env python3
"""
Example: Using SimEnv with Libero Environment and GR00T Policy

This example demonstrates how to use the SimEnv class to control simulated
robots in Libero environments using GR00T policies.

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
from strands_robots_sim import SimEnv, gr00t_inference


def main(max_episodes=10):
    """Simple example showing basic usage with real services."""
    from strands import Agent
    from strands_robots_sim import SimEnv, gr00t_inference
    import random
    import asyncio
    
    print("🎮 Creating simulation environment...")
    
    # Create simulation environment
    sim_env = SimEnv(
        tool_name="my_libero_sim",
        env_type="libero",
        #task_suite="libero_spatial",
        #data_config="libero_spatial"
        task_suite="libero_10",
        data_config="libero_10"
        #task_suite="libero_90",
        #data_config="libero_90"
    )
    
    # Create agent
    agent = Agent(tools=[sim_env, gr00t_inference])
    
    print("\n💡 Note: This example requires:")
    print("   1. Libero installed: pip install strands-robots[sim]")
    print("   2. GR00T inference service running on port 8000")
    print("   3. Docker with isaac-gr00t containers available")
    print("\n🚀 Starting example...\n")
    
    try:
        # Try to start GR00T service
        print("1. Starting GR00T inference service...")
        result = agent.tool.gr00t_inference(
            action="start",
            #checkpoint_path="/data/checkpoints/gr00t-libero-spatial",
            checkpoint_path="/data/checkpoints/gr00t-n1.5-libero-long-posttrain",
            port=8000,
            data_config="examples.Libero.custom_data_config:LiberoDataConfig"
        )
        print(f"   Result: {result}")
        
        # Check if service is running
        print("\n2. Checking GR00T service status...")
        status = agent.tool.gr00t_inference(action="status", port=8000)
        print(f"   Status: {status}")
        
        # Initialize sim_env to get available tasks
        print("\n2.5. Initializing simulation environment to get available tasks...")
        async def init_sim_env():
            return await sim_env.sim_env.initialize()
        
        if not asyncio.run(init_sim_env()):
            raise RuntimeError("❌ Failed to initialize simulation environment")
        
        available_tasks = sim_env.sim_env.available_tasks
        if not available_tasks:
            raise RuntimeError("❌ No available tasks found in the simulation environment")
        
        # Randomly select a task (random.choice is acceptable here - not security-sensitive)
        selected_task = random.choice(available_tasks)  # nosec B311
        print(f"   🎲 Randomly selected task: {selected_task}")
        
        # Set the task name in the environment
        sim_env.sim_env.set_task_name(selected_task)
        
        # Control simulated robot with natural language
        # NOTE: Using agent() with natural language adds ~3-5s LLM processing overhead
        # For batch experiments without this overhead, use direct tool calls:
        #   agent.tool.my_libero_sim(action="execute", instruction=selected_task,
        #                            policy_port=8000, max_episodes=max_episodes, ...)
        print(f"\n3. Running simulation task with video recording (max_episodes={max_episodes})...")
        result = agent(f"Run the task '{selected_task}' for {max_episodes} episode(s) with max_steps_per_episode=500 and record video")
        print(f"   Result: {result}")
        
        # Check final status
        print("\n4. Checking simulation status...")
        final_status = agent.tool.my_libero_sim(action="status")
        print(f"   Final status: {final_status}")
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   - Install simulation dependencies: pip install strands-robots[sim]")
        print("   - Ensure Docker is running and isaac-gr00t containers are available")
        print("   - Check that the GR00T inference service can start properly")
        
    finally:
        # Always try to cleanup
        print("\n5. Cleaning up...")
        try:
            agent.tool.gr00t_inference(action="stop", port=8000)
            print("   ✅ Cleanup completed")
        except Exception as e:
            print(f"   ⚠️ Cleanup warning: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Libero simulation with GR00T policy')
    parser.add_argument('--max-episodes', type=int, default=10,
                        help='Maximum number of episodes to run (default: 10)')
    args = parser.parse_args()

    main(max_episodes=args.max_episodes)
