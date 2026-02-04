#!/bin/bash

# Script to run libero example experiments with single episodes
# Usage: ./run_exp_libero_example.sh [number_of_runs]

# Default number of runs if not specified
DEFAULT_RUNS=10

# Get number of runs from command line argument or use default
if [ -z "$1" ]; then
    NUM_RUNS=$DEFAULT_RUNS
    echo "No argument provided. Using default: $NUM_RUNS run(s)"
else
    # Check if argument is a positive integer
    if ! [[ "$1" =~ ^[0-9]+$ ]] || [ "$1" -le 0 ]; then
        echo "Error: Argument must be a positive integer"
        echo "Usage: $0 [number_of_runs]"
        exit 1
    fi
    NUM_RUNS=$1
fi

# Create exps directory if it doesn't exist
mkdir -p exps

# Generate experiment ID with timestamp
EXP_ID=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="exps/libero_example_${EXP_ID}.log"

echo "=========================================" | tee -a "$LOG_FILE"
echo "Running libero_example.py experiment" | tee -a "$LOG_FILE"
echo "Experiment ID: $EXP_ID" | tee -a "$LOG_FILE"
echo "Number of runs: $NUM_RUNS" | tee -a "$LOG_FILE"
echo "Max episodes per run: 1" | tee -a "$LOG_FILE"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Initialize counters
SUCCESSFUL_RUNS=0
FAILED_RUNS=0

# Array to store individual run results
declare -a RUN_RESULTS

# Run the example NUM_RUNS times
for ((i=1; i<=NUM_RUNS; i++)); do
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "Run $i of $NUM_RUNS" | tee -a "$LOG_FILE"
    RUN_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    RUN_START_EPOCH=$(date +%s)
    echo "Start time: $RUN_START_TIME" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"

    # Run the Python script with max_episodes=1 and capture output
    OUTPUT=$(python examples/libero_example.py --max-episodes 1 2>&1)
    EXIT_CODE=$?

    RUN_END_EPOCH=$(date +%s)
    RUN_DURATION=$((RUN_END_EPOCH - RUN_START_EPOCH))

    # Extract task name from output
    TASK_NAME=$(echo "$OUTPUT" | grep -oP "Randomly selected task: \K.*" | head -1)
    if [ -z "$TASK_NAME" ]; then
        TASK_NAME="unknown"
    fi

    # Check if episode was successful (reward = 1.00)
    EPISODE_SUCCESS="false"
    if echo "$OUTPUT" | grep -q "✅ Success.*reward: 1.00"; then
        EPISODE_SUCCESS="true"
        ((SUCCESSFUL_RUNS++))
        echo "✅ Run $i SUCCESSFUL (reward = 1.00)" | tee -a "$LOG_FILE"
    elif echo "$OUTPUT" | grep -q "❌ Failed"; then
        EPISODE_SUCCESS="false"
        ((FAILED_RUNS++))
        echo "❌ Run $i FAILED (reward < 1.00)" | tee -a "$LOG_FILE"
    else
        # If no clear success/failure marker, check exit code
        if [ $EXIT_CODE -eq 0 ]; then
            # Check output for any success indicators
            if echo "$OUTPUT" | grep -q "Episode 1 completed"; then
                EPISODE_SUCCESS="false"
                ((FAILED_RUNS++))
                echo "❌ Run $i FAILED (completed but reward < 1.00)" | tee -a "$LOG_FILE"
            else
                EPISODE_SUCCESS="error"
                ((FAILED_RUNS++))
                echo "❌ Run $i ERROR (exit code: $EXIT_CODE)" | tee -a "$LOG_FILE"
            fi
        else
            EPISODE_SUCCESS="error"
            ((FAILED_RUNS++))
            echo "❌ Run $i ERROR (exit code: $EXIT_CODE)" | tee -a "$LOG_FILE"
        fi
    fi

    # Store run result
    RUN_RESULTS[$i]="Run $i: task=$TASK_NAME, success=$EPISODE_SUCCESS, duration=${RUN_DURATION}s, exit_code=$EXIT_CODE"

    echo "Task: $TASK_NAME" | tee -a "$LOG_FILE"
    echo "Duration: ${RUN_DURATION}s" | tee -a "$LOG_FILE"
    echo "Exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # Log detailed output
    echo "--- Detailed Output ---" >> "$LOG_FILE"
    echo "$OUTPUT" >> "$LOG_FILE"
    echo "--- End Output ---" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"

    # Add a small delay between runs if not the last run
    if [ $i -lt $NUM_RUNS ]; then
        echo "Waiting 2 seconds before next run..." | tee -a "$LOG_FILE"
        sleep 2
        echo "" | tee -a "$LOG_FILE"
    fi
done

# Calculate statistics
SUCCESS_RATE=0
if [ $NUM_RUNS -gt 0 ]; then
    SUCCESS_RATE=$(awk "BEGIN {printf \"%.1f\", ($SUCCESSFUL_RUNS / $NUM_RUNS) * 100}")
fi

# Print summary
echo "=========================================" | tee -a "$LOG_FILE"
echo "EXPERIMENT SUMMARY" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
echo "Experiment ID: $EXP_ID" | tee -a "$LOG_FILE"
echo "Total runs: $NUM_RUNS" | tee -a "$LOG_FILE"
echo "Successful (reward=1.0): $SUCCESSFUL_RUNS" | tee -a "$LOG_FILE"
echo "Failed (reward<1.0): $FAILED_RUNS" | tee -a "$LOG_FILE"
echo "Success Rate: ${SUCCESS_RATE}%" | tee -a "$LOG_FILE"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Log individual run results
echo "Individual Run Results:" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"
for ((i=1; i<=NUM_RUNS; i++)); do
    echo "${RUN_RESULTS[$i]}" | tee -a "$LOG_FILE"
done
echo "=========================================" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "📝 Full log saved to: $LOG_FILE" | tee -a "$LOG_FILE"

# Exit with error code if any runs failed
if [ $FAILED_RUNS -gt 0 ]; then
    exit 1
fi

exit 0
