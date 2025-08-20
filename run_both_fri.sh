#!/bin/bash

# Function to run with specific features and capture output
run_with_features() {
    local features="$1"
    local impl_name="$2"
    local output_var="$3"
    local log_var="$4"
   
   
    # Capture ALL output to files
    local output_file="/tmp/fri_bench_${impl_name}_$$.txt"
   
    # Run from workspace root, targeting neo-main - capture ALL output
    if [ -n "$features" ]; then
        cargo run --package neo-main --no-default-features --features "$features" > "$output_file" 2>&1
    else
        cargo run --package neo-main > "$output_file" 2>&1
    fi
    local exit_code=$?
   
    # Store the complete log output
    eval "$log_var=\"$(cat "$output_file")\""
   
    # Extract and store performance data
    if [ $exit_code -eq 0 ]; then
        local perf_data=$(grep -A 10 "🏁 FINAL PERFORMANCE SUMMARY" "$output_file" | tail -n +3 | head -n 8)
        eval "$output_var=\"$perf_data\""
    else
        eval "$output_var=\"FAILED - Exit code: $exit_code\""
    fi
   
    # Clean up temp files
    rm -f "$output_file"
   
    return $exit_code
}

full_output=""
full_output+="==========================================\n"
full_output+="BENCHMARKING: Custom FRI vs p3-fri (Plonky3)\n"
full_output+="==========================================\n"
full_output+="This script compares two different FRI implementations:\n"
full_output+="1. Custom FRI: Traditional implementation optimized for Neo-Lattice\n"
full_output+="2. p3-fri: Plonky3-based implementation providing alternative backend\n"
full_output+="\n"

# Run both implementations silently
run_with_features "neo-sumcheck/custom-fri" "Custom FRI" "custom_perf_data" "custom_logs"
custom_result=$?
run_with_features "neo-sumcheck/p3-fri" "p3-fri (Plonky3)" "p3_perf_data" "p3_logs"
p3_result=$?

full_output+="\n"
full_output+="\n"
full_output+="==========================================\n"
full_output+="🏁 FINAL IMPLEMENTATION COMPARISON\n"
full_output+="==========================================\n"
full_output+="📊 RESULTS SUMMARY:\n"
full_output+=" • Custom FRI Implementation: $([ $custom_result -eq 0 ] && echo "✅ SUCCESS" || echo "❌ FAILED")\n"
full_output+=" • p3-fri (Plonky3) Implementation: $([ $p3_result -eq 0 ] && echo "✅ SUCCESS" || echo "❌ FAILED")\n"
full_output+="\n"
full_output+="🏁 CUSTOM FRI PERFORMANCE SUMMARY\n"
full_output+="==========================================\n"
if [ $custom_result -eq 0 ]; then
    full_output+="$custom_perf_data\n"
else
    full_output+="FAILED - Exit code: $custom_result\n"
fi
full_output+="==========================================\n"
full_output+="\n"
full_output+="🏁 P3-FRI (PLONKY3) PERFORMANCE SUMMARY\n"
full_output+="==========================================\n"
if [ $p3_result -eq 0 ]; then
    full_output+="$p3_perf_data\n"
else
    full_output+="FAILED - Exit code: $p3_result\n"
fi
full_output+="==========================================\n"
full_output+="\n"
full_output+="🎯 PERFORMANCE COMPARISON:\n"
full_output+=" Compare the 'Total time' values above to see which is faster.\n"
full_output+="==========================================\n"
full_output+="\n"
full_output+="==========================================\n"
full_output+="📋 DETAILED EXECUTION LOGS\n"
full_output+="==========================================\n"
full_output+="\n"
full_output+="🔍 CUSTOM FRI DETAILED LOGS:\n"
full_output+="==========================================\n"
full_output+="$custom_logs\n"
full_output+="==========================================\n"
full_output+="\n"
full_output+="🔍 P3-FRI (PLONKY3) DETAILED LOGS:\n"
full_output+="==========================================\n"
full_output+="$p3_logs\n"
full_output+="==========================================\n"
full_output+="\n"

if [ $custom_result -eq 0 ] && [ $p3_result -eq 0 ]; then
    full_output+="🎉 EXCELLENT! Both FRI implementations succeeded!\n"
    full_output+="📈 Compare the performance summaries above to choose your preferred backend.\n"
    full_output+="💡 TIP: Look for the implementation with lower 'Total time' and 'variance' values.\n"
    echo -e "$full_output"
    exit 0
elif [ $custom_result -eq 0 ] || [ $p3_result -eq 0 ]; then
    full_output+="⚠️ PARTIAL SUCCESS: One FRI implementation succeeded, one failed.\n"
    full_output+="🔍 Check the detailed logs above to debug the failing implementation.\n"
    echo -e "$full_output"
    exit 1
else
    full_output+="❌ BOTH IMPLEMENTATIONS FAILED\n"
    full_output+="🚨 Check the detailed logs above - there may be a fundamental issue.\n"
    echo -e "$full_output"
    exit 2
fi