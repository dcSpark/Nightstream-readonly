#!/bin/bash

echo "🚀 Running FRI Implementation Comparison..."
echo "=============================================="
echo "This will test BOTH CUSTOM FRI and p3-fri implementations"
echo "in a single execution and compare their performance."
echo

# Run with both features enabled
cargo run --package neo-main --features "custom-fri,p3-fri"