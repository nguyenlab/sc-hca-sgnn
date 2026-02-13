#!/bin/bash
# Grid Search Runner - runs grid search for top models
# Usage: bash run_grid_search.sh [quick|full] [graph|function|all]

MODE=${1:-full}
LEVEL=${2:-graph}

echo "======================================================================"
echo "Running Grid Search for Smart Contract Vulnerability Detection"
echo "Mode: $MODE, Level: $LEVEL"
echo "======================================================================"

if [ "$MODE" = "quick" ]; then
    EPOCHS=30
    PATIENCE=20
    FLAG="--quick"
else
    EPOCHS=50
    PATIENCE=30
    FLAG=""
fi

# Top performing graph-level models
GRAPH_MODELS=("scvhunter" "mlagnn" "dr-gcn" "gat")

# Top performing function-level models
FUNC_MODELS=("func-hca-mean" "func-mlagnn" "func-scvhunter" "func-dr-gcn" "func-gat-mean")

run_graph_search() {
    echo ""
    echo "===== Running GRAPH-LEVEL Grid Search ====="
    mkdir -p results/grid_search/contract_level
    for MODEL in "${GRAPH_MODELS[@]}"; do
        echo ""
        echo "Starting grid search for: $MODEL"
        python scripts/grid_search.py \
            --model $MODEL \
            --level graph \
            --epochs $EPOCHS \
            --patience $PATIENCE \
            $FLAG \
            --output results/grid_search/contract_level/grid_search_${MODEL}.json
        
        if [ $? -eq 0 ]; then
            echo "✓ Completed: $MODEL"
        else
            echo "✗ Failed: $MODEL"
        fi
    done
}

run_function_search() {
    echo ""
    echo "===== Running FUNCTION-LEVEL Grid Search ====="
    mkdir -p results/grid_search/function_level
    for MODEL in "${FUNC_MODELS[@]}"; do
        echo ""
        echo "Starting grid search for: $MODEL"
        python scripts/grid_search.py \
            --model $MODEL \
            --level function \
            --epochs $EPOCHS \
            --patience $PATIENCE \
            $FLAG \
            --output results/grid_search/function_level/grid_search_${MODEL}.json
        
        if [ $? -eq 0 ]; then
            echo "✓ Completed: $MODEL"
        else
            echo "✗ Failed: $MODEL"
        fi
    done
}

# Run based on level parameter
case $LEVEL in
    graph)
        run_graph_search
        ;;
    function)
        run_function_search
        ;;
    all)
        run_graph_search
        run_function_search
        ;;
    *)
        echo "Unknown level: $LEVEL"
        echo "Usage: bash run_grid_search.sh [quick|full] [graph|function|all]"
        exit 1
        ;;
esac

echo ""
echo "======================================================================"
echo "Grid Search Complete!"
echo "======================================================================"
echo ""
echo "Analyzing results..."
echo ""
echo "Contract-level results:"
python scripts/analyze_grid_search.py results/grid_search/contract_level/*.json 2>/dev/null || echo "  No contract-level results yet"
echo ""
echo "Function-level results:"
python scripts/analyze_grid_search.py results/grid_search/function_level/*.json 2>/dev/null || echo "  No function-level results yet"
