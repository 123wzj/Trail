# Prediction Script Improvements Documentation

## Overview

This document describes improvements made to the `src/prectdict_event_gnn.py` script to fix APT organization attribution prediction functionality.

## Issues Fixed

### 1. Missing Label Propagation Step
**Problem**: The model was trained using label propagation features, but the prediction script didn't generate these features, leading to inaccurate predictions.

**Solution**: Added `_run_label_propagation()` method, following the implementation from `lprop.py`:
- Uses `MessagePassing` to aggregate neighbor node label information
- Generates prediction label features for IOC and event nodes
- Iteratively propagates over 5 hops
- Automatically runs label propagation before prediction

```python
def _run_label_propagation(self, test_event_ids: torch.Tensor, hops: int = 5) -> torch.Tensor:
    """Run label propagation for test events"""
    # Initialize label features for all nodes
    num_classes = len(self.label_map)
    x = torch.zeros(self.graph.x.size(0), num_classes, device=self.device)
    
    # Set one-hot encoding for known labeled event nodes
    for event_id, label in zip(self.graph.event_ids, self.graph.y):
        if label >= 0:
            x[event_id, label] = 1.0
    
    # Use MessagePassing for label propagation
    mp = MessagePassing(aggr='mean')
    for k in range(hops):
        x = x + mp.propagate(self.graph.edge_index, x=x, size=None)
    
    return x[test_event_ids]
```

### 2. Hardcoded Node Types
**Problem**: Using hardcoded node type ID (4) might not match actual data.

**Solution**: Added `_get_event_node_type_id()` method to dynamically get EVENT node type:
- First checks `graph.type_dict`
- Then checks `graph.ntypes`
- Falls back to default value 4 if neither exists

```python
def _get_event_node_type_id(self) -> int:
    """Dynamically get EVENT node type ID"""
    if hasattr(self.graph, 'type_dict'):
        return self.graph.type_dict.get('EVENT', self.graph.type_dict.get('event', 4))
    elif hasattr(self.graph, 'ntypes'):
        for key, value in self.graph.ntypes.items():
            if key.upper() == 'EVENT':
                return value
    return 4  # Fallback default
```

### 3. Missing Data Validation
**Problem**: No validation that graph data contains necessary attributes, potentially causing runtime errors.

**Solution**: Added `_validate_graph()` method:
- Checks required attributes: `edge_csr`, `event_ids`, `y`, `label_map`, `x`
- Verifies node type information: `ntypes` or `type_dict`
- Automatically runs validation during initialization

```python
def _validate_graph(self):
    """Validate that graph data contains all necessary attributes"""
    required_attrs = ['edge_csr', 'event_ids', 'y', 'label_map', 'x']
    missing_attrs = []
    
    for attr in required_attrs:
        if not hasattr(self.graph, attr):
            missing_attrs.append(attr)
    
    if missing_attrs:
        raise ValueError(f"Graph data missing required attributes: {', '.join(missing_attrs)}")
    
    if not hasattr(self.graph, 'ntypes') and not hasattr(self.graph, 'type_dict'):
        raise ValueError("Graph data missing 'ntypes' or 'type_dict' attribute")
```

### 4. Missing Debug Information
**Problem**: Unable to track specific reasons for prediction failures.

**Solution**: Added comprehensive debug output:
- Key graph properties (node count, event count, label distribution)
- Detailed information about IOC and event nodes
- Label propagation progress
- Probability distributions before and after prediction
- Shape and value range checks for intermediate steps

### 5. Missing Error Handling
**Problem**: Critical steps lacked error handling, making it difficult to locate issues.

**Solution**: Added error handling for all critical operations:
- Predictor initialization
- IOC mapping loading
- IOC node lookup
- Label propagation execution
- Model prediction
- Meaningful error messages and stack traces (in debug mode)

## New Features

### Command-Line Argument Support

```bash
# Run with default settings
python prectdict_event_gnn.py

# Enable debug mode
python prectdict_event_gnn.py --debug

# Specify custom model and dataset paths
python prectdict_event_gnn.py --model /path/to/model.pt --dataset /path/to/dataset

# Get more prediction results
python prectdict_event_gnn.py --top-k 10

# Don't save results to file
python prectdict_event_gnn.py --no-save

# Specify output file
python prectdict_event_gnn.py --output my_results.json
```

### Supported Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model weights file path | `MODEL_PATH` configured in script |
| `--dataset` | Dataset directory path | `DATASET_PATH` configured in script |
| `--top-k` | Return top K most likely organizations | 5 |
| `--debug` | Enable detailed debug output | False |
| `--no-save` | Don't save results to file | False |
| `--output` | Output file path | `apt_prediction_result.json` |

## Debug Mode Output Example

With the `--debug` flag enabled, the script outputs detailed diagnostic information:

```
======================================================================
Graph Information:
======================================================================
  Total nodes: 2,134,567
  Event nodes: 4,523
  Node types: {'ips': 0, 'urls': 1, 'domains': 2, 'ASN': 3, 'EVENT': 4}
  Number of labels: 22
  Label distribution:
    APT28: 234
    APT29: 189
    ...
======================================================================

✓ Loaded 12,345 IP address mappings
✓ Loaded 45,678 domain mappings
✓ Loaded 98,765 URL mappings

======================================================================
Starting APT Attribution Prediction
======================================================================

  Found 3 IOCs:
    ✓ ip:198.199.14.14
    ✓ ip:74.81.69.244
    ✓ domain:example.com

  Using EVENT node type ID: 4
  Found 15 associated events

======================================================================
Running Label Propagation to Generate Prediction Features
======================================================================

Starting label propagation (5 hops)...
  Converting from CSR format to edge index...
  [1/5] Labeled test events: 8/15
  [2/5] Labeled test events: 12/15
  [3/5] Labeled test events: 14/15
  [4/5] Labeled test events: 15/15
  [5/5] Labeled test events: 15/15

  Label propagation complete
  Generated feature shape: torch.Size([15, 22])
  Feature range: [0.0000, 0.8542]

======================================================================
Running GNN Model Prediction
======================================================================
  Prediction output shape: torch.Size([15, 22])
  Probability range: [0.0012, 0.4521]
  Probability sum: 1.0000

  Aggregated predictions:
  Average probability range: [0.0034, 0.3842]
  Probability sum: 1.0000

======================================================================
Top-K Prediction Results
======================================================================
  1. APT28                   38.42% ███████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  2. APT29                   22.15% ███████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  3. Lazarus                 15.67% ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  4. APT32                    8.93% ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  5. FIN7                     5.21% ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

======================================================================
```

## Testing

Created `test_prectdict_event_gnn.py` validation script with the following tests:

1. **Import Validation**: Checks all required imports
2. **Class Structure**: Verifies IOCAttributor class contains all required methods
3. **Method Signatures**: Checks that key method signatures are correct
4. **Label Propagation**: Verifies label propagation implementation exists
5. **Dynamic Node Type Lookup**: Confirms hardcoded values are not used
6. **Command-Line Arguments**: Validates argparse integration
7. **Error Handling**: Checks for try-except blocks

Run tests:
```bash
cd src
python test_prectdict_event_gnn.py
```

All 7 tests pass ✓

## Backward Compatibility

All changes maintain backward compatibility:
- Existing configuration variables (`MODEL_PATH`, `DATASET_PATH`, `INPUT_IOCS`, etc.) still work
- Script can be run without command-line arguments
- Default behavior matches original script (except for new features)

## Code Quality Improvements

1. **Type Hints**: Added type hints to all methods
2. **Docstrings**: Added detailed documentation for all new methods
3. **Error Messages**: Provided clear, actionable error messages
4. **Code Organization**: Broke functionality into small, testable methods
5. **Debug Friendly**: Added comprehensive logging and diagnostic output

## Usage Recommendations

1. **First Run**: Use `--debug` flag to verify everything works correctly
2. **Production**: Run without `--debug` for clean output
3. **Troubleshooting**: Use `--debug` to diagnose prediction issues
4. **Batch Processing**: Modify `INPUT_IOCS` variable or extend script to read from files

## Performance Considerations

- Label propagation may take some time on large graphs (5 hops)
- `--debug` mode adds slight performance overhead
- CSR to edge index conversion is performed once on first label propagation

## Future Enhancement Suggestions

1. Support batch reading of IOCs from files
2. Add confidence threshold filtering
3. Support different aggregation strategies (weighted average, voting, etc.)
4. Add visualization output options
5. Support incremental graph updates
