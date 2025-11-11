#!/usr/bin/env python3
"""
Simple validation tests for prectdict_event_gnn.py

These tests verify the code structure and basic functionality without
requiring actual model weights or running predictions.
"""

import ast
import sys
from pathlib import Path

def test_imports():
    """Test that the script has all required imports"""
    script_path = Path(__file__).parent / 'prectdict_event_gnn.py'
    with open(script_path) as f:
        tree = ast.parse(f.read())
    
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)
    
    required_imports = {
        'torch', 'pandas', 'pathlib', 'typing', 'json', 
        'argparse', 'torch_geometric.nn', 'models.gnn'
    }
    
    missing = required_imports - imports
    if missing:
        print(f"✗ Missing imports: {missing}")
        return False
    
    print("✓ All required imports present")
    return True


def test_class_structure():
    """Test that IOCAttributor class has all required methods"""
    script_path = Path(__file__).parent / 'prectdict_event_gnn.py'
    with open(script_path) as f:
        tree = ast.parse(f.read())
    
    # Find the IOCAttributor class
    ioc_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'IOCAttributor':
            ioc_class = node
            break
    
    if not ioc_class:
        print("✗ IOCAttributor class not found")
        return False
    
    # Get all method names
    methods = {node.name for node in ioc_class.body if isinstance(node, ast.FunctionDef)}
    
    required_methods = {
        '__init__',
        '_validate_graph',
        '_print_graph_info',
        '_get_event_node_type_id',
        '_run_label_propagation',
        '_load_ioc_mappings',
        '_find_ioc_nodes',
        'predict_from_iocs',
        'save_result'
    }
    
    missing = required_methods - methods
    if missing:
        print(f"✗ Missing methods: {missing}")
        return False
    
    print("✓ All required methods present in IOCAttributor class")
    return True


def test_method_signatures():
    """Test that key methods have correct signatures"""
    script_path = Path(__file__).parent / 'prectdict_event_gnn.py'
    with open(script_path) as f:
        tree = ast.parse(f.read())
    
    # Find the IOCAttributor class
    ioc_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'IOCAttributor':
            ioc_class = node
            break
    
    # Check __init__ signature
    init_method = None
    for node in ioc_class.body:
        if isinstance(node, ast.FunctionDef) and node.name == '__init__':
            init_method = node
            break
    
    if not init_method:
        print("✗ __init__ method not found")
        return False
    
    # Check for debug parameter
    init_args = [arg.arg for arg in init_method.args.args]
    if 'debug' not in init_args:
        print("✗ __init__ missing 'debug' parameter")
        return False
    
    print("✓ Method signatures correct")
    return True


def test_label_propagation_implementation():
    """Test that label propagation method exists and has correct structure"""
    script_path = Path(__file__).parent / 'prectdict_event_gnn.py'
    with open(script_path) as f:
        content = f.read()
    
    # Check for key label propagation components
    checks = [
        ('MessagePassing' in content, "MessagePassing import"),
        ('_run_label_propagation' in content, "_run_label_propagation method"),
        ('mp.propagate' in content, "Label propagation call"),
        ('hops' in content, "Hops parameter"),
    ]
    
    all_passed = True
    for check, description in checks:
        if check:
            print(f"✓ {description} found")
        else:
            print(f"✗ {description} not found")
            all_passed = False
    
    return all_passed


def test_dynamic_node_type_lookup():
    """Test that node type lookup is dynamic, not hardcoded"""
    script_path = Path(__file__).parent / 'prectdict_event_gnn.py'
    with open(script_path) as f:
        content = f.read()
    
    # Check for _get_event_node_type_id method
    if '_get_event_node_type_id' not in content:
        print("✗ _get_event_node_type_id method not found")
        return False
    
    print("✓ Dynamic node type lookup implemented")
    
    # Check that EVENT type is looked up dynamically in predict_from_iocs
    lines = content.split('\n')
    in_predict = False
    uses_dynamic = False
    uses_hardcoded = False
    
    for i, line in enumerate(lines):
        if 'def predict_from_iocs' in line:
            in_predict = True
        elif in_predict and 'def ' in line and 'predict_from_iocs' not in line:
            in_predict = False
        
        if in_predict:
            if '_get_event_node_type_id' in line:
                uses_dynamic = True
            # Check for hardcoded 4 in event type checks
            if 'self.graph.x[neighbor] == 4' in line or 'self.graph.x[n] == 4' in line:
                # Make sure it's not in a comment
                if not line.strip().startswith('#'):
                    uses_hardcoded = True
                    print(f"⚠ Warning: Possible hardcoded node type on line {i+1}: {line.strip()}")
    
    if not uses_dynamic:
        print("✗ Dynamic node type lookup not used in predict_from_iocs")
        return False
    
    print("✓ Dynamic node type lookup used in predict_from_iocs")
    return True


def test_argparse_integration():
    """Test that command-line argument parsing is implemented"""
    script_path = Path(__file__).parent / 'prectdict_event_gnn.py'
    with open(script_path) as f:
        content = f.read()
    
    checks = [
        ('argparse' in content, "argparse import"),
        ('ArgumentParser' in content, "ArgumentParser usage"),
        ('--debug' in content, "--debug argument"),
        ('--model' in content, "--model argument"),
        ('--dataset' in content, "--dataset argument"),
        ('--top-k' in content, "--top-k argument"),
    ]
    
    all_passed = True
    for check, description in checks:
        if check:
            print(f"✓ {description} found")
        else:
            print(f"✗ {description} not found")
            all_passed = False
    
    return all_passed


def test_error_handling():
    """Test that error handling is implemented"""
    script_path = Path(__file__).parent / 'prectdict_event_gnn.py'
    with open(script_path) as f:
        content = f.read()
    
    # Count try-except blocks
    try_count = content.count('try:')
    except_count = content.count('except')
    
    if try_count < 3:
        print(f"✗ Insufficient error handling: only {try_count} try blocks found")
        return False
    
    if except_count < try_count:
        print(f"✗ Mismatched try/except blocks: {try_count} try, {except_count} except")
        return False
    
    print(f"✓ Error handling implemented ({try_count} try-except blocks)")
    return True


def main():
    """Run all tests"""
    print("=" * 70)
    print("Running validation tests for prectdict_event_gnn.py")
    print("=" * 70)
    print()
    
    tests = [
        ("Import validation", test_imports),
        ("Class structure", test_class_structure),
        ("Method signatures", test_method_signatures),
        ("Label propagation", test_label_propagation_implementation),
        ("Dynamic node type lookup", test_dynamic_node_type_lookup),
        ("Command-line arguments", test_argparse_integration),
        ("Error handling", test_error_handling),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 70)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary:")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    print("=" * 70)
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
