#!/usr/bin/env python3
"""
Create minimal ONNX-like binary files for testing
"""

import os
import struct
import numpy as np

def write_varint(value):
    """Write a protobuf varint"""
    result = bytearray()
    while value >= 0x80:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value & 0x7F)
    return result

def write_string(s):
    """Write a length-prefixed string"""
    encoded = s.encode('utf-8')
    return write_varint(len(encoded)) + encoded

def write_field_header(field_number, wire_type):
    """Write a protobuf field header"""
    return write_varint((field_number << 3) | wire_type)

def create_minimal_add_model():
    """Create a minimal ONNX model for Add operation"""
    print("🔧 Creating minimal Add model...")
    
    model_data = bytearray()
    
    # Model header
    # Field 1: ir_version (varint)
    model_data.extend(write_field_header(1, 0))  # field 1, varint
    model_data.extend(write_varint(7))  # IR version 7
    
    # Field 8: producer_name (string)
    model_data.extend(write_field_header(8, 2))  # field 8, length-delimited
    model_data.extend(write_string("zig-ai-platform"))
    
    # Field 7: graph (message)
    graph_data = bytearray()
    
    # Graph field 1: node (repeated message)
    node_data = bytearray()
    
    # Node field 1: input (repeated string)
    node_data.extend(write_field_header(1, 2))  # field 1, length-delimited
    node_data.extend(write_string("A"))
    node_data.extend(write_field_header(1, 2))  # field 1, length-delimited
    node_data.extend(write_string("B"))
    
    # Node field 2: output (repeated string)
    node_data.extend(write_field_header(2, 2))  # field 2, length-delimited
    node_data.extend(write_string("C"))
    
    # Node field 4: op_type (string)
    node_data.extend(write_field_header(4, 2))  # field 4, length-delimited
    node_data.extend(write_string("Add"))
    
    # Node field 5: name (string)
    node_data.extend(write_field_header(5, 2))  # field 5, length-delimited
    node_data.extend(write_string("add_node"))
    
    # Add node to graph
    graph_data.extend(write_field_header(1, 2))  # field 1, length-delimited
    graph_data.extend(write_varint(len(node_data)))
    graph_data.extend(node_data)
    
    # Graph field 2: name (string)
    graph_data.extend(write_field_header(2, 2))  # field 2, length-delimited
    graph_data.extend(write_string("simple_add_graph"))
    
    # Graph field 11: input (repeated ValueInfoProto)
    for input_name in ["A", "B"]:
        input_data = bytearray()
        # ValueInfo field 1: name (string)
        input_data.extend(write_field_header(1, 2))
        input_data.extend(write_string(input_name))
        
        # Add input to graph
        graph_data.extend(write_field_header(11, 2))  # field 11, length-delimited
        graph_data.extend(write_varint(len(input_data)))
        graph_data.extend(input_data)
    
    # Graph field 12: output (repeated ValueInfoProto)
    output_data = bytearray()
    # ValueInfo field 1: name (string)
    output_data.extend(write_field_header(1, 2))
    output_data.extend(write_string("C"))
    
    # Add output to graph
    graph_data.extend(write_field_header(12, 2))  # field 12, length-delimited
    graph_data.extend(write_varint(len(output_data)))
    graph_data.extend(output_data)
    
    # Add graph to model
    model_data.extend(write_field_header(7, 2))  # field 7, length-delimited
    model_data.extend(write_varint(len(graph_data)))
    model_data.extend(graph_data)
    
    # Write to file
    os.makedirs("models", exist_ok=True)
    with open("models/minimal_add.onnx", "wb") as f:
        f.write(model_data)
    
    print(f"✅ Created minimal_add.onnx ({len(model_data)} bytes)")
    return True

def create_minimal_relu_model():
    """Create a minimal ONNX model for ReLU operation"""
    print("🔧 Creating minimal ReLU model...")
    
    model_data = bytearray()
    
    # Model header
    model_data.extend(write_field_header(1, 0))  # ir_version
    model_data.extend(write_varint(7))
    
    model_data.extend(write_field_header(8, 2))  # producer_name
    model_data.extend(write_string("zig-ai-platform"))
    
    # Graph
    graph_data = bytearray()
    
    # ReLU Node
    node_data = bytearray()
    
    # Input
    node_data.extend(write_field_header(1, 2))
    node_data.extend(write_string("X"))
    
    # Output
    node_data.extend(write_field_header(2, 2))
    node_data.extend(write_string("Y"))
    
    # Op type
    node_data.extend(write_field_header(4, 2))
    node_data.extend(write_string("Relu"))
    
    # Name
    node_data.extend(write_field_header(5, 2))
    node_data.extend(write_string("relu_node"))
    
    # Add node to graph
    graph_data.extend(write_field_header(1, 2))
    graph_data.extend(write_varint(len(node_data)))
    graph_data.extend(node_data)
    
    # Graph name
    graph_data.extend(write_field_header(2, 2))
    graph_data.extend(write_string("simple_relu_graph"))
    
    # Input
    input_data = bytearray()
    input_data.extend(write_field_header(1, 2))
    input_data.extend(write_string("X"))
    
    graph_data.extend(write_field_header(11, 2))
    graph_data.extend(write_varint(len(input_data)))
    graph_data.extend(input_data)
    
    # Output
    output_data = bytearray()
    output_data.extend(write_field_header(1, 2))
    output_data.extend(write_string("Y"))
    
    graph_data.extend(write_field_header(12, 2))
    graph_data.extend(write_varint(len(output_data)))
    graph_data.extend(output_data)
    
    # Add graph to model
    model_data.extend(write_field_header(7, 2))
    model_data.extend(write_varint(len(graph_data)))
    model_data.extend(graph_data)
    
    # Write to file
    with open("models/minimal_relu.onnx", "wb") as f:
        f.write(model_data)
    
    print(f"✅ Created minimal_relu.onnx ({len(model_data)} bytes)")
    return True

def create_test_data():
    """Create test input and expected output data"""
    print("🔧 Creating test data...")
    
    os.makedirs("models/test_data", exist_ok=True)
    
    # Test data for Add model
    input_a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_b = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=np.float32)
    expected_add = input_a + input_b
    
    np.save("models/test_data/add_input_a.npy", input_a)
    np.save("models/test_data/add_input_b.npy", input_b)
    np.save("models/test_data/add_expected.npy", expected_add)
    
    # Test data for ReLU model
    relu_input = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
    expected_relu = np.maximum(relu_input, 0.0)
    
    np.save("models/test_data/relu_input.npy", relu_input)
    np.save("models/test_data/relu_expected.npy", expected_relu)
    
    print("✅ Created test data files")
    return True

def create_simple_text_models():
    """Create simple text-based model descriptions for testing"""
    print("🔧 Creating text-based model descriptions...")
    
    os.makedirs("models", exist_ok=True)
    
    # Simple Add model description
    add_model = """
# Simple Add Model
model_name: simple_add
ir_version: 7
producer: zig-ai-platform

graph:
  name: simple_add_graph
  
  inputs:
    - name: A
      type: float32
      shape: [2, 3]
    - name: B
      type: float32
      shape: [2, 3]
  
  outputs:
    - name: C
      type: float32
      shape: [2, 3]
  
  nodes:
    - name: add_node
      op_type: Add
      inputs: [A, B]
      outputs: [C]
"""
    
    with open("models/simple_add.txt", "w") as f:
        f.write(add_model)
    
    # Simple ReLU model description
    relu_model = """
# Simple ReLU Model
model_name: simple_relu
ir_version: 7
producer: zig-ai-platform

graph:
  name: simple_relu_graph
  
  inputs:
    - name: X
      type: float32
      shape: [2, 3]
  
  outputs:
    - name: Y
      type: float32
      shape: [2, 3]
  
  nodes:
    - name: relu_node
      op_type: Relu
      inputs: [X]
      outputs: [Y]
"""
    
    with open("models/simple_relu.txt", "w") as f:
        f.write(relu_model)
    
    print("✅ Created text-based model descriptions")
    return True

def main():
    """Create all test models and data"""
    print("🚀 Creating minimal ONNX test models for real inference")
    print("=" * 50)
    
    success_count = 0
    total_tasks = 4
    
    # Create models
    if create_minimal_add_model():
        success_count += 1
        
    if create_minimal_relu_model():
        success_count += 1
        
    if create_test_data():
        success_count += 1
        
    if create_simple_text_models():
        success_count += 1
    
    print(f"\n🎉 Created {success_count}/{total_tasks} test resources successfully")
    
    if success_count >= 3:
        print("\n🔧 Next steps:")
        print("1. Run: zig build test-real-model")
        print("2. Test inference: zig build test-complete")
        print("3. Run end-to-end: zig build test-e2e")
        print("\n📁 Created files:")
        print("   - models/minimal_add.onnx")
        print("   - models/minimal_relu.onnx")
        print("   - models/simple_add.txt")
        print("   - models/simple_relu.txt")
        print("   - models/test_data/*.npy")
    
    return success_count >= 3

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
