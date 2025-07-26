#!/usr/bin/env python3
"""
Create a simple ONNX model for testing real inference
"""

import os
import numpy as np

def create_simple_add_model():
    """Create a simple addition model using raw ONNX protobuf"""
    print("🔧 Creating simple addition model...")
    
    # Create a minimal ONNX model manually
    # This is a simplified approach without using onnx library
    
    # ONNX protobuf structure for a simple Add operation:
    # ModelProto {
    #   graph: GraphProto {
    #     node: [NodeProto { op_type: "Add", input: ["A", "B"], output: ["C"] }]
    #     input: [ValueInfoProto { name: "A" }, ValueInfoProto { name: "B" }]
    #     output: [ValueInfoProto { name: "C" }]
    #   }
    # }
    
    # For now, create a text representation that our parser can handle
    model_content = """
    # Simple ONNX-like model for testing
    # This is a placeholder until we can create real ONNX files
    model_name: "simple_add"
    inputs: ["A", "B"]
    outputs: ["C"]
    nodes:
      - name: "add_node"
        op_type: "Add"
        inputs: ["A", "B"]
        outputs: ["C"]
    """
    
    os.makedirs("models", exist_ok=True)
    with open("models/simple_test.txt", "w") as f:
        f.write(model_content)
    
    print("✅ Created simple test model at models/simple_test.txt")
    return True

def create_minimal_onnx_bytes():
    """Create minimal ONNX model as raw bytes"""
    print("🔧 Creating minimal ONNX model as bytes...")
    
    # This creates a very basic ONNX-like binary file
    # In a real implementation, we'd use the onnx library
    
    # Minimal protobuf-like structure
    model_bytes = bytearray()
    
    # Add some basic protobuf-like headers
    # Field 1 (ir_version): varint 7
    model_bytes.extend([0x08, 0x07])  # field 1, varint, value 7
    
    # Field 7 (graph): length-delimited
    graph_data = bytearray()
    
    # Graph field 2 (name): "test_graph"
    graph_name = b"test_graph"
    graph_data.extend([0x12, len(graph_name)])  # field 2, length-delimited
    graph_data.extend(graph_name)
    
    # Add graph to model
    model_bytes.extend([0x3A, len(graph_data)])  # field 7, length-delimited
    model_bytes.extend(graph_data)
    
    # Write to file
    os.makedirs("models", exist_ok=True)
    with open("models/minimal_test.onnx", "wb") as f:
        f.write(model_bytes)
    
    print(f"✅ Created minimal ONNX model at models/minimal_test.onnx ({len(model_bytes)} bytes)")
    return True

def create_test_data():
    """Create test input data"""
    print("🔧 Creating test input data...")
    
    # Create simple test tensors
    test_data = {
        "input_a": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "input_b": np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
        "expected_output": np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    }
    
    os.makedirs("models", exist_ok=True)
    
    for name, data in test_data.items():
        np.save(f"models/{name}.npy", data)
        print(f"✅ Created {name}.npy: shape {data.shape}")
    
    return True

def main():
    """Create test models and data"""
    print("🚀 Creating test models for real inference testing")
    print("=" * 50)
    
    success_count = 0
    
    # Create different types of test models
    if create_simple_add_model():
        success_count += 1
    
    if create_minimal_onnx_bytes():
        success_count += 1
        
    if create_test_data():
        success_count += 1
    
    print(f"\n🎉 Created {success_count}/3 test resources successfully")
    
    if success_count > 0:
        print("\n🔧 Next steps:")
        print("1. Run: zig build test-real-model")
        print("2. Test parsing: zig run test_real_model.zig")
        print("3. Check models directory for created files")
    
    return success_count >= 2

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
