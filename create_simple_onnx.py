#!/usr/bin/env python3
"""
Create simple ONNX models for testing real inference
"""

import os
import numpy as np

def create_simple_add_model():
    """Create a simple addition model using numpy operations"""
    print("🔧 Creating simple addition model...")
    
    try:
        import onnx
        from onnx import helper, TensorProto, mapping
    except ImportError:
        print("❌ ONNX library not available. Installing...")
        os.system("pip install onnx")
        import onnx
        from onnx import helper, TensorProto, mapping

    # Create input tensors
    input_a = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 3])
    input_b = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2, 3])
    
    # Create output tensor
    output_c = helper.make_tensor_value_info('C', TensorProto.FLOAT, [2, 3])
    
    # Create Add node
    add_node = helper.make_node(
        'Add',
        inputs=['A', 'B'],
        outputs=['C'],
        name='add_node'
    )
    
    # Create graph
    graph = helper.make_graph(
        nodes=[add_node],
        name='simple_add_graph',
        inputs=[input_a, input_b],
        outputs=[output_c]
    )
    
    # Create model
    model = helper.make_model(graph, producer_name='zig-ai-platform')
    model.opset_import[0].version = 13
    
    # Verify model
    onnx.checker.check_model(model)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    onnx.save(model, "models/simple_add.onnx")
    
    print("✅ Created simple_add.onnx")
    return True

def create_simple_matmul_model():
    """Create a simple matrix multiplication model"""
    print("🔧 Creating simple MatMul model...")
    
    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError:
        print("❌ ONNX library not available")
        return False

    # Create input tensors
    input_a = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 3])
    input_b = helper.make_tensor_value_info('B', TensorProto.FLOAT, [3, 4])
    
    # Create output tensor
    output_c = helper.make_tensor_value_info('C', TensorProto.FLOAT, [2, 4])
    
    # Create MatMul node
    matmul_node = helper.make_node(
        'MatMul',
        inputs=['A', 'B'],
        outputs=['C'],
        name='matmul_node'
    )
    
    # Create graph
    graph = helper.make_graph(
        nodes=[matmul_node],
        name='simple_matmul_graph',
        inputs=[input_a, input_b],
        outputs=[output_c]
    )
    
    # Create model
    model = helper.make_model(graph, producer_name='zig-ai-platform')
    model.opset_import[0].version = 13
    
    # Verify model
    onnx.checker.check_model(model)
    
    # Save model
    onnx.save(model, "models/simple_matmul.onnx")
    
    print("✅ Created simple_matmul.onnx")
    return True

def create_simple_relu_model():
    """Create a simple ReLU activation model"""
    print("🔧 Creating simple ReLU model...")
    
    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError:
        print("❌ ONNX library not available")
        return False

    # Create input tensor
    input_x = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 3])
    
    # Create output tensor
    output_y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 3])
    
    # Create ReLU node
    relu_node = helper.make_node(
        'Relu',
        inputs=['X'],
        outputs=['Y'],
        name='relu_node'
    )
    
    # Create graph
    graph = helper.make_graph(
        nodes=[relu_node],
        name='simple_relu_graph',
        inputs=[input_x],
        outputs=[output_y]
    )
    
    # Create model
    model = helper.make_model(graph, producer_name='zig-ai-platform')
    model.opset_import[0].version = 13
    
    # Verify model
    onnx.checker.check_model(model)
    
    # Save model
    onnx.save(model, "models/simple_relu.onnx")
    
    print("✅ Created simple_relu.onnx")
    return True

def create_simple_constant_model():
    """Create a model with constant values"""
    print("🔧 Creating simple Constant model...")
    
    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper
    except ImportError:
        print("❌ ONNX library not available")
        return False

    # Create constant tensor
    const_tensor = numpy_helper.from_array(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        name='const_value'
    )
    
    # Create input tensor
    input_x = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 3])
    
    # Create output tensor
    output_y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 3])
    
    # Create Constant node
    const_node = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['const_value'],
        value=const_tensor,
        name='const_node'
    )
    
    # Create Add node
    add_node = helper.make_node(
        'Add',
        inputs=['X', 'const_value'],
        outputs=['Y'],
        name='add_node'
    )
    
    # Create graph
    graph = helper.make_graph(
        nodes=[const_node, add_node],
        name='simple_constant_graph',
        inputs=[input_x],
        outputs=[output_y]
    )
    
    # Create model
    model = helper.make_model(graph, producer_name='zig-ai-platform')
    model.opset_import[0].version = 13
    
    # Verify model
    onnx.checker.check_model(model)
    
    # Save model
    onnx.save(model, "models/simple_constant.onnx")
    
    print("✅ Created simple_constant.onnx")
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
    
    # Test data for MatMul model
    matmul_a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    matmul_b = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    expected_matmul = np.matmul(matmul_a, matmul_b)
    
    np.save("models/test_data/matmul_input_a.npy", matmul_a)
    np.save("models/test_data/matmul_input_b.npy", matmul_b)
    np.save("models/test_data/matmul_expected.npy", expected_matmul)
    
    # Test data for ReLU model
    relu_input = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
    expected_relu = np.maximum(relu_input, 0.0)
    
    np.save("models/test_data/relu_input.npy", relu_input)
    np.save("models/test_data/relu_expected.npy", expected_relu)
    
    # Test data for Constant model
    const_input = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    const_value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    expected_const = const_input + const_value
    
    np.save("models/test_data/const_input.npy", const_input)
    np.save("models/test_data/const_expected.npy", expected_const)
    
    print("✅ Created test data files")
    return True

def main():
    """Create all test models and data"""
    print("🚀 Creating ONNX test models for real inference")
    print("=" * 50)
    
    success_count = 0
    total_tasks = 5
    
    # Create models
    if create_simple_add_model():
        success_count += 1
    
    if create_simple_matmul_model():
        success_count += 1
        
    if create_simple_relu_model():
        success_count += 1
        
    if create_simple_constant_model():
        success_count += 1
        
    if create_test_data():
        success_count += 1
    
    print(f"\n🎉 Created {success_count}/{total_tasks} test resources successfully")
    
    if success_count >= 4:
        print("\n🔧 Next steps:")
        print("1. Run: zig build test-real-model")
        print("2. Test inference: zig build test-complete")
        print("3. Run end-to-end: zig build test-e2e")
    
    return success_count >= 4

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
