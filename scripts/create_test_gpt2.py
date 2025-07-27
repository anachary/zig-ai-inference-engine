!/usr/bin/env python3
"""
Create a simple test GPT-2-like ONNX model for zig-ai-platform
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, mapping
import os

def create_simple_gpt2_onnx():
    """Create a simple GPT-2-like ONNX model"""
    
    print("🔧 Creating simple GPT-2-like ONNX model...")
    
    # Model parameters
    vocab_size = 50257  # GPT-2 vocabulary size
    hidden_size = 768   # GPT-2 hidden size
    seq_len = 10        # Sequence length for this test
    
    # Create input
    input_ids = helper.make_tensor_value_info(
        'input_ids', 
        TensorProto.INT64, 
        [1, seq_len]  # [batch_size, sequence_length]
    )
    
    # Create output
    logits = helper.make_tensor_value_info(
        'logits', 
        TensorProto.FLOAT, 
        [1, seq_len, vocab_size]  # [batch_size, sequence_length, vocab_size]
    )
    
    # Create embedding weight
    embedding_weight = helper.make_tensor(
        'embedding_weight',
        TensorProto.FLOAT,
        [vocab_size, hidden_size],
        np.random.normal(0, 0.02, (vocab_size, hidden_size)).astype(np.float32).flatten()
    )
    
    # Create output projection weight
    output_weight = helper.make_tensor(
        'output_weight',
        TensorProto.FLOAT,
        [hidden_size, vocab_size],
        np.random.normal(0, 0.02, (hidden_size, vocab_size)).astype(np.float32).flatten()
    )
    
    # Create nodes
    nodes = []
    
    # 1. Embedding lookup: Gather(embedding_weight, input_ids)
    nodes.append(helper.make_node(
        'Gather',
        inputs=['embedding_weight', 'input_ids'],
        outputs=['embeddings'],
        axis=0
    ))
    
    # 2. Simple transformation (just a reshape to add batch dimension properly)
    nodes.append(helper.make_node(
        'Reshape',
        inputs=['embeddings'],
        outputs=['reshaped_embeddings'],
        # We'll add the shape as an initializer
    ))
    
    # 3. Output projection: MatMul(reshaped_embeddings, output_weight)
    nodes.append(helper.make_node(
        'MatMul',
        inputs=['reshaped_embeddings', 'output_weight'],
        outputs=['logits']
    ))
    
    # Create reshape shape tensor
    reshape_shape = helper.make_tensor(
        'reshape_shape',
        TensorProto.INT64,
        [3],
        np.array([1, seq_len, hidden_size], dtype=np.int64)
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes,
        'simple_gpt2',
        [input_ids],
        [logits],
        [embedding_weight, output_weight, reshape_shape]
    )
    
    # Create the model
    model = helper.make_model(graph, producer_name='zig-ai-platform')
    model.opset_import[0].version = 11
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/simple_gpt2.onnx'
    onnx.save(model, model_path)
    
    print(f"✅ Created simple GPT-2 model: {model_path}")
    
    # Verify the model
    try:
        onnx.checker.check_model(model)
        print("✅ Model verification passed")
        
        # Get file size
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"📊 Model size: {size_mb:.1f}MB")
        
        print(f"\n🎯 Model details:")
        print(f"   - Vocabulary size: {vocab_size}")
        print(f"   - Hidden size: {hidden_size}")
        print(f"   - Input shape: [1, {seq_len}]")
        print(f"   - Output shape: [1, {seq_len}, {vocab_size}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False

def create_minimal_gpt2():
    """Create an even simpler model if the above fails"""
    
    print("🔧 Creating minimal GPT-2-like model...")
    
    # Very simple parameters
    vocab_size = 1000   # Smaller vocabulary
    hidden_size = 64    # Smaller hidden size
    seq_len = 5         # Shorter sequence
    
    # Create a simple linear transformation model
    # Input: [batch, seq_len] -> Output: [batch, seq_len, vocab_size]
    
    input_ids = helper.make_tensor_value_info(
        'input_ids', 
        TensorProto.INT64, 
        [1, seq_len]
    )
    
    logits = helper.make_tensor_value_info(
        'logits', 
        TensorProto.FLOAT, 
        [1, seq_len, vocab_size]
    )
    
    # Create a simple lookup table
    lookup_table = helper.make_tensor(
        'lookup_table',
        TensorProto.FLOAT,
        [1000, vocab_size],  # Map any input token to vocab_size outputs
        np.random.normal(0, 1, (1000, vocab_size)).astype(np.float32).flatten()
    )
    
    # Single Gather operation
    nodes = [
        helper.make_node(
            'Gather',
            inputs=['lookup_table', 'input_ids'],
            outputs=['logits'],
            axis=0
        )
    ]
    
    graph = helper.make_graph(
        nodes,
        'minimal_gpt2',
        [input_ids],
        [logits],
        [lookup_table]
    )
    
    model = helper.make_model(graph, producer_name='zig-ai-platform')
    model.opset_import[0].version = 11
    
    model_path = 'models/minimal_gpt2.onnx'
    onnx.save(model, model_path)
    
    print(f"✅ Created minimal GPT-2 model: {model_path}")
    return True

def main():
    """Main function"""
    print("🤖 Creating Test GPT-2 ONNX Model")
    print("=" * 40)
    
    try:
        # Try the simple model first
        if create_simple_gpt2_onnx():
            print(f"\n🎉 Success! Test GPT-2 model created.")
            print(f"\n🚀 Test with zig-ai-platform:")
            print(f"   zig build")
            print(f"   .\\zig-out\\bin\\zig-ai.exe chat --model models\\simple_gpt2.onnx")
            return
    except Exception as e:
        print(f"❌ Simple model creation failed: {e}")
    
    try:
        # Fallback to minimal model
        print("\n🔄 Trying minimal model...")
        if create_minimal_gpt2():
            print(f"\n🎉 Minimal GPT-2 model created.")
            print(f"\n🚀 Test with zig-ai-platform:")
            print(f"   zig build")
            print(f"   .\\zig-out\\bin\\zig-ai.exe chat --model models\\minimal_gpt2.onnx")
            return
    except Exception as e:
        print(f"❌ Minimal model creation failed: {e}")
    
    print("\n❌ Could not create test model.")
    print("💡 You may need to install: pip install onnx numpy")

if __name__ == "__main__":
    main()
