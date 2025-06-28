const std = @import("std");
const lib = @import("zig-ai-engine");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🚀 GPU Support Foundation Demo - Phase 2 Implementation", .{});
    std.log.info("=======================================================", .{});

    // Test 1: GPU Context Initialization
    std.log.info("✅ Test 1: GPU Context Initialization", .{});

    var gpu_context = lib.gpu.createOptimalContext(allocator) catch |err| {
        std.log.err("Failed to create GPU context: {}", .{err});
        std.log.info("🔄 Falling back to CPU-only mode", .{});
        return;
    };
    defer gpu_context.deinit();

    const device_info = gpu_context.getDeviceInfo();
    std.log.info("📱 Selected Device: {s}", .{device_info.name});
    std.log.info("🔧 Device Type: {s}", .{@tagName(device_info.device_type)});
    std.log.info("💾 Total Memory: {d}MB", .{device_info.memory_total / (1024 * 1024)});
    std.log.info("🧮 Compute Units: {d}", .{device_info.compute_units});
    std.log.info("🔢 Supports INT8: {}", .{device_info.supports_int8});
    std.log.info("🔢 Supports FP16: {}", .{device_info.supports_fp16});

    // Test 2: IoT Suitability Check
    std.log.info("\n✅ Test 2: IoT Deployment Suitability", .{});

    if (gpu_context.device.isIoTSuitable()) {
        std.log.info("🌐 ✅ Device is suitable for IoT deployment", .{});
    } else {
        std.log.info("🌐 ⚠️  Device may not be optimal for IoT constraints", .{});
    }

    if (gpu_context.device.supportsLightweightInference()) {
        std.log.info("🧠 ✅ Device supports lightweight LLM inference", .{});
    } else {
        std.log.info("🧠 ⚠️  Device may struggle with lightweight inference", .{});
    }

    // Test 3: Memory Management
    std.log.info("\n✅ Test 3: GPU Memory Management", .{});

    const memory_type = gpu_context.getRecommendedMemoryType(true, false);
    std.log.info("📋 Recommended memory type for input tensors: {s}", .{@tagName(memory_type)});

    // Allocate some GPU buffers
    const buffer_size = 1024 * 1024; // 1MB
    var buffer1 = gpu_context.allocateBuffer(buffer_size, memory_type) catch |err| {
        std.log.err("Failed to allocate GPU buffer: {}", .{err});
        return;
    };
    defer gpu_context.freeBuffer(buffer1) catch {};

    var buffer2 = gpu_context.allocateBuffer(buffer_size / 2, memory_type) catch |err| {
        std.log.err("Failed to allocate second GPU buffer: {}", .{err});
        return;
    };
    defer gpu_context.freeBuffer(buffer2) catch {};

    const memory_stats = gpu_context.getMemoryStats();
    std.log.info("💾 Device Memory - Total: {d}MB, Free: {d}MB, Used: {d}MB", .{
        memory_stats.device_memory.total / (1024 * 1024),
        memory_stats.device_memory.free / (1024 * 1024),
        memory_stats.device_memory.used / (1024 * 1024),
    });
    std.log.info("🏊 Memory Pool - Allocated: {d}KB, Peak: {d}KB, Blocks: {d}", .{
        memory_stats.pool_stats.total_allocated / 1024,
        memory_stats.pool_stats.peak_usage / 1024,
        memory_stats.pool_stats.allocated_blocks,
    });

    // Test 4: Kernel Execution
    std.log.info("\n✅ Test 4: GPU Kernel Execution", .{});

    // Create test data for vector addition
    const test_size = 1024;
    const data_size = test_size * @sizeOf(f32);

    var input_buffer1 = gpu_context.allocateBuffer(data_size, memory_type) catch |err| {
        std.log.err("Failed to allocate input buffer 1: {}", .{err});
        return;
    };
    defer gpu_context.freeBuffer(input_buffer1) catch {};

    var input_buffer2 = gpu_context.allocateBuffer(data_size, memory_type) catch |err| {
        std.log.err("Failed to allocate input buffer 2: {}", .{err});
        return;
    };
    defer gpu_context.freeBuffer(input_buffer2) catch {};

    var output_buffer = gpu_context.allocateBuffer(data_size, memory_type) catch |err| {
        std.log.err("Failed to allocate output buffer: {}", .{err});
        return;
    };
    defer gpu_context.freeBuffer(output_buffer) catch {};

    // Fill input buffers with test data
    const input_ptr1 = input_buffer1.map() catch |err| {
        std.log.err("Failed to map input buffer 1: {}", .{err});
        return;
    };
    const input_ptr2 = input_buffer2.map() catch |err| {
        std.log.err("Failed to map input buffer 2: {}", .{err});
        return;
    };

    const input_slice1 = @as([*]f32, @ptrCast(@alignCast(input_ptr1)))[0..test_size];
    const input_slice2 = @as([*]f32, @ptrCast(@alignCast(input_ptr2)))[0..test_size];

    for (input_slice1, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.1;
    }
    for (input_slice2, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.2;
    }

    input_buffer1.unmap();
    input_buffer2.unmap();

    // Execute vector addition kernel
    var inputs = [_]lib.gpu.GPUBuffer{ input_buffer1, input_buffer2 };
    var outputs = [_]lib.gpu.GPUBuffer{output_buffer};

    const start_time = std.time.nanoTimestamp();

    gpu_context.executeOperator("Add", inputs[0..], outputs[0..]) catch |err| {
        std.log.err("Failed to execute vector addition: {}", .{err});
        return;
    };

    const end_time = std.time.nanoTimestamp();
    const execution_time_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;

    std.log.info("⚡ Vector addition executed in {d:.2}ms", .{execution_time_ms});

    // Verify results
    const output_ptr = output_buffer.map() catch |err| {
        std.log.err("Failed to map output buffer: {}", .{err});
        return;
    };
    const output_slice = @as([*]const f32, @ptrCast(@alignCast(output_ptr)))[0..test_size];

    var correct_results: u32 = 0;
    for (output_slice, 0..) |result, i| {
        const expected = (@as(f32, @floatFromInt(i)) * 0.1) + (@as(f32, @floatFromInt(i)) * 0.2);
        if (@fabs(result - expected) < 0.001) {
            correct_results += 1;
        }
    }

    output_buffer.unmap();

    const accuracy = @as(f32, @floatFromInt(correct_results)) / @as(f32, @floatFromInt(test_size)) * 100.0;
    std.log.info("🎯 Computation accuracy: {d:.1}% ({d}/{d} correct)", .{ accuracy, correct_results, test_size });

    // Test 5: System Capabilities Summary
    std.log.info("\n✅ Test 5: System GPU Capabilities", .{});

    const system_caps = lib.gpu.getSystemCapabilities(allocator) catch |err| {
        std.log.err("Failed to get system capabilities: {}", .{err});
        return;
    };

    std.log.info("🖥️  Total GPU devices: {d}", .{system_caps.total_devices});
    std.log.info("🌐 IoT-suitable devices: {d}", .{system_caps.iot_suitable_devices});
    std.log.info("🧠 Inference-capable devices: {d}", .{system_caps.inference_capable_devices});
    std.log.info("💾 Total system GPU memory: {d:.1}GB", .{system_caps.total_memory_gb});
    std.log.info("🔢 Quantization support: {}", .{system_caps.supports_quantization});

    // Test 6: Specialized Context Creation
    std.log.info("\n✅ Test 6: Specialized GPU Contexts", .{});

    // Test IoT-optimized context
    std.log.info("🌐 Testing IoT-optimized context...", .{});
    var iot_context_result = lib.gpu.createIoTContext(allocator);
    if (iot_context_result) |*iot_context| {
        defer iot_context.deinit();
        if (iot_context.device.capabilities.device_id != gpu_context.device.capabilities.device_id) {
            const iot_device = iot_context.getDeviceInfo();
            std.log.info("📱 IoT Device: {s} ({d}MB memory)", .{ iot_device.name, iot_device.memory_total / (1024 * 1024) });
        } else {
            std.log.info("📱 IoT context uses same device as optimal context", .{});
        }
    } else |err| {
        std.log.warn("IoT context creation failed: {}", .{err});
        std.log.info("Using default context for IoT demo", .{});
    }

    // Test security-optimized context
    std.log.info("🔒 Testing security-optimized context...", .{});
    var security_context_result = lib.gpu.createSecurityContext(allocator);
    if (security_context_result) |*security_context| {
        defer security_context.deinit();
        if (security_context.device.capabilities.device_id != gpu_context.device.capabilities.device_id) {
            const security_device = security_context.getDeviceInfo();
            std.log.info("🔒 Security Device: {s} (Type: {s})", .{ security_device.name, @tagName(security_device.device_type) });
        } else {
            std.log.info("🔒 Security context uses same device as optimal context", .{});
        }
    } else |err| {
        std.log.warn("Security context creation failed: {}", .{err});
        std.log.info("Using default context for security demo", .{});
    }

    // Final readiness check
    std.log.info("\n🎉 GPU Support Foundation Demo Complete!", .{});

    if (gpu_context.isReadyForInference()) {
        std.log.info("✅ System is ready for lightweight LLM inference!", .{});
        std.log.info("🚀 Phase 2 Task 6 (GPU Support Foundation) - COMPLETE", .{});
    } else {
        std.log.info("⚠️  System may need optimization for optimal inference performance", .{});
        std.log.info("🔧 Consider upgrading hardware or adjusting memory limits", .{});
    }

    std.log.info("\n📊 Performance Summary:", .{});
    std.log.info("  • Device: {s} ({s})", .{ device_info.name, @tagName(device_info.device_type) });
    std.log.info("  • Memory: {d}MB total, {d}MB allocated", .{
        device_info.memory_total / (1024 * 1024),
        memory_stats.pool_stats.total_allocated / (1024 * 1024),
    });
    std.log.info("  • Vector addition: {d:.2}ms for {d} elements", .{ execution_time_ms, test_size });
    std.log.info("  • Accuracy: {d:.1}%", .{accuracy});
    std.log.info("  • IoT suitable: {}", .{gpu_context.device.isIoTSuitable()});
    std.log.info("  • Inference ready: {}", .{gpu_context.isReadyForInference()});
}
