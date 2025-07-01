const std = @import("std");
const print = std.debug.print;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    _ = gpa.allocator(); // For future use

    print("ONNX File Diagnostic Tool\n", .{});
    print("============================\n\n", .{});

    const model_path = "models/model_fp16.onnx";

    // Check if file exists and get basic info
    const file = std.fs.cwd().openFile(model_path, .{}) catch |err| {
        print("❌ Cannot open file: {}\n", .{err});
        return;
    };
    defer file.close();

    const file_size = try file.getEndPos();
    print("📂 File: {s}\n", .{model_path});
    print("📊 Size: {d:.2} MB ({} bytes)\n", .{ @as(f64, @floatFromInt(file_size)) / (1024.0 * 1024.0), file_size });

    // Read first 64 bytes to analyze header
    var header_buffer: [64]u8 = undefined;
    _ = try file.read(&header_buffer);

    print("\nFile Header Analysis\n", .{});
    print("----------------------\n", .{});
    print("First 32 bytes (hex): ", .{});
    for (header_buffer[0..32]) |byte| {
        print("{X:0>2} ", .{byte});
    }
    print("\n", .{});

    print("First 16 bytes (dec): ", .{});
    for (header_buffer[0..16]) |byte| {
        print("{:3} ", .{byte});
    }
    print("\n", .{});

    // Analyze protobuf structure
    print("\nProtobuf Analysis\n", .{});
    print("-------------------\n", .{});

    var i: usize = 0;
    while (i < 32 and i < header_buffer.len) {
        const byte = header_buffer[i];
        const field_number = byte >> 3;
        const wire_type = byte & 0x07;

        if (field_number > 0 and field_number <= 15 and wire_type <= 5) {
            print("Byte {}: Field {} Wire Type {} ", .{ i, field_number, wire_type });
            switch (wire_type) {
                0 => print("(varint)\n", .{}),
                1 => print("(64-bit)\n", .{}),
                2 => print("(length-delimited)\n", .{}),
                3 => print("(start group - deprecated)\n", .{}),
                4 => print("(end group - deprecated)\n", .{}),
                5 => print("(32-bit)\n", .{}),
                else => print("(unknown)\n", .{}),
            }
        }
        i += 1;
    }

    // Check for ONNX-specific patterns
    print("\nONNX Pattern Detection\n", .{});
    print("-------------------------\n", .{});

    // Look for common ONNX field numbers
    var found_ir_version = false;
    var found_graph = false;
    var found_opset = false;

    for (header_buffer[0..32], 0..) |byte, idx| {
        const field_number = byte >> 3;
        const wire_type = byte & 0x07;

        switch (field_number) {
            1 => if (wire_type == 0) {
                found_ir_version = true;
                print("Found IR version field at byte {}\n", .{idx});
            },
            2 => if (wire_type == 2) {
                found_graph = true;
                print("Found graph field at byte {}\n", .{idx});
            },
            8 => if (wire_type == 2) {
                found_opset = true;
                print("Found opset import field at byte {}\n", .{idx});
            },
            else => {},
        }
    }

    print("\nDetection Summary\n", .{});
    print("-------------------\n", .{});
    print("IR Version field: {s}\n", .{if (found_ir_version) "Found" else "Not found"});
    print("Graph field: {s}\n", .{if (found_graph) "Found" else "Not found"});
    print("Opset field: {s}\n", .{if (found_opset) "Found" else "Not found"});

    const likely_onnx = found_ir_version or found_graph;
    print("\nLikely ONNX format: {s}\n", .{if (likely_onnx) "Yes" else "No"});

    if (!likely_onnx) {
        print("\nThis file may not be a standard ONNX file, or it uses\n", .{});
        print("   a different protobuf encoding than expected.\n", .{});
        print("\nPossible issues:\n", .{});
        print("   - File is corrupted\n", .{});
        print("   - File is not actually ONNX format\n", .{});
        print("   - File uses newer ONNX version with different structure\n", .{});
        print("   - File is compressed or encoded\n", .{});
    } else {
        print("\nFile appears to be a valid ONNX model!\n", .{});
        print("The parsing issues are likely due to:\n", .{});
        print("   - Incomplete protobuf parser implementation\n", .{});
        print("   - Missing support for specific ONNX features\n", .{});
        print("   - Complex model structure not yet supported\n", .{});
    }

    print("\nRecommendations\n", .{});
    print("------------------\n", .{});
    print("1. Enhance protobuf parser to handle more field types\n", .{});
    print("2. Implement missing ONNX operators and features\n", .{});
    print("3. Add better error recovery and partial parsing\n", .{});
    print("4. Test with simpler ONNX models first\n", .{});
}
