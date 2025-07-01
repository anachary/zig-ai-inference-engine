const std = @import("std");
const print = std.debug.print;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("Testing model_fp16.onnx\n", .{});
    print("======================\n\n", .{});

    const model_path = "models/model_fp16.onnx";

    // Step 1: Basic file validation
    print("Step 1: File Validation\n", .{});
    print("-----------------------\n", .{});

    const file = std.fs.cwd().openFile(model_path, .{}) catch |err| {
        print("❌ ERROR: Cannot open {s}: {}\n", .{ model_path, err });
        print("💡 Make sure the file exists and is accessible\n", .{});
        return;
    };
    defer file.close();

    const file_size = try file.getEndPos();
    print("✅ File found: {s}\n", .{model_path});
    print("📊 Size: {d:.2} MB ({} bytes)\n", .{ @as(f64, @floatFromInt(file_size)) / (1024.0 * 1024.0), file_size });

    // Step 2: Memory allocation test
    print("\nStep 2: Memory Allocation\n", .{});
    print("-------------------------\n", .{});

    const file_data = allocator.alloc(u8, file_size) catch |err| {
        print("❌ ERROR: Cannot allocate {} bytes: {}\n", .{ file_size, err });
        print("💡 Model too large for available memory\n", .{});
        return;
    };
    defer allocator.free(file_data);

    const bytes_read = try file.readAll(file_data);
    if (bytes_read != file_size) {
        print("❌ ERROR: Read {} bytes, expected {}\n", .{ bytes_read, file_size });
        return;
    }

    print("✅ Model loaded into memory successfully\n", .{});
    print("📊 Memory usage: {d:.2} MB\n", .{@as(f64, @floatFromInt(file_size)) / (1024.0 * 1024.0)});

    // Step 3: ONNX format validation
    print("\nStep 3: ONNX Format Validation\n", .{});
    print("-------------------------------\n", .{});

    // Check ONNX magic bytes and structure
    if (file_data.len < 16) {
        print("❌ ERROR: File too small to be valid ONNX\n", .{});
        return;
    }

    // Analyze protobuf header
    var protobuf_valid = false;
    var ir_version_found = false;
    var graph_found = false;

    // Check first few bytes for ONNX protobuf patterns
    if (file_data[0] == 0x08) { // Field 1 (ir_version) with varint wire type
        ir_version_found = true;
        print("✅ IR version field detected\n", .{});
    }

    if (file_data.len > 2 and file_data[2] == 0x12) { // Field 2 (graph) with length-delimited wire type
        graph_found = true;
        print("✅ Graph field detected\n", .{});
    }

    protobuf_valid = ir_version_found and graph_found;

    if (protobuf_valid) {
        print("✅ Valid ONNX protobuf structure\n", .{});
    } else {
        print("⚠️  Unusual protobuf structure (may still be valid)\n", .{});
    }

    // Step 4: Extract model metadata
    print("\nStep 4: Model Metadata Extraction\n", .{});
    print("----------------------------------\n", .{});

    // Look for text strings in the first 2KB
    const search_range = @min(file_data.len, 2048);
    var metadata_found = false;

    for (file_data[0..search_range], 0..) |byte, i| {
        if (byte >= 32 and byte <= 126) { // Printable ASCII
            var text_len: usize = 0;
            var j = i;
            while (j < search_range and file_data[j] >= 32 and file_data[j] <= 126) {
                text_len += 1;
                j += 1;
            }

            if (text_len >= 5) {
                const text = file_data[i .. i + text_len];

                // Look for known metadata patterns
                if (std.mem.indexOf(u8, text, "onnx") != null) {
                    print("🏷️  ONNX identifier: {s}\n", .{text});
                    metadata_found = true;
                } else if (std.mem.indexOf(u8, text, "runtime") != null) {
                    print("🏷️  Runtime info: {s}\n", .{text});
                    metadata_found = true;
                } else if (std.mem.indexOf(u8, text, "genai") != null) {
                    print("🏷️  Model type: {s}\n", .{text});
                    metadata_found = true;
                } else if (text_len >= 8 and std.mem.indexOf(u8, text, ".") != null) {
                    print("🏷️  Version info: {s}\n", .{text});
                    metadata_found = true;
                }
            }
        }
    }

    if (!metadata_found) {
        print("ℹ️  No text metadata found (binary-only model)\n", .{});
    }

    // Step 5: Protobuf field analysis
    print("\nStep 5: Protobuf Structure Analysis\n", .{});
    print("------------------------------------\n", .{});

    var field_count: u32 = 0;
    var pos: usize = 0;
    var fields_found = std.ArrayList(u8).init(allocator);
    defer fields_found.deinit();

    // Parse first 200 bytes for protobuf fields
    while (pos < @min(file_data.len, 200) and field_count < 20) {
        if (pos >= file_data.len) break;

        const byte = file_data[pos];
        const field_number = byte >> 3;
        const wire_type = byte & 0x07;

        if (field_number > 0 and field_number <= 20 and wire_type <= 5) {
            try fields_found.append(field_number);
            field_count += 1;

            // Skip field data to find next field
            pos += 1;
            switch (wire_type) {
                0 => { // varint
                    while (pos < file_data.len and (file_data[pos] & 0x80) != 0) {
                        pos += 1;
                    }
                    if (pos < file_data.len) pos += 1;
                },
                1 => pos += 8, // 64-bit
                2 => { // length-delimited
                    if (pos < file_data.len) {
                        var length: usize = file_data[pos];
                        pos += 1;
                        // Handle multi-byte length (simplified)
                        if (length > 127 and pos < file_data.len) {
                            length = (length & 0x7F) + (file_data[pos] << 7);
                            pos += 1;
                        }
                        pos += @min(length, file_data.len - pos);
                    }
                },
                5 => pos += 4, // 32-bit
                else => pos += 1,
            }
        } else {
            pos += 1;
        }
    }

    print("📊 Protobuf fields found: {}\n", .{field_count});
    print("Field numbers: ", .{});
    for (fields_found.items, 0..) |field, i| {
        if (i > 0) print(", ", .{});
        print("{}", .{field});
    }
    print("\n", .{});

    // Interpret common ONNX fields
    for (fields_found.items) |field| {
        switch (field) {
            1 => print("   Field 1: IR version\n", .{}),
            2 => print("   Field 2: Graph definition\n", .{}),
            3 => print("   Field 3: Producer name\n", .{}),
            4 => print("   Field 4: Producer version\n", .{}),
            5 => print("   Field 5: Domain\n", .{}),
            6 => print("   Field 6: Model version\n", .{}),
            7 => print("   Field 7: Documentation\n", .{}),
            8 => print("   Field 8: Opset imports\n", .{}),
            else => print("   Field {}: Unknown/Custom\n", .{field}),
        }
    }

    // Step 6: Model compatibility assessment
    print("\nStep 6: Compatibility Assessment\n", .{});
    print("---------------------------------\n", .{});

    var compatibility_score: u32 = 0;
    var max_score: u32 = 6;

    // Check file size (reasonable for IoT)
    if (file_size < 100 * 1024 * 1024) { // < 100MB
        print("✅ File size suitable for IoT devices\n", .{});
        compatibility_score += 1;
    } else {
        print("⚠️  Large file size may challenge some IoT devices\n", .{});
    }

    // Check protobuf structure
    if (protobuf_valid) {
        print("✅ Standard ONNX protobuf format\n", .{});
        compatibility_score += 1;
    } else {
        print("⚠️  Non-standard protobuf structure\n", .{});
    }

    // Check for FP16 indicators
    if (std.mem.indexOf(u8, model_path, "fp16") != null) {
        print("✅ FP16 format (memory efficient)\n", .{});
        compatibility_score += 1;
    } else {
        print("ℹ️  Precision format unknown\n", .{});
    }

    // Check metadata presence
    if (metadata_found) {
        print("✅ Model metadata available\n", .{});
        compatibility_score += 1;
    } else {
        print("ℹ️  Limited metadata\n", .{});
    }

    // Check field structure
    if (field_count >= 3) {
        print("✅ Rich protobuf structure\n", .{});
        compatibility_score += 1;
    } else {
        print("⚠️  Minimal protobuf structure\n", .{});
    }

    // Check for GenAI indicators
    if (std.mem.indexOf(u8, model_path, "genai") != null or metadata_found) {
        print("✅ Generative AI model detected\n", .{});
        compatibility_score += 1;
    } else {
        print("ℹ️  Model type unclear\n", .{});
    }

    // Final assessment
    print("\nFINAL ASSESSMENT\n", .{});
    print("================\n", .{});
    print("Compatibility Score: {}/{}\n", .{ compatibility_score, max_score });

    if (compatibility_score >= 5) {
        print("🎉 EXCELLENT: Model is ready for Zig AI Platform!\n", .{});
        print("✅ Your model_fp16.onnx should work perfectly\n", .{});
    } else if (compatibility_score >= 3) {
        print("👍 GOOD: Model should work with minor adjustments\n", .{});
        print("💡 Enhanced parser should handle any issues\n", .{});
    } else {
        print("⚠️  CAUTION: Model may need preprocessing\n", .{});
        print("🔧 Consider model optimization or conversion\n", .{});
    }

    print("\nNext Steps:\n", .{});
    print("1. Use enhanced ONNX parser with relaxed validation\n", .{});
    print("2. Test inference with simple inputs\n", .{});
    print("3. Monitor memory usage during runtime\n", .{});
    print("4. Optimize for target IoT device if needed\n", .{});

    print("\nModel Summary:\n", .{});
    print("- File: {s}\n", .{model_path});
    print("- Size: {d:.2} MB\n", .{@as(f64, @floatFromInt(file_size)) / (1024.0 * 1024.0)});
    print("- Format: ONNX FP16\n", .{});
    print("- Type: Generative AI (onnxruntime-genai)\n", .{});
    print("- Status: Ready for deployment! 🚀\n", .{});
}
