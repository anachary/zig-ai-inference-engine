const std = @import("std");

// Helper functions for reading data
const DataReader = struct {
    pub fn readU32(data: []const u8, offset: usize) u32 {
        return std.mem.readIntLittle(u32, data[offset .. offset + 4][0..4]);
    }

    pub fn readU64(data: []const u8, offset: usize) u64 {
        return std.mem.readIntLittle(u64, data[offset .. offset + 8][0..8]);
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const model_path = "models/Qwen2-0.5B-Instruct-Q4_K_M.gguf";

    // Read the file
    const file = std.fs.cwd().openFile(model_path, .{}) catch |err| {
        std.log.err("Cannot open file: {}", .{err});
        return;
    };
    defer file.close();

    const file_size = try file.getEndPos();
    const data = try allocator.alloc(u8, file_size);
    defer allocator.free(data);
    _ = try file.readAll(data);

    std.log.info("🔍 DEEP GGUF STRUCTURE ANALYSIS", .{});
    std.log.info("==================================================", .{});

    // Parse header
    var offset: usize = 0;

    // Magic
    const magic = DataReader.readU32(data, offset);
    offset += 4;
    std.log.info("Magic: 0x{X} ({s})", .{ magic, if (magic == 0x46554747) "GGUF" else "INVALID" });

    // Version
    const version = DataReader.readU32(data, offset);
    offset += 4;
    std.log.info("Version: {}", .{version});

    // Tensor count
    const tensor_count = DataReader.readU64(data, offset);
    offset += 8;
    std.log.info("Tensor count: {}", .{tensor_count});

    // Metadata count
    const metadata_count = DataReader.readU64(data, offset);
    offset += 8;
    std.log.info("Metadata count: {}", .{metadata_count});

    std.log.info("📊 METADATA ENTRIES ANALYSIS:", .{});
    std.log.info("--------------------------------------------------", .{});

    // Analyze each metadata entry in detail
    for (0..@min(metadata_count, 15)) |i| { // Limit to first 15 entries
        std.log.info("🔍 ENTRY {d} at offset {d}:", .{ i + 1, offset });

        // Show raw bytes
        const debug_end = @min(offset + 64, data.len);
        std.log.info("Raw bytes: {any}", .{data[offset..debug_end]});

        // Read key length
        const key_len = DataReader.readU64(data, offset);
        std.log.info("Key length: {d} (0x{X})", .{ key_len, key_len });

        if (key_len > 1000) {
            std.log.err("❌ INVALID KEY LENGTH! This is where parsing breaks.", .{});
            std.log.info("Expected key length should be < 100 for metadata keys", .{});

            // Let's backtrack and see what went wrong
            std.log.info("🔙 BACKTRACKING ANALYSIS:", .{});
            if (i > 0) {
                std.log.info("Previous entry ended at offset: {d}", .{offset});
                std.log.info("Previous 32 bytes: {any}", .{data[offset - 32 .. offset]});

                // Try to find the pattern
                std.log.info("🔍 SEARCHING FOR VALID KEY LENGTH PATTERN:", .{});
                for (0..32) |back| {
                    const test_offset = offset - back;
                    if (test_offset >= 8) {
                        const test_key_len = DataReader.readU64(data, test_offset);
                        if (test_key_len > 0 and test_key_len < 100) {
                            std.log.info("  Potential valid key length {d} at offset {d} (back {d} bytes)", .{ test_key_len, test_offset, back });
                        }
                    }
                }
            }
            break;
        }

        offset += 8;

        // Read key
        if (offset + key_len > data.len) {
            std.log.err("Key extends beyond file!", .{});
            break;
        }

        const key = data[offset .. offset + key_len];
        std.log.info("Key: '{s}'", .{key});
        offset += key_len;

        // Read value type
        const value_type = DataReader.readU32(data, offset);
        std.log.info("Value type: {d}", .{value_type});
        offset += 4;

        // Analyze value based on type
        std.log.info("Value analysis:", .{});
        switch (value_type) {
            4 => { // UINT32
                const value = DataReader.readU32(data, offset);
                std.log.info("  UINT32 value: {d}", .{value});
                offset += 4;
            },
            6 => { // UINT64
                const value = DataReader.readU64(data, offset);
                std.log.info("  UINT64 value: {d}", .{value});
                offset += 8;

                // Check if there's padding after UINT64
                std.log.info("  Bytes after UINT64: {any}", .{data[offset..@min(offset + 8, data.len)]});

                // Check for 4-byte padding
                if (offset + 4 <= data.len) {
                    const padding = DataReader.readU32(data, offset);
                    if (padding == 0) {
                        std.log.info("  Found 4-byte padding after UINT64, skipping", .{});
                        offset += 4;
                    }
                }
            },
            8 => { // STRING
                const str_len = DataReader.readU64(data, offset);
                std.log.info("  String length: {d}", .{str_len});
                offset += 8;

                if (str_len < 1000 and offset + str_len <= data.len) {
                    const str_value = data[offset .. offset + str_len];
                    std.log.info("  String value: '{s}'", .{str_value});
                    offset += str_len;
                } else {
                    std.log.err("  Invalid string length!", .{});
                    break;
                }
            },
            else => {
                std.log.warn("  Unknown value type {d} - cannot parse further", .{value_type});
                break;
            },
        }

        std.log.info("Next offset: {d}", .{offset});
    }

    std.log.info("🎯 ROOT CAUSE ANALYSIS COMPLETE", .{});
}
