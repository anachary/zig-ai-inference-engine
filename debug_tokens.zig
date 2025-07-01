const std = @import("std");
const print = std.debug.print;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("🔍 Debugging specific tokens\n", .{});

    const vocab_path = "models/gpt2/onnx/vocab.json";
    const vocab_file = std.fs.cwd().openFile(vocab_path, .{}) catch |err| {
        print("❌ Failed to open vocabulary file: {any}\n", .{err});
        return;
    };
    defer vocab_file.close();

    const vocab_content = try vocab_file.readToEndAlloc(allocator, 10 * 1024 * 1024);
    defer allocator.free(vocab_content);

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, vocab_content, .{}) catch |err| {
        print("❌ Failed to parse JSON: {any}\n", .{err});
        return;
    };
    defer parsed.deinit();

    const json_obj = parsed.value.object;

    // Debug the specific tokens we're seeing
    const debug_tokens = [_]i64{ 15496, 3, 24, 6, 5, 11 };

    print("\n🔍 Looking up specific tokens:\n", .{});

    // Also check what "world" should be
    print("\n🔍 Looking for 'world' and ' world' tokens:\n", .{});
    var iterator2 = json_obj.iterator();
    while (iterator2.next()) |entry| {
        const word = entry.key_ptr.*;
        if (std.mem.eql(u8, word, "world") or std.mem.eql(u8, word, " world")) {
            const token_value = entry.value_ptr.*;
            const token_id = switch (token_value) {
                .integer => |int_val| @as(i64, @intCast(int_val)),
                .float => |float_val| @as(i64, @intFromFloat(float_val)),
                else => continue,
            };
            print("   Found '{s}' -> token {d}\n", .{ word, token_id });
        }
    }

    for (debug_tokens) |target_token| {
        var found = false;
        var iterator = json_obj.iterator();
        while (iterator.next()) |entry| {
            const word = entry.key_ptr.*;
            const token_value = entry.value_ptr.*;

            const token_id = switch (token_value) {
                .integer => |int_val| @as(i64, @intCast(int_val)),
                .float => |float_val| @as(i64, @intFromFloat(float_val)),
                else => continue,
            };

            if (token_id == target_token) {
                print("   Token {d} -> '{s}' (length: {d})\n", .{ target_token, word, word.len });

                // Show character codes for debugging
                print("     Character codes: ", .{});
                for (word) |char| {
                    print("{d} ", .{char});
                }
                print("\n", .{});

                found = true;
                break;
            }
        }
        if (!found) {
            print("   Token {d} -> NOT FOUND\n", .{target_token});
        }
    }

    print("\n✅ Token lookup completed!\n", .{});
}
