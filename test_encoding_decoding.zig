const std = @import("std");
const print = std.debug.print;

// Test encoding and decoding with the actual GPT-2 vocabulary
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("🧪 Testing GPT-2 Encoding/Decoding\n", .{});
    print("===================================\n", .{});

    // Test 1: Load the actual GPT-2 vocabulary
    print("📚 Loading GPT-2 vocabulary...\n", .{});

    const vocab_path = "models/gpt2/onnx/vocab.json";
    const vocab_file = std.fs.cwd().openFile(vocab_path, .{}) catch |err| {
        print("❌ Failed to open vocabulary file: {any}\n", .{err});
        print("   Make sure the file exists at: {s}\n", .{vocab_path});
        return;
    };
    defer vocab_file.close();

    const vocab_content = try vocab_file.readToEndAlloc(allocator, 10 * 1024 * 1024); // 10MB max
    defer allocator.free(vocab_content);

    print("✅ Loaded vocabulary file: {} bytes\n", .{vocab_content.len});

    // Parse JSON vocabulary
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, vocab_content, .{}) catch |err| {
        print("❌ Failed to parse JSON: {any}\n", .{err});
        return;
    };
    defer parsed.deinit();

    const json_obj = parsed.value.object;
    print("✅ Parsed vocabulary: {} entries\n", .{json_obj.count()});

    // Test 2: Test specific token mappings
    print("\n🔍 Testing specific token mappings:\n", .{});

    const test_words = [_][]const u8{ "Hello", "world", "I", "am", "the", ".", "?", "what", "is", "artificial", "intelligence" };

    for (test_words) |word| {
        if (json_obj.get(word)) |token_value| {
            const token_id = switch (token_value) {
                .integer => |int_val| @as(i64, @intCast(int_val)),
                .float => |float_val| @as(i64, @intFromFloat(float_val)),
                else => -1,
            };
            print("   '{s}' -> token {d}\n", .{ word, token_id });
        } else {
            print("   '{s}' -> NOT FOUND\n", .{word});
        }
    }

    // Test 3: Test reverse mapping (token to word)
    print("\n🔄 Testing reverse mapping (token -> word):\n", .{});

    const test_tokens = [_]i64{ 15496, 995, 314, 716, 262, 13, 30, 644, 318, 11666, 4430 };

    var iterator = json_obj.iterator();
    for (test_tokens) |target_token| {
        var found = false;
        iterator = json_obj.iterator(); // Reset iterator
        while (iterator.next()) |entry| {
            const word = entry.key_ptr.*;
            const token_value = entry.value_ptr.*;

            const token_id = switch (token_value) {
                .integer => |int_val| @as(i64, @intCast(int_val)),
                .float => |float_val| @as(i64, @intFromFloat(float_val)),
                else => continue,
            };

            if (token_id == target_token) {
                print("   token {d} -> '{s}'\n", .{ target_token, word });
                found = true;
                break;
            }
        }
        if (!found) {
            print("   token {d} -> NOT FOUND\n", .{target_token});
        }
    }

    // Test 4: Test sentence tokenization
    print("\n📝 Testing sentence tokenization:\n", .{});

    const test_sentences = [_][]const u8{ "Hello world", "What is artificial intelligence?", "I am fine.", "How are you today?" };

    for (test_sentences) |sentence| {
        print("   Input: '{s}'\n", .{sentence});

        // Simple word-based tokenization (like current implementation)
        var word_iter = std.mem.split(u8, sentence, " ");
        var tokens = std.ArrayList(i64).init(allocator);
        defer tokens.deinit();

        while (word_iter.next()) |word| {
            if (word.len == 0) continue;

            // Clean word (remove punctuation for lookup)
            var clean_word = std.ArrayList(u8).init(allocator);
            defer clean_word.deinit();

            for (word) |char| {
                if (std.ascii.isAlphanumeric(char)) {
                    try clean_word.append(char);
                }
            }

            const clean_word_str = clean_word.items;

            if (json_obj.get(clean_word_str)) |token_value| {
                const token_id = switch (token_value) {
                    .integer => |int_val| @as(i64, @intCast(int_val)),
                    .float => |float_val| @as(i64, @intFromFloat(float_val)),
                    else => -1,
                };
                try tokens.append(token_id);
                print("     '{s}' -> {d}\n", .{ clean_word_str, token_id });
            } else {
                print("     '{s}' -> NOT FOUND (would use hash fallback)\n", .{clean_word_str});
                // Hash fallback like in current implementation
                var hash: u32 = 0;
                for (clean_word_str) |char| {
                    hash = hash *% 31 +% char;
                }
                const fallback_token = @as(i64, @intCast(hash % 50000)) + 100;
                try tokens.append(fallback_token);
                print("     '{s}' -> {d} (hash fallback)\n", .{ clean_word_str, fallback_token });
            }
        }

        print("   Tokens: ", .{});
        for (tokens.items) |token| {
            print("{d} ", .{token});
        }
        print("\n\n", .{});
    }

    print("✅ Encoding/Decoding test completed!\n", .{});
}
