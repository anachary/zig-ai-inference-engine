const std = @import("std");

// Copy the vocabulary structures from main.zig for testing
const TokenWordPair = struct {
    token: i64,
    word: []const u8,
};

const ModelVocabulary = struct {
    pairs: []TokenWordPair,
    vocab_size: usize,
    allocator: std.mem.Allocator,
    special_tokens: struct {
        bos: i64 = 1,
        eos: i64 = 2,
        pad: i64 = 0,
        unk: i64 = 3,
    },
    
    pub fn init(allocator: std.mem.Allocator) ModelVocabulary {
        return ModelVocabulary{
            .pairs = &[_]TokenWordPair{},
            .vocab_size = 0,
            .allocator = allocator,
            .special_tokens = .{},
        };
    }
    
    pub fn deinit(self: *ModelVocabulary) void {
        if (self.pairs.len > 0) {
            self.allocator.free(self.pairs);
        }
    }
    
    pub fn getWord(self: *const ModelVocabulary, token: i64) ?[]const u8 {
        for (self.pairs) |pair| {
            if (pair.token == token) {
                return pair.word;
            }
        }
        return null;
    }
    
    pub fn addPairs(self: *ModelVocabulary, new_pairs: []const TokenWordPair) !void {
        self.pairs = try self.allocator.dupe(TokenWordPair, new_pairs);
        self.vocab_size = new_pairs.len;
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("🧪 Testing Vocabulary Initialization\n", .{});
    print("=====================================\n", .{});

    // Test 1: Create empty vocabulary
    var vocab = ModelVocabulary.init(allocator);
    defer vocab.deinit();

    print("✅ Step 1: Empty vocabulary created\n", .{});
    print("   Vocab size: {}\n", .{vocab.vocab_size});
    print("   Pairs length: {}\n", .{vocab.pairs.len});

    // Test 2: Add some test tokens
    const test_tokens = [_]TokenWordPair{
        .{ .token = 40, .word = "I" },
        .{ .token = 716, .word = "am" },
        .{ .token = 1049, .word = "fine" },
        .{ .token = 11, .word = "." },
    };

    try vocab.addPairs(&test_tokens);
    print("✅ Step 2: Added {} test tokens\n", .{test_tokens.len});
    print("   Vocab size: {}\n", .{vocab.vocab_size});
    print("   Pairs length: {}\n", .{vocab.pairs.len});

    // Test 3: Test token lookup
    print("✅ Step 3: Testing token lookup\n", .{});
    const test_token_ids = [_]i64{ 40, 716, 1049, 11, 999 };
    
    for (test_token_ids) |token_id| {
        if (vocab.getWord(token_id)) |word| {
            print("   Token {} → '{}'\n", .{ token_id, word });
        } else {
            print("   Token {} → NOT FOUND\n", .{token_id});
        }
    }

    // Test 4: Load enhanced vocabulary like in main.zig
    print("✅ Step 4: Loading enhanced vocabulary\n", .{});
    
    const enhanced_tokens = [_]TokenWordPair{
        // Special tokens
        .{ .token = 0, .word = "<pad>" },
        .{ .token = 1, .word = "<bos>" },
        .{ .token = 2, .word = "<eos>" },
        .{ .token = 3, .word = "<unk>" },
        
        // Common tokens
        .{ .token = 40, .word = "I" },
        .{ .token = 716, .word = "am" },
        .{ .token = 994, .word = "here" },
        .{ .token = 284, .word = "to" },
        .{ .token = 1037, .word = "help" },
        .{ .token = 345, .word = "you" },
        .{ .token = 351, .word = "with" },
        .{ .token = 597, .word = "any" },
        .{ .token = 2683, .word = "questions" },
        .{ .token = 11, .word = "." },
    };

    try vocab.addPairs(&enhanced_tokens);
    print("   Enhanced vocab size: {}\n", .{vocab.vocab_size});
    print("   Enhanced pairs length: {}\n", .{vocab.pairs.len});

    // Test 5: Test the enhanced vocabulary
    print("✅ Step 5: Testing enhanced vocabulary lookup\n", .{});
    const enhanced_test_tokens = [_]i64{ 40, 716, 994, 284, 1037, 345, 11, 999 };
    
    for (enhanced_test_tokens) |token_id| {
        if (vocab.getWord(token_id)) |word| {
            print("   Token {} → '{}'\n", .{ token_id, word });
        } else {
            print("   Token {} → NOT FOUND\n", .{token_id});
        }
    }

    print("\n🎯 Vocabulary Debug Test Complete!\n", .{});
}

const print = std.debug.print;
