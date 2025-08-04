const std = @import("std");
const Tokenizer = @import("../../core/tokenizer.zig").Tokenizer;
const TokenId = @import("../../core/tokenizer.zig").TokenId;
const TokenizeResult = @import("../../core/tokenizer.zig").TokenizeResult;
const SpecialTokens = @import("../../core/tokenizer.zig").SpecialTokens;
const Vocabulary = @import("../../core/tokenizer.zig").Vocabulary;
const VocabEntry = @import("../../core/tokenizer.zig").VocabEntry;
const TokenType = @import("../../core/tokenizer.zig").TokenType;

/// BPE merge rule
const MergeRule = struct {
    left: []const u8,
    right: []const u8,
    merged: []const u8,
    priority: u32,

    pub fn init(left: []const u8, right: []const u8, merged: []const u8, priority: u32) MergeRule {
        return MergeRule{
            .left = left,
            .right = right,
            .merged = merged,
            .priority = priority,
        };
    }
};

/// BPE tokenizer implementation
pub const BPETokenizer = struct {
    vocabulary: Vocabulary,
    merges: []MergeRule,
    special_tokens: SpecialTokens,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, vocab_entries: []VocabEntry, merges: []MergeRule, special_tokens: SpecialTokens) !BPETokenizer {
        const vocabulary = try Vocabulary.init(allocator, vocab_entries);

        return BPETokenizer{
            .vocabulary = vocabulary,
            .merges = merges,
            .special_tokens = special_tokens,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BPETokenizer) void {
        self.vocabulary.deinit();

        // Clean up merges
        for (self.merges) |merge| {
            self.allocator.free(merge.left);
            self.allocator.free(merge.right);
            self.allocator.free(merge.merged);
        }
        self.allocator.free(self.merges);
    }

    pub fn encode(self: *BPETokenizer, text: []const u8) !TokenizeResult {
        // Simple BPE encoding implementation
        // 1. Split text into characters
        var tokens = std.ArrayList([]const u8).init(self.allocator);
        defer tokens.deinit();

        // Start with character-level tokens
        var i: usize = 0;
        while (i < text.len) {
            const char_len = std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
            const char = text[i .. i + char_len];
            try tokens.append(try self.allocator.dupe(u8, char));
            i += char_len;
        }

        // Apply BPE merges
        var changed = true;
        while (changed) {
            changed = false;

            // Find the highest priority merge that can be applied
            var best_merge: ?MergeRule = null;
            var best_pos: usize = 0;

            for (self.merges) |merge| {
                var pos: usize = 0;
                while (pos + 1 < tokens.items.len) {
                    if (std.mem.eql(u8, tokens.items[pos], merge.left) and
                        std.mem.eql(u8, tokens.items[pos + 1], merge.right))
                    {
                        if (best_merge == null or merge.priority < best_merge.?.priority) {
                            best_merge = merge;
                            best_pos = pos;
                        }
                    }
                    pos += 1;
                }
            }

            // Apply the best merge
            if (best_merge) |merge| {
                // Free the old tokens
                self.allocator.free(tokens.items[best_pos]);
                self.allocator.free(tokens.items[best_pos + 1]);

                // Replace with merged token
                tokens.items[best_pos] = try self.allocator.dupe(u8, merge.merged);

                // Remove the second token
                _ = tokens.orderedRemove(best_pos + 1);

                changed = true;
            }
        }

        // Convert tokens to IDs
        var token_ids = std.ArrayList(TokenId).init(self.allocator);
        defer token_ids.deinit();

        for (tokens.items) |token| {
            if (self.vocabulary.getTokenId(token)) |id| {
                try token_ids.append(id);
            } else {
                // Use UNK token if available
                if (self.special_tokens.unk) |unk_id| {
                    try token_ids.append(unk_id);
                }
            }
            self.allocator.free(token);
        }

        return TokenizeResult{
            .tokens = try token_ids.toOwnedSlice(),
            .allocator = self.allocator,
        };
    }

    pub fn decode(self: *BPETokenizer, tokens: []const TokenId) ![]u8 {
        var result = std.ArrayList(u8).init(self.allocator);
        defer result.deinit();

        for (tokens) |token_id| {
            if (self.vocabulary.getToken(token_id)) |token| {
                try result.appendSlice(token);
            }
        }

        return result.toOwnedSlice();
    }

    pub fn tokenToString(self: *BPETokenizer, token: TokenId) ![]u8 {
        if (self.vocabulary.getToken(token)) |token_str| {
            return self.allocator.dupe(u8, token_str);
        }
        return error.InvalidToken;
    }

    pub fn stringToToken(self: *BPETokenizer, string: []const u8) ?TokenId {
        return self.vocabulary.getTokenId(string);
    }
};

/// Create BPE tokenizer from vocabulary file
pub fn create(allocator: std.mem.Allocator, vocab_path: []const u8) !Tokenizer {
    _ = vocab_path; // Unused for now
    // Load vocabulary (simplified implementation)
    var vocab_entries = std.ArrayList(VocabEntry).init(allocator);
    defer vocab_entries.deinit();

    // Add some basic tokens for testing
    try vocab_entries.append(VocabEntry.init(try allocator.dupe(u8, "<unk>"), 0.0, .unk));
    try vocab_entries.append(VocabEntry.init(try allocator.dupe(u8, "<s>"), 0.0, .bos));
    try vocab_entries.append(VocabEntry.init(try allocator.dupe(u8, "</s>"), 0.0, .eos));
    try vocab_entries.append(VocabEntry.init(try allocator.dupe(u8, "hello"), 0.0, .normal));
    try vocab_entries.append(VocabEntry.init(try allocator.dupe(u8, "world"), 0.0, .normal));
    try vocab_entries.append(VocabEntry.init(try allocator.dupe(u8, "!"), 0.0, .normal));

    // Create special tokens
    var special_tokens = SpecialTokens.init();
    special_tokens.unk = 0;
    special_tokens.bos = 1;
    special_tokens.eos = 2;

    // Create empty merges for now
    var merges = try allocator.alloc(MergeRule, 0);

    // Create BPE tokenizer
    var bpe_tokenizer = try allocator.create(BPETokenizer);
    bpe_tokenizer.* = try BPETokenizer.init(
        allocator,
        try vocab_entries.toOwnedSlice(),
        merges,
        special_tokens,
    );

    // Create vtable
    const vtable = &Tokenizer.VTable{
        .deinit = bpeDeinit,
        .encode = bpeEncode,
        .decode = bpeDecode,
        .tokenToString = bpeTokenToString,
        .stringToToken = bpeStringToToken,
    };

    return Tokenizer.init(
        allocator,
        @intCast(vocab_entries.items.len),
        special_tokens,
        vtable,
        bpe_tokenizer,
    );
}

/// Load BPE tokenizer from HuggingFace tokenizer.json format
pub fn loadFromJson(allocator: std.mem.Allocator, json_path: []const u8) !Tokenizer {
    // This would parse the HuggingFace tokenizer.json format
    // For now, use the simple create function
    return create(allocator, json_path);
}

// VTable implementations
fn bpeDeinit(impl: *anyopaque, allocator: std.mem.Allocator) void {
    const bpe_tokenizer: *BPETokenizer = @ptrCast(@alignCast(impl));
    bpe_tokenizer.deinit();
    allocator.destroy(bpe_tokenizer);
}

fn bpeEncode(impl: *anyopaque, text: []const u8, allocator: std.mem.Allocator) anyerror!TokenizeResult {
    _ = allocator; // The BPE tokenizer uses its own allocator
    const bpe_tokenizer: *BPETokenizer = @ptrCast(@alignCast(impl));
    return bpe_tokenizer.encode(text);
}

fn bpeDecode(impl: *anyopaque, tokens: []const TokenId, allocator: std.mem.Allocator) anyerror![]u8 {
    _ = allocator; // The BPE tokenizer uses its own allocator
    const bpe_tokenizer: *BPETokenizer = @ptrCast(@alignCast(impl));
    return bpe_tokenizer.decode(tokens);
}

fn bpeTokenToString(impl: *anyopaque, token: TokenId, allocator: std.mem.Allocator) anyerror![]u8 {
    _ = allocator; // The BPE tokenizer uses its own allocator
    const bpe_tokenizer: *BPETokenizer = @ptrCast(@alignCast(impl));
    return bpe_tokenizer.tokenToString(token);
}

fn bpeStringToToken(impl: *anyopaque, string: []const u8) ?TokenId {
    const bpe_tokenizer: *BPETokenizer = @ptrCast(@alignCast(impl));
    return bpe_tokenizer.stringToToken(string);
}

test "bpe tokenizer creation" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var tokenizer = try create(allocator, "test");
    defer tokenizer.deinit();

    try testing.expect(tokenizer.vocab_size > 0);
    try testing.expect(tokenizer.special_tokens.unk != null);
}

test "bpe encoding" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var tokenizer = try create(allocator, "test");
    defer tokenizer.deinit();

    var result = try tokenizer.encode("hello world!");
    defer result.deinit();

    try testing.expect(result.tokens.len > 0);
}
