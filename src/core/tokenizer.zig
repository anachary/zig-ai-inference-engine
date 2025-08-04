const std = @import("std");

/// Token ID type
pub const TokenId = u32;

/// Special token types
pub const SpecialTokens = struct {
    bos: ?TokenId = null, // Beginning of sequence
    eos: ?TokenId = null, // End of sequence
    unk: ?TokenId = null, // Unknown token
    pad: ?TokenId = null, // Padding token

    pub fn init() SpecialTokens {
        return SpecialTokens{};
    }
};

/// Tokenization result
pub const TokenizeResult = struct {
    tokens: []TokenId,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *TokenizeResult) void {
        self.allocator.free(self.tokens);
    }
};

/// Universal tokenizer interface
pub const Tokenizer = struct {
    vocab_size: u32,
    special_tokens: SpecialTokens,
    allocator: std.mem.Allocator,

    // Virtual function table for tokenizer-specific operations
    vtable: *const VTable,
    impl: *anyopaque,

    pub const VTable = struct {
        deinit: *const fn (impl: *anyopaque, allocator: std.mem.Allocator) void,
        encode: *const fn (impl: *anyopaque, text: []const u8, allocator: std.mem.Allocator) anyerror!TokenizeResult,
        decode: *const fn (impl: *anyopaque, tokens: []const TokenId, allocator: std.mem.Allocator) anyerror![]u8,
        tokenToString: *const fn (impl: *anyopaque, token: TokenId, allocator: std.mem.Allocator) anyerror![]u8,
        stringToToken: *const fn (impl: *anyopaque, string: []const u8) ?TokenId,
    };

    pub fn init(allocator: std.mem.Allocator, vocab_size: u32, special_tokens: SpecialTokens, vtable: *const VTable, impl: *anyopaque) Tokenizer {
        return Tokenizer{
            .vocab_size = vocab_size,
            .special_tokens = special_tokens,
            .allocator = allocator,
            .vtable = vtable,
            .impl = impl,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        self.vtable.deinit(self.impl, self.allocator);
    }

    /// Encode text to tokens
    pub fn encode(self: *Tokenizer, text: []const u8) !TokenizeResult {
        return self.vtable.encode(self.impl, text, self.allocator);
    }

    /// Decode tokens to text
    pub fn decode(self: *Tokenizer, tokens: []const TokenId) ![]u8 {
        return self.vtable.decode(self.impl, tokens, self.allocator);
    }

    /// Convert single token to string
    pub fn tokenToString(self: *Tokenizer, token: TokenId) ![]u8 {
        return self.vtable.tokenToString(self.impl, token, self.allocator);
    }

    /// Convert string to token (if exact match exists)
    pub fn stringToToken(self: *Tokenizer, string: []const u8) ?TokenId {
        return self.vtable.stringToToken(self.impl, string);
    }

    /// Encode with special tokens
    pub fn encodeWithSpecial(self: *Tokenizer, text: []const u8, add_bos: bool, add_eos: bool) !TokenizeResult {
        var result = try self.encode(text);

        if (!add_bos and !add_eos) {
            return result;
        }

        // Calculate new size
        var new_size = result.tokens.len;
        if (add_bos and self.special_tokens.bos != null) new_size += 1;
        if (add_eos and self.special_tokens.eos != null) new_size += 1;

        // Create new token array
        var new_tokens = try self.allocator.alloc(TokenId, new_size);
        var offset: usize = 0;

        // Add BOS token
        if (add_bos and self.special_tokens.bos != null) {
            new_tokens[offset] = self.special_tokens.bos.?;
            offset += 1;
        }

        // Copy original tokens
        @memcpy(new_tokens[offset .. offset + result.tokens.len], result.tokens);
        offset += result.tokens.len;

        // Add EOS token
        if (add_eos and self.special_tokens.eos != null) {
            new_tokens[offset] = self.special_tokens.eos.?;
        }

        // Clean up old result
        result.deinit();

        return TokenizeResult{
            .tokens = new_tokens,
            .allocator = self.allocator,
        };
    }

    /// Check if token is special
    pub fn isSpecialToken(self: *Tokenizer, token: TokenId) bool {
        return (self.special_tokens.bos != null and token == self.special_tokens.bos.?) or
            (self.special_tokens.eos != null and token == self.special_tokens.eos.?) or
            (self.special_tokens.unk != null and token == self.special_tokens.unk.?) or
            (self.special_tokens.pad != null and token == self.special_tokens.pad.?);
    }

    /// Get token type
    pub fn getTokenType(self: *Tokenizer, token: TokenId) TokenType {
        if (self.special_tokens.bos != null and token == self.special_tokens.bos.?) return .bos;
        if (self.special_tokens.eos != null and token == self.special_tokens.eos.?) return .eos;
        if (self.special_tokens.unk != null and token == self.special_tokens.unk.?) return .unk;
        if (self.special_tokens.pad != null and token == self.special_tokens.pad.?) return .pad;
        return .normal;
    }
};

pub const TokenType = enum {
    normal,
    bos,
    eos,
    unk,
    pad,
};

/// Vocabulary entry
pub const VocabEntry = struct {
    token: []const u8,
    score: f32,
    type: TokenType,

    pub fn init(token: []const u8, score: f32, token_type: TokenType) VocabEntry {
        return VocabEntry{
            .token = token,
            .score = score,
            .type = token_type,
        };
    }
};

/// Vocabulary management
pub const Vocabulary = struct {
    entries: []VocabEntry,
    token_to_id: std.StringHashMap(TokenId),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, entries: []VocabEntry) !Vocabulary {
        var token_to_id = std.StringHashMap(TokenId).init(allocator);

        for (entries, 0..) |entry, i| {
            try token_to_id.put(entry.token, @intCast(i));
        }

        return Vocabulary{
            .entries = entries,
            .token_to_id = token_to_id,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Vocabulary) void {
        self.token_to_id.deinit();
        for (self.entries) |entry| {
            self.allocator.free(entry.token);
        }
        self.allocator.free(self.entries);
    }

    pub fn getTokenId(self: *Vocabulary, token: []const u8) ?TokenId {
        return self.token_to_id.get(token);
    }

    pub fn getToken(self: *Vocabulary, id: TokenId) ?[]const u8 {
        if (id >= self.entries.len) return null;
        return self.entries[id].token;
    }

    pub fn size(self: *Vocabulary) u32 {
        return @intCast(self.entries.len);
    }
};

test "special tokens" {
    const testing = std.testing;

    var special = SpecialTokens.init();
    special.bos = 1;
    special.eos = 2;

    try testing.expect(special.bos.? == 1);
    try testing.expect(special.eos.? == 2);
    try testing.expect(special.unk == null);
}

test "vocabulary" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var entries = try allocator.alloc(VocabEntry, 3);
    entries[0] = VocabEntry.init(try allocator.dupe(u8, "hello"), 0.0, .normal);
    entries[1] = VocabEntry.init(try allocator.dupe(u8, "world"), 0.0, .normal);
    entries[2] = VocabEntry.init(try allocator.dupe(u8, "<eos>"), 0.0, .eos);

    var vocab = try Vocabulary.init(allocator, entries);
    defer vocab.deinit();

    try testing.expect(vocab.getTokenId("hello").? == 0);
    try testing.expect(vocab.getTokenId("world").? == 1);
    try testing.expect(vocab.getTokenId("<eos>").? == 2);
    try testing.expect(vocab.getTokenId("unknown") == null);

    try testing.expectEqualStrings(vocab.getToken(0).?, "hello");
    try testing.expectEqualStrings(vocab.getToken(1).?, "world");
}
