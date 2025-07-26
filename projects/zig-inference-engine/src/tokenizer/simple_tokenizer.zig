const std = @import("std");
const Allocator = std.mem.Allocator;

/// Simple tokenizer for demonstration purposes
/// In a real implementation, this would use BPE, SentencePiece, or similar
pub const SimpleTokenizer = struct {
    allocator: Allocator,
    vocab: std.StringHashMap(u32),
    reverse_vocab: std.ArrayList([]const u8),
    vocab_size: u32,

    const Self = @This();

    /// Initialize the tokenizer with a simple vocabulary
    pub fn init(allocator: Allocator) !Self {
        var tokenizer = Self{
            .allocator = allocator,
            .vocab = std.StringHashMap(u32).init(allocator),
            .reverse_vocab = std.ArrayList([]const u8).init(allocator),
            .vocab_size = 0,
        };

        // Build a simple demonstration vocabulary
        try tokenizer.buildSimpleVocab();

        return tokenizer;
    }

    pub fn deinit(self: *Self) void {
        self.vocab.deinit();
        for (self.reverse_vocab.items) |token| {
            self.allocator.free(token);
        }
        self.reverse_vocab.deinit();
    }

    /// Build a simple vocabulary for demonstration
    fn buildSimpleVocab(self: *Self) !void {
        // Special tokens
        try self.addToken("<pad>"); // 0
        try self.addToken("<unk>"); // 1
        try self.addToken("<bos>"); // 2 - Beginning of sequence
        try self.addToken("<eos>"); // 3 - End of sequence

        // Common words and characters
        const common_tokens = [_][]const u8{
            "hello",  "world",    "how",   "are",     "you",  "what", "is",    "the",   "a",      "an",
            "and",    "or",       "but",   "in",      "on",   "at",   "to",    "for",   "with",   "by",
            "this",   "that",     "these", "those",   "i",    "me",   "my",    "we",    "us",     "our",
            "he",     "him",      "his",   "she",     "her",  "they", "them",  "their", "it",     "its",
            "good",   "bad",      "great", "nice",    "fine", "ok",   "yes",   "no",    "maybe",  "please",
            "thank",  "thanks",   "sorry", "excuse",  "help", "can",  "could", "would", "should", "will",
            "shall",  "do",       "does",  "did",     "have", "has",  "had",   "be",    "am",     "is",
            "was",    "were",     "been",  "being",   "go",   "went", "come",  "came",  "see",    "saw",
            "know",   "knew",     "think", "thought", "say",  "said", "tell",  "told",  "ask",    "asked",
            "answer", "answered", ".",     ",",       "!",    "?",    ":",     ";",     "'",      "\"",
            " ",      "\n",       "\t",
        };

        for (common_tokens) |token| {
            try self.addToken(token);
        }

        // Add numbers 0-9
        for (0..10) |i| {
            const num_str = try std.fmt.allocPrint(self.allocator, "{d}", .{i});
            defer self.allocator.free(num_str); // Free the temporary allocation
            try self.addToken(num_str);
        }

        std.log.info("✅ Built simple vocabulary with {} tokens", .{self.vocab_size});
    }

    fn addToken(self: *Self, token: []const u8) !void {
        const token_copy = try self.allocator.dupe(u8, token);
        try self.vocab.put(token_copy, self.vocab_size);
        try self.reverse_vocab.append(token_copy);
        self.vocab_size += 1;
    }

    /// Tokenize text into token IDs
    pub fn encode(self: *Self, text: []const u8) ![]u32 {
        var tokens = std.ArrayList(u32).init(self.allocator);
        defer tokens.deinit();

        // Add beginning of sequence token
        try tokens.append(2); // <bos>

        // Simple word-based tokenization (split on spaces)
        var word_iter = std.mem.split(u8, text, " ");
        while (word_iter.next()) |word| {
            if (word.len == 0) continue;

            // Clean the word (remove punctuation for simplicity)
            var clean_word = try self.allocator.alloc(u8, word.len);
            defer self.allocator.free(clean_word);

            var clean_len: usize = 0;
            for (word) |char| {
                if (std.ascii.isAlphanumeric(char)) {
                    clean_word[clean_len] = std.ascii.toLower(char);
                    clean_len += 1;
                }
            }

            if (clean_len > 0) {
                const clean_slice = clean_word[0..clean_len];
                const token_id = self.vocab.get(clean_slice) orelse 1; // <unk>
                try tokens.append(token_id);
            }
        }

        // Add end of sequence token
        try tokens.append(3); // <eos>

        return try self.allocator.dupe(u32, tokens.items);
    }

    /// Decode token IDs back to text
    pub fn decode(self: *Self, token_ids: []const u32) ![]u8 {
        var result = std.ArrayList(u8).init(self.allocator);
        defer result.deinit();

        for (token_ids, 0..) |token_id, i| {
            // Skip special tokens in output
            if (token_id == 0 or token_id == 2 or token_id == 3) continue;

            if (token_id < self.vocab_size) {
                const token = self.reverse_vocab.items[token_id];

                if (i > 0 and token_id != 1) { // Add space before non-first, non-unk tokens
                    try result.append(' ');
                }

                try result.appendSlice(token);
            }
        }

        return try self.allocator.dupe(u8, result.items);
    }

    /// Get vocabulary size
    pub fn getVocabSize(self: *const Self) u32 {
        return self.vocab_size;
    }

    /// Get token by ID
    pub fn getToken(self: *const Self, token_id: u32) ?[]const u8 {
        if (token_id < self.vocab_size) {
            return self.reverse_vocab.items[token_id];
        }
        return null;
    }

    /// Get token ID by token string
    pub fn getTokenId(self: *const Self, token: []const u8) ?u32 {
        return self.vocab.get(token);
    }
};

/// Tokenizer errors
pub const TokenizerError = error{
    InvalidToken,
    VocabularyFull,
    EncodingFailed,
    DecodingFailed,
};

// Tests
test "simple tokenizer basic functionality" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var tokenizer = try SimpleTokenizer.init(allocator);
    defer tokenizer.deinit();

    // Test encoding
    const text = "hello world how are you";
    const tokens = try tokenizer.encode(text);
    defer allocator.free(tokens);

    std.log.info("Encoded '{}' to {} tokens", .{ text, tokens.len });

    // Test decoding
    const decoded = try tokenizer.decode(tokens);
    defer allocator.free(decoded);

    std.log.info("Decoded back to: '{s}'", .{decoded});

    // Basic checks
    try std.testing.expect(tokens.len > 0);
    try std.testing.expect(decoded.len > 0);
}
