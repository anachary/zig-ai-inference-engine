const std = @import("std");

/// Supported tokenizer types
pub const TokenizerType = enum {
    bpe,
    sentencepiece,
    tiktoken,
    wordpiece,
    unigram,
    
    pub fn toString(self: TokenizerType) []const u8 {
        return switch (self) {
            .bpe => "BPE",
            .sentencepiece => "SentencePiece",
            .tiktoken => "TikToken",
            .wordpiece => "WordPiece",
            .unigram => "Unigram",
        };
    }
    
    pub fn fromString(name: []const u8) ?TokenizerType {
        if (std.mem.eql(u8, name, "bpe")) return .bpe;
        if (std.mem.eql(u8, name, "sentencepiece")) return .sentencepiece;
        if (std.mem.eql(u8, name, "tiktoken")) return .tiktoken;
        if (std.mem.eql(u8, name, "wordpiece")) return .wordpiece;
        if (std.mem.eql(u8, name, "unigram")) return .unigram;
        return null;
    }
};

/// Detect tokenizer type from file or model metadata
pub fn detectTokenizerType(path: []const u8) TokenizerType {
    // Simple heuristics based on file name/extension
    if (std.mem.indexOf(u8, path, "tokenizer.json") != null) {
        return .bpe; // HuggingFace tokenizers are usually BPE
    }
    if (std.mem.indexOf(u8, path, ".model") != null) {
        return .sentencepiece;
    }
    if (std.mem.indexOf(u8, path, "vocab.txt") != null) {
        return .wordpiece; // BERT-style
    }
    
    // Default to BPE as it's most common
    return .bpe;
}

/// Tokenizer capabilities
pub const TokenizerCapabilities = struct {
    supports_subword: bool,
    supports_special_tokens: bool,
    supports_normalization: bool,
    supports_pretokenization: bool,
    
    pub fn init() TokenizerCapabilities {
        return TokenizerCapabilities{
            .supports_subword = false,
            .supports_special_tokens = false,
            .supports_normalization = false,
            .supports_pretokenization = false,
        };
    }
};

/// Get capabilities for a tokenizer type
pub fn getCapabilities(tokenizer_type: TokenizerType) TokenizerCapabilities {
    return switch (tokenizer_type) {
        .bpe => TokenizerCapabilities{
            .supports_subword = true,
            .supports_special_tokens = true,
            .supports_normalization = true,
            .supports_pretokenization = true,
        },
        .sentencepiece => TokenizerCapabilities{
            .supports_subword = true,
            .supports_special_tokens = true,
            .supports_normalization = true,
            .supports_pretokenization = false,
        },
        .tiktoken => TokenizerCapabilities{
            .supports_subword = true,
            .supports_special_tokens = true,
            .supports_normalization = false,
            .supports_pretokenization = true,
        },
        .wordpiece => TokenizerCapabilities{
            .supports_subword = true,
            .supports_special_tokens = true,
            .supports_normalization = true,
            .supports_pretokenization = true,
        },
        .unigram => TokenizerCapabilities{
            .supports_subword = true,
            .supports_special_tokens = true,
            .supports_normalization = true,
            .supports_pretokenization = false,
        },
    };
}

test "tokenizer type detection" {
    const testing = std.testing;
    
    try testing.expect(detectTokenizerType("tokenizer.json") == .bpe);
    try testing.expect(detectTokenizerType("sentencepiece.model") == .sentencepiece);
    try testing.expect(detectTokenizerType("vocab.txt") == .wordpiece);
    try testing.expect(detectTokenizerType("unknown.file") == .bpe);
}

test "tokenizer capabilities" {
    const testing = std.testing;
    
    const bpe_caps = getCapabilities(.bpe);
    try testing.expect(bpe_caps.supports_subword);
    try testing.expect(bpe_caps.supports_special_tokens);
    
    const wordpiece_caps = getCapabilities(.wordpiece);
    try testing.expect(wordpiece_caps.supports_subword);
    try testing.expect(wordpiece_caps.supports_normalization);
}
