const std = @import("std");

// Helper functions for reading data
const DataReader = struct {
    pub fn readU32(data: []const u8, offset: usize) u32 {
        return std.mem.readIntLittle(u32, data[offset .. offset + 4][0..4]);
    }
    
    pub fn readU64(data: []const u8, offset: usize) u64 {
        return std.mem.readIntLittle(u64, data[offset .. offset + 8][0..8]);
    }
    
    pub fn readI32(data: []const u8, offset: usize) i32 {
        return std.mem.readIntLittle(i32, data[offset .. offset + 4][0..4]);
    }
    
    pub fn readI64(data: []const u8, offset: usize) i64 {
        return std.mem.readIntLittle(i64, data[offset .. offset + 8][0..8]);
    }
};

const CleanGGUFParser = struct {
    allocator: std.mem.Allocator,
    
    // Model parameters
    vocab_size: u32 = 0,
    hidden_size: u32 = 0,
    num_layers: u32 = 0,
    num_heads: u32 = 0,
    context_length: u32 = 0,
    
    pub fn init(allocator: std.mem.Allocator) CleanGGUFParser {
        return CleanGGUFParser{
            .allocator = allocator,
        };
    }
    
    /// Parse GGUF file with clean type-first approach
    pub fn parseFile(self: *CleanGGUFParser, file_path: []const u8) !void {
        // Read file
        const file = try std.fs.cwd().openFile(file_path, .{});
        defer file.close();
        
        const file_size = try file.getEndPos();
        const data = try self.allocator.alloc(u8, file_size);
        defer self.allocator.free(data);
        _ = try file.readAll(data);
        
        std.log.info("🎯 CLEAN GGUF PARSER - Type-First Approach");
        
        // Parse header
        var offset: usize = 0;
        
        // Magic
        const magic = DataReader.readU32(data, offset);
        offset += 4;
        if (magic != 0x46554747) {
            return error.InvalidMagic;
        }
        std.log.info("✅ Valid GGUF magic");
        
        // Version
        const version = DataReader.readU32(data, offset);
        offset += 4;
        std.log.info("📋 Version: {d}", .{version});
        
        // Tensor count
        const tensor_count = DataReader.readU64(data, offset);
        offset += 8;
        std.log.info("📦 Tensor count: {d}", .{tensor_count});
        
        // Metadata count
        const metadata_count = DataReader.readU64(data, offset);
        offset += 8;
        std.log.info("📊 Metadata count: {d}", .{metadata_count});
        
        // Parse metadata with type-first approach
        offset = try self.parseMetadata(data, offset, metadata_count);
        
        std.log.info("🎉 PARSING COMPLETE!");
        std.log.info("  Vocab size: {d}", .{self.vocab_size});
        std.log.info("  Hidden size: {d}", .{self.hidden_size});
        std.log.info("  Layers: {d}", .{self.num_layers});
        std.log.info("  Heads: {d}", .{self.num_heads});
    }
    
    /// Parse metadata with clean type-first approach
    fn parseMetadata(self: *CleanGGUFParser, data: []const u8, start_offset: usize, count: u64) !usize {
        var offset = start_offset;
        
        for (0..count) |i| {
            std.log.debug("Entry {d}/{d} at offset {d}", .{ i + 1, count, offset });
            
            // Step 1: Read entry header
            const key_len = DataReader.readU64(data, offset);
            offset += 8;
            
            if (key_len > 1024) {
                std.log.err("Invalid key length {d}", .{key_len});
                return error.InvalidMetadata;
            }
            
            const key = data[offset .. offset + key_len];
            offset += key_len;
            
            const value_type = DataReader.readU32(data, offset);
            offset += 4;
            
            std.log.debug("  Key: '{s}', Type: {d}", .{ key, value_type });
            
            // Step 2: Handle value based on type
            offset = try self.handleValue(data, offset, key, value_type);
        }
        
        return offset;
    }
    
    /// Handle value based on type with smart allocation
    fn handleValue(self: *CleanGGUFParser, data: []const u8, offset: usize, key: []const u8, value_type: u32) !usize {
        var new_offset = offset;
        
        switch (value_type) {
            4, 5, 6, 7 => { // Integer types
                const value = try self.readInteger(data, new_offset, value_type);
                new_offset += switch (value_type) {
                    4, 5 => 4,
                    6, 7 => 8,
                    else => 0,
                };
                
                // Extract values we care about
                if (std.mem.indexOf(u8, key, "vocab_size") != null) {
                    self.vocab_size = value;
                    std.log.info("  ✅ Vocab size: {d}", .{value});
                } else if (std.mem.indexOf(u8, key, "embedding_length") != null or 
                          std.mem.indexOf(u8, key, "n_embd") != null) {
                    self.hidden_size = value;
                    std.log.info("  ✅ Hidden size: {d}", .{value});
                } else if (std.mem.indexOf(u8, key, "block_count") != null or 
                          std.mem.indexOf(u8, key, "n_layer") != null) {
                    self.num_layers = value;
                    std.log.info("  ✅ Layer count: {d}", .{value});
                } else if (std.mem.indexOf(u8, key, "head_count") != null or 
                          std.mem.indexOf(u8, key, "n_head") != null) {
                    self.num_heads = value;
                    std.log.info("  ✅ Attention heads: {d}", .{value});
                }
                
                // Handle UINT64 padding
                if (value_type == 6 or value_type == 7) {
                    if (new_offset + 4 <= data.len) {
                        const potential_padding = DataReader.readU32(data, new_offset);
                        if (potential_padding == 0) {
                            new_offset += 4;
                            std.log.debug("  🔧 Applied UINT64 padding", .{});
                        }
                    }
                }
            },
            8 => { // STRING
                const str_len = DataReader.readU64(data, new_offset);
                new_offset += 8;
                const str_value = data[new_offset .. new_offset + str_len];
                new_offset += str_len;
                std.log.debug("  📝 String '{s}': '{s}'", .{ key, str_value });
            },
            else => {
                // Skip unknown types
                const skip_size = switch (value_type) {
                    0, 1 => 1,
                    2, 3 => 2,
                    9 => 4,
                    10 => 8,
                    11 => 1,
                    else => 0,
                };
                new_offset += skip_size;
                std.log.debug("  ⏭️  Skipped type {d} ({d} bytes)", .{ value_type, skip_size });
            },
        }
        
        return new_offset;
    }
    
    /// Read integer value
    fn readInteger(self: *CleanGGUFParser, data: []const u8, offset: usize, value_type: u32) !u32 {
        _ = self;
        return switch (value_type) {
            4 => DataReader.readU32(data, offset),
            5 => @as(u32, @bitCast(DataReader.readI32(data, offset))),
            6 => @as(u32, @truncate(DataReader.readU64(data, offset))),
            7 => @as(u32, @truncate(@as(u64, @bitCast(DataReader.readI64(data, offset))))),
            else => 0,
        };
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    var parser = CleanGGUFParser.init(allocator);
    try parser.parseFile("models/Qwen2-0.5B-Instruct-Q4_K_M.gguf");
}
