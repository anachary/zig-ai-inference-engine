const std = @import("std");
const Allocator = std.mem.Allocator;

/// Lightweight protobuf parser specifically for ONNX format
/// This is a custom implementation to avoid external dependencies
pub const ProtobufParser = struct {
    allocator: Allocator,
    data: []const u8,
    pos: usize,

    const Self = @This();

    pub const ProtobufError = error{
        InvalidWireType,
        UnexpectedEndOfData,
        InvalidVarint,
        InvalidString,
        UnsupportedFieldType,
        OutOfMemory,
    };

    pub const WireType = enum(u3) {
        varint = 0,
        fixed64 = 1,
        length_delimited = 2,
        start_group = 3, // Deprecated
        end_group = 4, // Deprecated
        fixed32 = 5,
        _,
    };

    pub const FieldHeader = struct {
        field_number: u32,
        wire_type: WireType,
    };

    pub fn init(allocator: Allocator, data: []const u8) Self {
        return Self{
            .allocator = allocator,
            .data = data,
            .pos = 0,
        };
    }

    pub fn hasMoreData(self: *const Self) bool {
        return self.pos < self.data.len;
    }

    pub fn readFieldHeader(self: *Self) ProtobufError!FieldHeader {
        const tag = try self.readVarint();
        const wire_type = @as(WireType, @enumFromInt(@as(u3, @truncate(tag & 0x7))));
        const field_number = @as(u32, @truncate(tag >> 3));

        return FieldHeader{
            .field_number = field_number,
            .wire_type = wire_type,
        };
    }

    pub fn readVarint(self: *Self) ProtobufError!u64 {
        var result: u64 = 0;
        var shift: u6 = 0;

        while (self.pos < self.data.len) {
            const byte = self.data[self.pos];
            self.pos += 1;

            result |= @as(u64, byte & 0x7F) << shift;

            if ((byte & 0x80) == 0) {
                return result;
            }

            shift += 7;
            if (shift >= 64) {
                return ProtobufError.InvalidVarint;
            }
        }

        return ProtobufError.UnexpectedEndOfData;
    }

    pub fn readFixed32(self: *Self) ProtobufError!u32 {
        if (self.pos + 4 > self.data.len) {
            return ProtobufError.UnexpectedEndOfData;
        }

        const result = std.mem.readIntLittle(u32, self.data[self.pos..self.pos + 4]);
        self.pos += 4;
        return result;
    }

    pub fn readFixed64(self: *Self) ProtobufError!u64 {
        if (self.pos + 8 > self.data.len) {
            return ProtobufError.UnexpectedEndOfData;
        }

        const result = std.mem.readIntLittle(u64, self.data[self.pos..self.pos + 8]);
        self.pos += 8;
        return result;
    }

    pub fn readBytes(self: *Self) ProtobufError![]const u8 {
        const length = try self.readVarint();
        if (self.pos + length > self.data.len) {
            return ProtobufError.UnexpectedEndOfData;
        }

        const result = self.data[self.pos..self.pos + length];
        self.pos += length;
        return result;
    }

    pub fn readString(self: *Self) ProtobufError![]const u8 {
        return self.readBytes();
    }

    pub fn readFloat(self: *Self) ProtobufError!f32 {
        const bits = try self.readFixed32();
        return @as(f32, @bitCast(bits));
    }

    pub fn readDouble(self: *Self) ProtobufError!f64 {
        const bits = try self.readFixed64();
        return @as(f64, @bitCast(bits));
    }

    pub fn readInt32(self: *Self) ProtobufError!i32 {
        const value = try self.readVarint();
        return @as(i32, @truncate(value));
    }

    pub fn readInt64(self: *Self) ProtobufError!i64 {
        const value = try self.readVarint();
        return @as(i64, @bitCast(value));
    }

    pub fn readUInt32(self: *Self) ProtobufError!u32 {
        const value = try self.readVarint();
        return @as(u32, @truncate(value));
    }

    pub fn readUInt64(self: *Self) ProtobufError!u64 {
        return self.readVarint();
    }

    pub fn readBool(self: *Self) ProtobufError!bool {
        const value = try self.readVarint();
        return value != 0;
    }

    pub fn skipField(self: *Self, wire_type: WireType) ProtobufError!void {
        switch (wire_type) {
            .varint => {
                _ = try self.readVarint();
            },
            .fixed64 => {
                _ = try self.readFixed64();
            },
            .length_delimited => {
                _ = try self.readBytes();
            },
            .fixed32 => {
                _ = try self.readFixed32();
            },
            .start_group, .end_group => {
                return ProtobufError.UnsupportedFieldType;
            },
            _ => {
                return ProtobufError.InvalidWireType;
            },
        }
    }

    pub fn getRemainingBytes(self: *const Self) []const u8 {
        return self.data[self.pos..];
    }

    pub fn getPosition(self: *const Self) usize {
        return self.pos;
    }

    pub fn setPosition(self: *Self, pos: usize) void {
        self.pos = @min(pos, self.data.len);
    }

    pub fn rewind(self: *Self) void {
        self.pos = 0;
    }

    pub fn peek(self: *const Self, offset: usize) ?u8 {
        const peek_pos = self.pos + offset;
        if (peek_pos >= self.data.len) return null;
        return self.data[peek_pos];
    }

    pub fn readRepeatedVarint(self: *Self, allocator: Allocator) ProtobufError![]u64 {
        const data = try self.readBytes();
        var sub_parser = ProtobufParser.init(allocator, data);
        
        var values = std.ArrayList(u64).init(allocator);
        defer values.deinit();

        while (sub_parser.hasMoreData()) {
            const value = try sub_parser.readVarint();
            try values.append(value);
        }

        return values.toOwnedSlice();
    }

    pub fn readRepeatedFloat(self: *Self, allocator: Allocator) ProtobufError![]f32 {
        const data = try self.readBytes();
        if (data.len % 4 != 0) {
            return ProtobufError.InvalidString;
        }

        const count = data.len / 4;
        const values = try allocator.alloc(f32, count);

        for (0..count) |i| {
            const offset = i * 4;
            const bits = std.mem.readIntLittle(u32, data[offset..offset + 4]);
            values[i] = @as(f32, @bitCast(bits));
        }

        return values;
    }

    pub fn readRepeatedDouble(self: *Self, allocator: Allocator) ProtobufError![]f64 {
        const data = try self.readBytes();
        if (data.len % 8 != 0) {
            return ProtobufError.InvalidString;
        }

        const count = data.len / 8;
        const values = try allocator.alloc(f64, count);

        for (0..count) |i| {
            const offset = i * 8;
            const bits = std.mem.readIntLittle(u64, data[offset..offset + 8]);
            values[i] = @as(f64, @bitCast(bits));
        }

        return values;
    }
};

// Tests
test "protobuf varint encoding" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test data: varint encoding of 300 (0xAC 0x02)
    const data = [_]u8{ 0xAC, 0x02 };
    var parser = ProtobufParser.init(allocator, &data);

    const value = try parser.readVarint();
    try testing.expect(value == 300);
}

test "protobuf field header" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test data: field number 1, wire type 2 (length delimited)
    // Tag = (1 << 3) | 2 = 10 = 0x0A
    const data = [_]u8{0x0A};
    var parser = ProtobufParser.init(allocator, &data);

    const header = try parser.readFieldHeader();
    try testing.expect(header.field_number == 1);
    try testing.expect(header.wire_type == .length_delimited);
}

test "protobuf string reading" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test data: length 5, followed by "hello"
    const data = [_]u8{ 0x05, 'h', 'e', 'l', 'l', 'o' };
    var parser = ProtobufParser.init(allocator, &data);

    const str = try parser.readString();
    try testing.expect(std.mem.eql(u8, str, "hello"));
}

test "protobuf fixed32 reading" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test data: 0x12345678 in little endian
    const data = [_]u8{ 0x78, 0x56, 0x34, 0x12 };
    var parser = ProtobufParser.init(allocator, &data);

    const value = try parser.readFixed32();
    try testing.expect(value == 0x12345678);
}
