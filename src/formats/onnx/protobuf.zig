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

    pub fn readFieldHeader(self: *Self) !FieldHeader {
        const tag = try self.readVarint();
        const wire_type_raw = @as(u3, @truncate(tag & 0x7));
        const wire_type = @as(WireType, @enumFromInt(wire_type_raw));
        const field_number = @as(u32, @truncate(tag >> 3));

        return FieldHeader{
            .field_number = field_number,
            .wire_type = wire_type,
        };
    }

    pub fn readVarint(self: *Self) !u64 {
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

    pub fn readString(self: *Self) ![]const u8 {
        const length = try self.readVarint();
        if (self.pos + length > self.data.len) {
            return ProtobufError.UnexpectedEndOfData;
        }

        const start = self.pos;
        self.pos += @as(usize, @intCast(length));
        return self.data[start..self.pos];
    }

    pub fn readBytes(self: *Self) ![]const u8 {
        return self.readString(); // Same format as string
    }

    pub fn readFixed32(self: *Self) !u32 {
        if (self.pos + 4 > self.data.len) {
            return ProtobufError.UnexpectedEndOfData;
        }

        const result = std.mem.readIntLittle(u32, self.data[self.pos .. self.pos + 4][0..4]);
        self.pos += 4;
        return result;
    }

    pub fn readFixed64(self: *Self) !u64 {
        if (self.pos + 8 > self.data.len) {
            return ProtobufError.UnexpectedEndOfData;
        }

        const result = std.mem.readIntLittle(u64, self.data[self.pos .. self.pos + 8][0..8]);
        self.pos += 8;
        return result;
    }

    pub fn readFloat(self: *Self) !f32 {
        const bits = try self.readFixed32();
        return @as(f32, @bitCast(bits));
    }

    pub fn readDouble(self: *Self) !f64 {
        const bits = try self.readFixed64();
        return @as(f64, @bitCast(bits));
    }

    pub fn skipField(self: *Self, wire_type: WireType) !void {
        switch (wire_type) {
            .varint => {
                _ = try self.readVarint();
            },
            .fixed64 => {
                _ = try self.readFixed64();
            },
            .length_delimited => {
                const length = try self.readVarint();
                if (self.pos + length > self.data.len) {
                    return ProtobufError.UnexpectedEndOfData;
                }
                self.pos += @as(usize, @intCast(length));
            },
            .fixed32 => {
                _ = try self.readFixed32();
            },
            else => {
                return ProtobufError.InvalidWireType;
            },
        }
    }

    pub fn readRepeatedVarint(self: *Self, allocator: Allocator) ![]u64 {
        const data_bytes = try self.readBytes();
        var sub_parser = ProtobufParser.init(allocator, data_bytes);

        var values = std.ArrayList(u64).init(allocator);
        defer values.deinit();

        while (sub_parser.hasMoreData()) {
            const value = try sub_parser.readVarint();
            try values.append(value);
        }

        return values.toOwnedSlice();
    }

    pub fn readRepeatedFloat(self: *Self, allocator: Allocator) ![]f32 {
        const data_bytes = try self.readBytes();
        if (data_bytes.len % 4 != 0) {
            return ProtobufError.InvalidString;
        }

        const count = data_bytes.len / 4;
        const values = try allocator.alloc(f32, count);

        for (0..count) |i| {
            const offset = i * 4;
            const bits = std.mem.readIntLittle(u32, data_bytes[offset .. offset + 4]);
            values[i] = @as(f32, @bitCast(bits));
        }

        return values;
    }

    pub fn readRepeatedDouble(self: *Self, allocator: Allocator) ![]f64 {
        const data_bytes = try self.readBytes();
        if (data_bytes.len % 8 != 0) {
            return ProtobufError.InvalidString;
        }

        const count = data_bytes.len / 8;
        const values = try allocator.alloc(f64, count);

        for (0..count) |i| {
            const offset = i * 8;
            const bits = std.mem.readIntLittle(u64, data_bytes[offset .. offset + 8]);
            values[i] = @as(f64, @bitCast(bits));
        }

        return values;
    }

    pub fn getCurrentPosition(self: *const Self) usize {
        return self.pos;
    }

    pub fn setPosition(self: *Self, pos: usize) void {
        self.pos = @min(pos, self.data.len);
    }

    pub fn getRemainingBytes(self: *const Self) []const u8 {
        return self.data[self.pos..];
    }
};

/// Helper functions for ONNX-specific protobuf parsing
pub const ONNXProtobufHelper = struct {
    pub fn parseStringList(parser: *ProtobufParser, allocator: Allocator) ![][]const u8 {
        var strings = std.ArrayList([]const u8).init(allocator);
        defer strings.deinit();

        const data_bytes = try parser.readBytes();
        var sub_parser = ProtobufParser.init(allocator, data_bytes);

        while (sub_parser.hasMoreData()) {
            const header = try sub_parser.readFieldHeader();
            if (header.wire_type == .length_delimited) {
                const str = try sub_parser.readString();
                const owned_str = try allocator.dupe(u8, str);
                try strings.append(owned_str);
            } else {
                try sub_parser.skipField(header.wire_type);
            }
        }

        return strings.toOwnedSlice();
    }

    pub fn parseInt64List(parser: *ProtobufParser, allocator: Allocator) ![]i64 {
        const varints = try parser.readRepeatedVarint(allocator);
        defer allocator.free(varints);

        const result = try allocator.alloc(i64, varints.len);
        for (varints, 0..) |varint, i| {
            result[i] = @as(i64, @bitCast(varint));
        }

        return result;
    }
};

// Test function to verify protobuf parsing
pub fn testProtobufParser() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test data: simple protobuf message
    const test_data = [_]u8{ 0x08, 0x96, 0x01, 0x12, 0x04, 0x74, 0x65, 0x73, 0x74 };

    var parser = ProtobufParser.init(allocator, &test_data);

    // Read first field (varint)
    const header1 = try parser.readFieldHeader();
    std.log.info("Field 1: number={}, wire_type={}", .{ header1.field_number, header1.wire_type });
    const value1 = try parser.readVarint();
    std.log.info("Value 1: {}", .{value1});

    // Read second field (string)
    const header2 = try parser.readFieldHeader();
    std.log.info("Field 2: number={}, wire_type={}", .{ header2.field_number, header2.wire_type });
    const value2 = try parser.readString();
    std.log.info("Value 2: {s}", .{value2});

    std.log.info("✅ Protobuf parser test completed successfully");
}
