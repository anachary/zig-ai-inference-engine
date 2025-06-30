const std = @import("std");

/// Comprehensive error types for the Zig AI ecosystem
/// Provides consistent error handling across all projects

/// Core system errors
pub const CoreError = error{
    /// Memory-related errors
    OutOfMemory,
    InvalidAllocation,
    MemoryCorruption,
    AllocationFailed,
    
    /// Configuration errors
    InvalidConfiguration,
    MissingConfiguration,
    ConfigurationConflict,
    
    /// Initialization errors
    InitializationFailed,
    AlreadyInitialized,
    NotInitialized,
    
    /// Resource errors
    ResourceExhausted,
    ResourceNotFound,
    ResourceBusy,
    ResourceLocked,
};

/// Tensor operation errors
pub const TensorError = error{
    /// Shape-related errors
    InvalidShape,
    ShapeMismatch,
    DimensionMismatch,
    InvalidDimension,
    
    /// Data type errors
    UnsupportedDataType,
    DataTypeMismatch,
    InvalidDataType,
    
    /// Index and access errors
    IndexOutOfBounds,
    InvalidIndex,
    InvalidAccess,
    
    /// Operation errors
    InvalidOperation,
    OperationNotSupported,
    BroadcastError,
    
    /// Device errors
    DeviceMismatch,
    DeviceNotSupported,
    DeviceError,
};

/// Model format and parsing errors
pub const ModelError = error{
    /// Format errors
    InvalidFormat,
    UnsupportedFormat,
    FormatVersionMismatch,
    CorruptedModel,
    
    /// Parsing errors
    ParseError,
    InvalidProtobuf,
    MissingRequiredField,
    InvalidFieldValue,
    
    /// Graph errors
    InvalidGraph,
    MissingGraph,
    CyclicGraph,
    UnconnectedNode,
    
    /// Operator errors
    UnsupportedOperator,
    InvalidOperator,
    MissingOperator,
    OperatorVersionMismatch,
    
    /// Model validation errors
    ValidationFailed,
    InconsistentModel,
    MissingMetadata,
};

/// Inference engine errors
pub const InferenceError = error{
    /// Model loading errors
    ModelNotLoaded,
    ModelLoadFailed,
    ModelValidationFailed,
    
    /// Input/output errors
    InvalidInput,
    InvalidOutput,
    InputShapeMismatch,
    OutputShapeMismatch,
    
    /// Execution errors
    ExecutionFailed,
    KernelLaunchFailed,
    SynchronizationFailed,
    TimeoutError,
    
    /// Optimization errors
    OptimizationFailed,
    UnsupportedOptimization,
    
    /// Backend errors
    BackendError,
    BackendNotSupported,
    BackendInitializationFailed,
};

/// Device and hardware errors
pub const DeviceError = error{
    /// Device discovery and initialization
    DeviceNotFound,
    DeviceNotSupported,
    DeviceInitializationFailed,
    DriverError,
    
    /// Memory management
    OutOfDeviceMemory,
    DeviceMemoryCorruption,
    InvalidDevicePointer,
    
    /// Execution errors
    DeviceExecutionFailed,
    DeviceSynchronizationFailed,
    DeviceTimeout,
    
    /// Hardware errors
    HardwareError,
    OverheatingError,
    PowerError,
};

/// Network and server errors
pub const NetworkError = error{
    /// Connection errors
    ConnectionFailed,
    ConnectionTimeout,
    ConnectionReset,
    ConnectionRefused,
    
    /// Protocol errors
    InvalidRequest,
    InvalidResponse,
    ProtocolError,
    UnsupportedProtocol,
    
    /// Server errors
    ServerError,
    ServerOverloaded,
    ServerUnavailable,
    
    /// Authentication and authorization
    AuthenticationFailed,
    AuthorizationFailed,
    InvalidCredentials,
    
    /// Data transfer errors
    TransferFailed,
    DataCorruption,
    IncompleteTransfer,
};

/// File system and I/O errors
pub const IOError = error{
    /// File operations
    FileNotFound,
    FileAccessDenied,
    FileCorrupted,
    FileTooBig,
    
    /// Directory operations
    DirectoryNotFound,
    DirectoryNotEmpty,
    InvalidPath,
    
    /// Read/write operations
    ReadError,
    WriteError,
    SeekError,
    FlushError,
    
    /// Permissions
    PermissionDenied,
    InsufficientPermissions,
};

/// Unified error type that encompasses all error categories
pub const ZigAIError = CoreError || TensorError || ModelError || InferenceError || DeviceError || NetworkError || IOError;

/// Error context for better debugging
pub const ErrorContext = struct {
    error_code: ZigAIError,
    message: []const u8,
    file: []const u8,
    line: u32,
    function: []const u8,
    timestamp: i64,
    additional_info: ?[]const u8,

    pub fn init(
        error_code: ZigAIError,
        message: []const u8,
        file: []const u8,
        line: u32,
        function: []const u8,
        additional_info: ?[]const u8,
    ) ErrorContext {
        return ErrorContext{
            .error_code = error_code,
            .message = message,
            .file = file,
            .line = line,
            .function = function,
            .timestamp = std.time.timestamp(),
            .additional_info = additional_info,
        };
    }

    pub fn format(
        self: ErrorContext,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print(
            "Error: {} - {} at {}:{} in {}() [{}]",
            .{ self.error_code, self.message, self.file, self.line, self.function, self.timestamp }
        );
        if (self.additional_info) |info| {
            try writer.print(" - Additional info: {s}", .{info});
        }
    }
};

/// Error reporting utilities
pub const ErrorReporter = struct {
    allocator: std.mem.Allocator,
    contexts: std.ArrayList(ErrorContext),
    max_contexts: usize,

    pub fn init(allocator: std.mem.Allocator, max_contexts: usize) ErrorReporter {
        return ErrorReporter{
            .allocator = allocator,
            .contexts = std.ArrayList(ErrorContext).init(allocator),
            .max_contexts = max_contexts,
        };
    }

    pub fn deinit(self: *ErrorReporter) void {
        self.contexts.deinit();
    }

    pub fn report(self: *ErrorReporter, context: ErrorContext) !void {
        if (self.contexts.items.len >= self.max_contexts) {
            _ = self.contexts.orderedRemove(0);
        }
        try self.contexts.append(context);
        
        // Log the error
        std.log.err("{}", .{context});
    }

    pub fn getRecentErrors(self: *const ErrorReporter) []const ErrorContext {
        return self.contexts.items;
    }

    pub fn clearErrors(self: *ErrorReporter) void {
        self.contexts.clearRetainingCapacity();
    }
};

/// Macro for creating error contexts
pub fn createErrorContext(
    error_code: ZigAIError,
    message: []const u8,
    additional_info: ?[]const u8,
) ErrorContext {
    return ErrorContext.init(
        error_code,
        message,
        @src().file,
        @src().line,
        @src().fn_name,
        additional_info,
    );
}

/// Result type for operations that can fail
pub fn Result(comptime T: type) type {
    return union(enum) {
        ok: T,
        err: ErrorContext,

        pub fn isOk(self: @This()) bool {
            return switch (self) {
                .ok => true,
                .err => false,
            };
        }

        pub fn isErr(self: @This()) bool {
            return !self.isOk();
        }

        pub fn unwrap(self: @This()) T {
            return switch (self) {
                .ok => |value| value,
                .err => |context| {
                    std.log.err("Unwrapped error result: {}", .{context});
                    @panic("Unwrapped error result");
                },
            };
        }

        pub fn unwrapOr(self: @This(), default: T) T {
            return switch (self) {
                .ok => |value| value,
                .err => default,
            };
        }

        pub fn expect(self: @This(), message: []const u8) T {
            return switch (self) {
                .ok => |value| value,
                .err => |context| {
                    std.log.err("{s}: {}", .{ message, context });
                    @panic(message);
                },
            };
        }
    };
}

/// Error conversion utilities
pub fn convertError(err: anyerror) ZigAIError {
    return switch (err) {
        // Memory errors
        error.OutOfMemory => CoreError.OutOfMemory,
        
        // File errors
        error.FileNotFound => IOError.FileNotFound,
        error.AccessDenied => IOError.FileAccessDenied,
        error.PermissionDenied => IOError.PermissionDenied,
        
        // Network errors
        error.ConnectionRefused => NetworkError.ConnectionRefused,
        error.ConnectionTimedOut => NetworkError.ConnectionTimeout,
        error.ConnectionResetByPeer => NetworkError.ConnectionReset,
        
        // Default to generic error
        else => CoreError.InitializationFailed,
    };
}

/// Test utilities for error handling
pub const testing = struct {
    pub fn expectError(expected: ZigAIError, actual: ZigAIError) !void {
        if (expected != actual) {
            std.log.err("Expected error {}, got {}", .{ expected, actual });
            return error.TestExpectedError;
        }
    }

    pub fn expectOk(comptime T: type, result: Result(T)) !T {
        return switch (result) {
            .ok => |value| value,
            .err => |context| {
                std.log.err("Expected Ok, got error: {}", .{context});
                return error.TestExpectedOk;
            },
        };
    }

    pub fn expectErr(comptime T: type, result: Result(T)) !ErrorContext {
        return switch (result) {
            .ok => {
                std.log.err("Expected error, got Ok", .{});
                return error.TestExpectedError;
            },
            .err => |context| context,
        };
    }
};
