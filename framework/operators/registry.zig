const std = @import("std");
const Allocator = std.mem.Allocator;
const interfaces = @import("../core/interfaces.zig");
const base = @import("base.zig");

const Tensor = interfaces.Tensor;
const Attributes = interfaces.Attributes;
const ExecutionContext = interfaces.ExecutionContext;
const FrameworkError = interfaces.FrameworkError;
const OperatorInterface = base.OperatorInterface;

/// Operator registry for managing and executing operators
pub const OperatorRegistry = struct {
    operators: std.StringHashMap(OperatorInterface.Definition),
    operator_versions: std.StringHashMap(std.ArrayList([]const u8)),
    search_paths: std.ArrayList([]const u8),
    allocator: Allocator,

    const Self = @This();

    /// Registry errors
    pub const RegistryError = error{
        OperatorNotFound,
        OperatorAlreadyExists,
        InvalidOperatorDefinition,
        VersionConflict,
        LoadingFailed,
    } || FrameworkError;

    /// Operator discovery information
    pub const OperatorInfo = struct {
        name: []const u8,
        version: []const u8,
        domain: []const u8,
        description: []const u8,
        supported_types: []const Tensor.DataType,
        min_inputs: u32,
        max_inputs: u32,
        min_outputs: u32,
        max_outputs: u32,
    };

    /// Initialize the operator registry
    pub fn init(allocator: Allocator) Self {
        return Self{
            .operators = std.StringHashMap(OperatorInterface.Definition).init(allocator),
            .operator_versions = std.StringHashMap(std.ArrayList([]const u8)).init(allocator),
            .search_paths = std.ArrayList([]const u8).init(allocator),
            .allocator = allocator,
        };
    }

    /// Deinitialize the registry
    pub fn deinit(self: *Self) void {
        // Clean up operator versions
        var version_iter = self.operator_versions.iterator();
        while (version_iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.operator_versions.deinit();
        
        self.operators.deinit();
        self.search_paths.deinit();
    }

    /// Register a new operator
    pub fn registerOperator(self: *Self, definition: OperatorInterface.Definition) RegistryError!void {
        const key = try self.createOperatorKey(definition.metadata.name, definition.metadata.version);
        defer self.allocator.free(key);

        // Check if operator already exists
        if (self.operators.contains(key)) {
            return RegistryError.OperatorAlreadyExists;
        }

        // Validate operator definition
        try self.validateOperatorDefinition(&definition);

        // Register the operator
        try self.operators.put(try self.allocator.dupe(u8, key), definition);

        // Track version
        try self.trackOperatorVersion(definition.metadata.name, definition.metadata.version);

        std.log.info("Registered operator: {s} v{s}", .{ definition.metadata.name, definition.metadata.version });
    }

    /// Override an existing operator
    pub fn overrideOperator(self: *Self, definition: OperatorInterface.Definition) RegistryError!void {
        const key = try self.createOperatorKey(definition.metadata.name, definition.metadata.version);
        defer self.allocator.free(key);

        // Validate operator definition
        try self.validateOperatorDefinition(&definition);

        // Override the operator (replace if exists, add if not)
        try self.operators.put(try self.allocator.dupe(u8, key), definition);

        // Track version
        try self.trackOperatorVersion(definition.metadata.name, definition.metadata.version);

        std.log.info("Overridden operator: {s} v{s}", .{ definition.metadata.name, definition.metadata.version });
    }

    /// Get operator definition
    pub fn getOperator(self: *const Self, name: []const u8, version: ?[]const u8) ?OperatorInterface.Definition {
        const actual_version = version orelse self.getLatestVersion(name) orelse return null;
        const key = self.createOperatorKey(name, actual_version) catch return null;
        defer self.allocator.free(key);

        return self.operators.get(key);
    }

    /// Execute an operator
    pub fn executeOperator(
        self: *const Self,
        name: []const u8,
        inputs: []const Tensor,
        outputs: []Tensor,
        attributes: *const Attributes,
        context: *ExecutionContext,
        version: ?[]const u8,
    ) RegistryError!void {
        const definition = self.getOperator(name, version) orelse return RegistryError.OperatorNotFound;

        // Validate inputs and outputs
        try self.validateExecution(&definition, inputs, outputs, attributes);

        // Execute the operator
        try definition.compute_fn(inputs, outputs, attributes, context);
    }

    /// Validate operator execution
    fn validateExecution(
        self: *const Self,
        definition: *const OperatorInterface.Definition,
        inputs: []const Tensor,
        outputs: []Tensor,
        attributes: *const Attributes,
    ) RegistryError!void {
        _ = self;

        // Check input/output counts
        if (inputs.len < definition.metadata.min_inputs or inputs.len > definition.metadata.max_inputs) {
            return RegistryError.InvalidInput;
        }

        if (outputs.len < definition.metadata.min_outputs or outputs.len > definition.metadata.max_outputs) {
            return RegistryError.InvalidOutput;
        }

        // Collect input shapes and types
        var input_shapes = try self.allocator.alloc([]const usize, inputs.len);
        defer self.allocator.free(input_shapes);
        
        var input_types = try self.allocator.alloc(Tensor.DataType, inputs.len);
        defer self.allocator.free(input_types);

        for (inputs, 0..) |input, i| {
            input_shapes[i] = input.shape;
            input_types[i] = input.dtype;
        }

        // Run operator-specific validation
        try definition.validate_fn(input_shapes, input_types, attributes);
    }

    /// Infer output shapes for an operator
    pub fn inferShapes(
        self: *const Self,
        name: []const u8,
        input_shapes: []const []const usize,
        attributes: *const Attributes,
        version: ?[]const u8,
    ) RegistryError![][]usize {
        const definition = self.getOperator(name, version) orelse return RegistryError.OperatorNotFound;
        return definition.infer_shapes_fn(input_shapes, attributes, self.allocator);
    }

    /// Get optimization hints for an operator
    pub fn getOptimizationHints(
        self: *const Self,
        name: []const u8,
        inputs: []const Tensor,
        attributes: *const Attributes,
        context: *ExecutionContext,
        version: ?[]const u8,
    ) RegistryError!?OperatorInterface.OptimizationHint {
        const definition = self.getOperator(name, version) orelse return RegistryError.OperatorNotFound;
        
        if (definition.optimize_fn) |optimize_fn| {
            return optimize_fn(inputs, attributes, context);
        }
        
        return null;
    }

    /// List all registered operators
    pub fn listOperators(self: *const Self) ![]OperatorInfo {
        var operators = std.ArrayList(OperatorInfo).init(self.allocator);
        defer operators.deinit();

        var iterator = self.operators.iterator();
        while (iterator.next()) |entry| {
            const definition = entry.value_ptr.*;
            
            // Extract supported types from type constraints
            var supported_types = std.ArrayList(Tensor.DataType).init(self.allocator);
            defer supported_types.deinit();
            
            for (definition.metadata.type_constraints) |constraint| {
                for (constraint.allowed_types) |dtype| {
                    try supported_types.append(dtype);
                }
            }

            try operators.append(OperatorInfo{
                .name = definition.metadata.name,
                .version = definition.metadata.version,
                .domain = definition.metadata.domain,
                .description = definition.metadata.description,
                .supported_types = try supported_types.toOwnedSlice(),
                .min_inputs = definition.metadata.min_inputs,
                .max_inputs = definition.metadata.max_inputs,
                .min_outputs = definition.metadata.min_outputs,
                .max_outputs = definition.metadata.max_outputs,
            });
        }

        return operators.toOwnedSlice();
    }

    /// Add search path for operator discovery
    pub fn addSearchPath(self: *Self, path: []const u8) !void {
        try self.search_paths.append(try self.allocator.dupe(u8, path));
    }

    /// Discover operators in search paths
    pub fn discoverOperators(self: *Self) !void {
        for (self.search_paths.items) |path| {
            try self.discoverOperatorsInPath(path);
        }
    }

    /// Check if operator exists
    pub fn hasOperator(self: *const Self, name: []const u8, version: ?[]const u8) bool {
        return self.getOperator(name, version) != null;
    }

    /// Get all versions of an operator
    pub fn getOperatorVersions(self: *const Self, name: []const u8) ?[]const []const u8 {
        if (self.operator_versions.get(name)) |versions| {
            return versions.items;
        }
        return null;
    }

    /// Get latest version of an operator
    pub fn getLatestVersion(self: *const Self, name: []const u8) ?[]const u8 {
        if (self.getOperatorVersions(name)) |versions| {
            if (versions.len > 0) {
                return versions[versions.len - 1];
            }
        }
        return null;
    }

    // Private helper methods

    fn createOperatorKey(self: *const Self, name: []const u8, version: []const u8) ![]u8 {
        return std.fmt.allocPrint(self.allocator, "{s}:{s}", .{ name, version });
    }

    fn validateOperatorDefinition(self: *const Self, definition: *const OperatorInterface.Definition) RegistryError!void {
        _ = self;
        
        // Basic validation
        if (definition.metadata.name.len == 0) {
            return RegistryError.InvalidOperatorDefinition;
        }
        
        if (definition.metadata.version.len == 0) {
            return RegistryError.InvalidOperatorDefinition;
        }
        
        if (definition.metadata.min_inputs > definition.metadata.max_inputs) {
            return RegistryError.InvalidOperatorDefinition;
        }
        
        if (definition.metadata.min_outputs > definition.metadata.max_outputs) {
            return RegistryError.InvalidOperatorDefinition;
        }
    }

    fn trackOperatorVersion(self: *Self, name: []const u8, version: []const u8) !void {
        const owned_name = try self.allocator.dupe(u8, name);
        const owned_version = try self.allocator.dupe(u8, version);
        
        if (self.operator_versions.getPtr(owned_name)) |versions| {
            // Check if version already exists
            for (versions.items) |existing_version| {
                if (std.mem.eql(u8, existing_version, owned_version)) {
                    self.allocator.free(owned_version);
                    return;
                }
            }
            try versions.append(owned_version);
        } else {
            var versions = std.ArrayList([]const u8).init(self.allocator);
            try versions.append(owned_version);
            try self.operator_versions.put(owned_name, versions);
        }
    }

    fn discoverOperatorsInPath(self: *Self, path: []const u8) !void {
        _ = self;
        _ = path;
        // TODO: Implement operator discovery from filesystem
        // This would scan for .zig files or shared libraries containing operators
        std.log.info("Operator discovery not yet implemented for path: {s}", .{path});
    }
};
