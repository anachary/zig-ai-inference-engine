const std = @import("std");
const framework = @import("../../../framework/lib.zig");

const Tensor = framework.Tensor;
const Attributes = framework.Attributes;
const ExecutionContext = framework.ExecutionContext;
const FrameworkError = framework.FrameworkError;
const OperatorInterface = framework.OperatorInterface;
const BaseOperator = framework.BaseOperator;

/// If operator for conditional execution
pub const If = BaseOperator(struct {
    const Self = @This();

    pub fn getMetadata() OperatorInterface.Metadata {
        return OperatorInterface.Metadata{
            .name = "If",
            .version = "1.0.0",
            .description = "Conditional execution based on a boolean condition",
            .domain = "ai.onnx",
            .min_inputs = 1,
            .max_inputs = 1,
            .min_outputs = 1,
            .max_outputs = std.math.maxInt(u32),
            .supports_inplace = false,
            .supports_broadcasting = false,
            .type_constraints = &[_]OperatorInterface.TypeConstraint{
                OperatorInterface.TypeConstraint{
                    .name = "B",
                    .allowed_types = &[_]Tensor.DataType{.bool},
                    .description = "Constrain condition to boolean tensor",
                },
            },
        };
    }

    pub fn validate(
        input_shapes: []const []const usize,
        input_types: []const Tensor.DataType,
        attributes: *const Attributes,
    ) FrameworkError!void {
        if (input_shapes.len != 1) {
            return FrameworkError.InvalidInput;
        }

        if (input_types[0] != .bool) {
            return FrameworkError.DataTypeMismatch;
        }

        // Condition should be a scalar
        if (input_shapes[0].len != 0 and framework.utils.calculateTotalElements(input_shapes[0]) != 1) {
            return FrameworkError.ShapeMismatch;
        }

        // Validate that then_branch and else_branch attributes exist
        if (attributes.get("then_branch") == null or attributes.get("else_branch") == null) {
            return FrameworkError.ValidationFailed;
        }
    }

    pub fn inferShapes(
        input_shapes: []const []const usize,
        attributes: *const Attributes,
        allocator: std.mem.Allocator,
    ) FrameworkError![][]usize {
        _ = input_shapes;
        _ = attributes;
        
        // Output shapes depend on the executed branch
        // For now, return empty - this would need graph analysis
        const output_shapes = try allocator.alloc([]usize, 0);
        return output_shapes;
    }

    pub fn compute(
        inputs: []const Tensor,
        outputs: []Tensor,
        attributes: *const Attributes,
        context: *ExecutionContext,
    ) FrameworkError!void {
        _ = outputs;
        _ = context;
        
        if (inputs.len != 1) {
            return FrameworkError.InvalidInput;
        }

        const condition = &inputs[0];
        if (condition.dtype != .bool) {
            return FrameworkError.DataTypeMismatch;
        }

        const condition_data = condition.getData(bool);
        const condition_value = condition_data[0];

        // Get branch graphs
        const then_branch = attributes.get("then_branch");
        const else_branch = attributes.get("else_branch");

        if (then_branch == null or else_branch == null) {
            return FrameworkError.ValidationFailed;
        }

        // TODO: Execute the appropriate branch graph
        // This requires a graph execution engine
        if (condition_value) {
            // Execute then_branch
            std.log.info("Executing then branch");
        } else {
            // Execute else_branch
            std.log.info("Executing else branch");
        }

        // For now, this is a placeholder implementation
        return FrameworkError.UnsupportedOperation;
    }
});

/// Where operator for element-wise conditional selection
pub const Where = BaseOperator(struct {
    const Self = @This();

    pub fn getMetadata() OperatorInterface.Metadata {
        return OperatorInterface.Metadata{
            .name = "Where",
            .version = "1.0.0",
            .description = "Element-wise selection based on condition: condition ? x : y",
            .domain = "ai.onnx",
            .min_inputs = 3,
            .max_inputs = 3,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = false,
            .supports_broadcasting = true,
            .type_constraints = &[_]OperatorInterface.TypeConstraint{
                OperatorInterface.TypeConstraint{
                    .name = "B",
                    .allowed_types = &[_]Tensor.DataType{.bool},
                    .description = "Constrain condition to boolean tensor",
                },
                OperatorInterface.TypeConstraint{
                    .name = "T",
                    .allowed_types = &[_]Tensor.DataType{ .f32, .f16, .i32, .i16, .i8 },
                    .description = "Constrain x and y to numeric tensors",
                },
            },
        };
    }

    pub fn validate(
        input_shapes: []const []const usize,
        input_types: []const Tensor.DataType,
        attributes: *const Attributes,
    ) FrameworkError!void {
        _ = attributes;

        if (input_shapes.len != 3) {
            return FrameworkError.InvalidInput;
        }

        // First input (condition) should be boolean
        if (input_types[0] != .bool) {
            return FrameworkError.DataTypeMismatch;
        }

        // Second and third inputs should have the same type
        if (input_types[1] != input_types[2]) {
            return FrameworkError.DataTypeMismatch;
        }

        // Check broadcasting compatibility
        const condition_shape = input_shapes[0];
        const x_shape = input_shapes[1];
        const y_shape = input_shapes[2];

        if (!Self.checkBroadcastCompatibility(condition_shape, x_shape) or
            !Self.checkBroadcastCompatibility(condition_shape, y_shape) or
            !Self.checkBroadcastCompatibility(x_shape, y_shape)) {
            return FrameworkError.ShapeMismatch;
        }
    }

    pub fn inferShapes(
        input_shapes: []const []const usize,
        attributes: *const Attributes,
        allocator: std.mem.Allocator,
    ) FrameworkError![][]usize {
        _ = attributes;

        if (input_shapes.len != 3) {
            return FrameworkError.InvalidInput;
        }

        const condition_shape = input_shapes[0];
        const x_shape = input_shapes[1];
        const y_shape = input_shapes[2];

        // Calculate broadcast shape
        var temp_shape = try Self.calculateBroadcastShape(condition_shape, x_shape, allocator);
        defer allocator.free(temp_shape);
        
        const output_shapes = try allocator.alloc([]usize, 1);
        output_shapes[0] = try Self.calculateBroadcastShape(temp_shape, y_shape, allocator);
        
        return output_shapes;
    }

    pub fn compute(
        inputs: []const Tensor,
        outputs: []Tensor,
        attributes: *const Attributes,
        context: *ExecutionContext,
    ) FrameworkError!void {
        _ = attributes;

        if (inputs.len != 3 or outputs.len != 1) {
            return FrameworkError.InvalidInput;
        }

        const condition = &inputs[0];
        const x = &inputs[1];
        const y = &inputs[2];
        const output = &outputs[0];

        if (condition.dtype != .bool) {
            return FrameworkError.DataTypeMismatch;
        }

        if (x.dtype != y.dtype or x.dtype != output.dtype) {
            return FrameworkError.DataTypeMismatch;
        }

        switch (x.dtype) {
            .f32 => try whereF32(condition, x, y, output, context),
            .i32 => try whereI32(condition, x, y, output, context),
            else => return FrameworkError.UnsupportedOperation,
        }
    }

    fn whereF32(condition: *const Tensor, x: *const Tensor, y: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        _ = context;
        
        const condition_data = condition.getData(bool);
        const x_data = x.getData(f32);
        const y_data = y.getData(f32);
        const output_data = output.getMutableData(f32);

        const output_elements = framework.utils.calculateTotalElements(output.shape);

        for (0..output_elements) |i| {
            const cond_idx = calculateBroadcastIndex(i, output.shape, condition.shape, condition.strides);
            const x_idx = calculateBroadcastIndex(i, output.shape, x.shape, x.strides);
            const y_idx = calculateBroadcastIndex(i, output.shape, y.shape, y.strides);

            output_data[i] = if (condition_data[cond_idx]) x_data[x_idx] else y_data[y_idx];
        }
    }

    fn whereI32(condition: *const Tensor, x: *const Tensor, y: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        _ = context;
        
        const condition_data = condition.getData(bool);
        const x_data = x.getData(i32);
        const y_data = y.getData(i32);
        const output_data = output.getMutableData(i32);

        const output_elements = framework.utils.calculateTotalElements(output.shape);

        for (0..output_elements) |i| {
            const cond_idx = calculateBroadcastIndex(i, output.shape, condition.shape, condition.strides);
            const x_idx = calculateBroadcastIndex(i, output.shape, x.shape, x.strides);
            const y_idx = calculateBroadcastIndex(i, output.shape, y.shape, y.strides);

            output_data[i] = if (condition_data[cond_idx]) x_data[x_idx] else y_data[y_idx];
        }
    }

    fn calculateBroadcastIndex(linear_idx: usize, output_shape: []const usize, input_shape: []const usize, input_strides: []const usize) usize {
        var idx: usize = 0;
        var remaining = linear_idx;
        
        const ndim = output_shape.len;
        var i = ndim;
        while (i > 0) {
            i -= 1;
            const coord = remaining % output_shape[i];
            remaining /= output_shape[i];
            
            const input_coord = if (i < input_shape.len and input_shape[i] > 1) coord % input_shape[i] else 0;
            if (i < input_strides.len) {
                idx += input_coord * input_strides[i];
            }
        }
        
        return idx;
    }
});

/// Loop operator for iterative execution
pub const Loop = BaseOperator(struct {
    const Self = @This();

    pub fn getMetadata() OperatorInterface.Metadata {
        return OperatorInterface.Metadata{
            .name = "Loop",
            .version = "1.0.0",
            .description = "Generic loop construct for iterative execution",
            .domain = "ai.onnx",
            .min_inputs = 2,
            .max_inputs = std.math.maxInt(u32),
            .min_outputs = 1,
            .max_outputs = std.math.maxInt(u32),
            .supports_inplace = false,
            .supports_broadcasting = false,
            .type_constraints = &[_]OperatorInterface.TypeConstraint{
                OperatorInterface.TypeConstraint{
                    .name = "I",
                    .allowed_types = &[_]Tensor.DataType{.i64},
                    .description = "Constrain max_trip_count to int64",
                },
                OperatorInterface.TypeConstraint{
                    .name = "B",
                    .allowed_types = &[_]Tensor.DataType{.bool},
                    .description = "Constrain condition to boolean",
                },
            },
        };
    }

    pub fn validate(
        input_shapes: []const []const usize,
        input_types: []const Tensor.DataType,
        attributes: *const Attributes,
    ) FrameworkError!void {
        if (input_shapes.len < 2) {
            return FrameworkError.InvalidInput;
        }

        // First input should be int64 (max_trip_count)
        if (input_types[0] != .i64) {
            return FrameworkError.DataTypeMismatch;
        }

        // Second input should be bool (condition)
        if (input_types[1] != .bool) {
            return FrameworkError.DataTypeMismatch;
        }

        // Validate that body attribute exists
        if (attributes.get("body") == null) {
            return FrameworkError.ValidationFailed;
        }
    }

    pub fn inferShapes(
        input_shapes: []const []const usize,
        attributes: *const Attributes,
        allocator: std.mem.Allocator,
    ) FrameworkError![][]usize {
        _ = input_shapes;
        _ = attributes;
        
        // Output shapes depend on the loop body
        // For now, return empty - this would need graph analysis
        const output_shapes = try allocator.alloc([]usize, 0);
        return output_shapes;
    }

    pub fn compute(
        inputs: []const Tensor,
        outputs: []Tensor,
        attributes: *const Attributes,
        context: *ExecutionContext,
    ) FrameworkError!void {
        _ = outputs;
        _ = context;
        
        if (inputs.len < 2) {
            return FrameworkError.InvalidInput;
        }

        const max_trip_count = &inputs[0];
        const condition = &inputs[1];

        if (max_trip_count.dtype != .i64 or condition.dtype != .bool) {
            return FrameworkError.DataTypeMismatch;
        }

        const max_trips = max_trip_count.getData(i64)[0];
        var keep_going = condition.getData(bool)[0];

        // Get loop body graph
        const body = attributes.get("body");
        if (body == null) {
            return FrameworkError.ValidationFailed;
        }

        // TODO: Execute loop body iteratively
        // This requires a graph execution engine
        var trip_count: i64 = 0;
        while (keep_going and trip_count < max_trips) {
            std.log.info("Loop iteration: {}", .{trip_count});
            
            // Execute body graph
            // Update keep_going based on body output
            // Update loop carried dependencies
            
            trip_count += 1;
            
            // For now, break to avoid infinite loop
            break;
        }

        // For now, this is a placeholder implementation
        return FrameworkError.UnsupportedOperation;
    }
});

/// Scan operator for cumulative operations
pub const Scan = BaseOperator(struct {
    const Self = @This();

    pub fn getMetadata() OperatorInterface.Metadata {
        return OperatorInterface.Metadata{
            .name = "Scan",
            .version = "1.0.0",
            .description = "Scan operation for cumulative computations",
            .domain = "ai.onnx",
            .min_inputs = 1,
            .max_inputs = std.math.maxInt(u32),
            .min_outputs = 1,
            .max_outputs = std.math.maxInt(u32),
            .supports_inplace = false,
            .supports_broadcasting = false,
            .type_constraints = &[_]OperatorInterface.TypeConstraint{
                OperatorInterface.TypeConstraint{
                    .name = "T",
                    .allowed_types = &[_]Tensor.DataType{ .f32, .f16, .i32, .i16, .i8 },
                    .description = "Constrain input and output types to numeric tensors",
                },
            },
        };
    }

    pub fn validate(
        input_shapes: []const []const usize,
        input_types: []const Tensor.DataType,
        attributes: *const Attributes,
    ) FrameworkError!void {
        _ = input_types;
        
        if (input_shapes.len < 1) {
            return FrameworkError.InvalidInput;
        }

        // Validate that body attribute exists
        if (attributes.get("body") == null) {
            return FrameworkError.ValidationFailed;
        }

        // Validate scan_input_axes and scan_input_directions
        const num_scan_inputs = attributes.getInt("num_scan_inputs", 1);
        if (num_scan_inputs < 1) {
            return FrameworkError.ValidationFailed;
        }
    }

    pub fn inferShapes(
        input_shapes: []const []const usize,
        attributes: *const Attributes,
        allocator: std.mem.Allocator,
    ) FrameworkError![][]usize {
        _ = attributes;
        
        if (input_shapes.len < 1) {
            return FrameworkError.InvalidInput;
        }

        // For now, assume output has same shape as input
        const output_shapes = try allocator.alloc([]usize, 1);
        output_shapes[0] = try allocator.dupe(usize, input_shapes[0]);
        
        return output_shapes;
    }

    pub fn compute(
        inputs: []const Tensor,
        outputs: []Tensor,
        attributes: *const Attributes,
        context: *ExecutionContext,
    ) FrameworkError!void {
        _ = inputs;
        _ = outputs;
        _ = attributes;
        _ = context;
        
        // TODO: Implement scan operation
        // This requires a graph execution engine to execute the body
        std.log.info("Scan operation not yet implemented");
        return FrameworkError.UnsupportedOperation;
    }
});

// Tests
test "Where operator" {
    const allocator = std.testing.allocator;
    
    const shape = [_]usize{ 2, 2 };
    var condition = try framework.utils.createTensor(allocator, &shape, .bool);
    defer condition.deinit();
    var x = try framework.utils.createTensor(allocator, &shape, .f32);
    defer x.deinit();
    var y = try framework.utils.createTensor(allocator, &shape, .f32);
    defer y.deinit();
    var output = try framework.utils.createTensor(allocator, &shape, .f32);
    defer output.deinit();
    
    // Set test data
    const cond_data = [_]bool{ true, false, false, true };
    const x_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    
    try framework.utils.setTensorData(&condition, bool, &cond_data);
    try framework.utils.setTensorData(&x, f32, &x_data);
    try framework.utils.setTensorData(&y, f32, &y_data);
    
    const inputs = [_]Tensor{ condition, x, y };
    var outputs = [_]Tensor{output};
    
    var attrs = framework.utils.createAttributes(allocator);
    defer attrs.deinit();
    
    var context = framework.utils.createExecutionContext(allocator);
    
    try Where.compute(&inputs, &outputs, &attrs, &context);
    
    // Expected: [1.0, 6.0, 7.0, 4.0] (condition ? x : y)
    const expected = [_]f32{ 1.0, 6.0, 7.0, 4.0 };
    const result_data = framework.utils.getTensorData(&output, f32);
    try std.testing.expectEqualSlices(f32, &expected, result_data);
}
