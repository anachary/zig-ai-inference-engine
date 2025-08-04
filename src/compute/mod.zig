const std = @import("std");

/// Compute backend types
pub const Backend = enum {
    cpu,
    gpu,
    wasm,
};

/// Compute device abstraction (placeholder)
pub const Device = struct {
    backend: Backend,
    
    pub fn init(backend: Backend) Device {
        return Device{ .backend = backend };
    }
};
