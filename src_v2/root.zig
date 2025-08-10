pub const core = @import("core/api.zig");
pub const bootstrap = @import("bootstrap.zig");
pub const sampling_factory = @import("runtime/sampling/factory.zig");
pub const tok = @import("tokenizers/gguf_vocab.zig");
pub const chat = @import("chat/session.zig");
const weight_store = @import("runtime/weight_store.zig");

pub const loadModel = core.loadModel;
pub const forward = core.forward;
pub const generate = core.generate;
pub const generateN = core.generateN;

pub fn setWeightCacheCapMB(mb: usize) void {
    weight_store.WeightStore.setCacheCapBytes(mb * 1024 * 1024);
}
