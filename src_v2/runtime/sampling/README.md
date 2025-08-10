Sampling architecture
- Composable pipeline per the Registry-Per-Choice rule
- Order: penalties → temperature → top-k → top-p/min-p → softmax → selector
- Registries: LogitTransformRegistry, SelectorRegistry
- Presets (optional) live in SamplerRegistry and build via factory

