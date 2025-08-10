This folder will contain pre-rendered static diagram images (SVG/PNG) for project-plan docs.

Suggested filenames:
- cli-user-flow.svg
- parsing-flow.svg
- runtime-execution-llama.svg

Regeneration (using mermaid-cli):
- Extract each mermaid block into a standalone .mmd file under src_v2/project-plan/images/src/
- Run: npx @mermaid-js/mermaid-cli -i src_v2/project-plan/images/src/cli-user-flow.mmd -o src_v2/project-plan/images/cli-user-flow.svg

