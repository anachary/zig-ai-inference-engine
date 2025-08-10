# Diagrams

We store Mermaid sources alongside exported images.

- ollama_gguf_llama_flow.mmd – Mermaid source
- ollama_gguf_llama_flow.png – PNG export (generate via Kroki or Mermaid CLI)

Export instructions:
- Using Mermaid CLI (preferred locally):
  - Install: `npm install -g @mermaid-js/mermaid-cli`
  - Render: `mmdc -i docs/diagrams/ollama_gguf_llama_flow.mmd -o docs/diagrams/ollama_gguf_llama_flow.png`
- Using Kroki (HTTP):
  - POST the file to https://kroki.io/mermaid/png with content-type text/plain
  - Example (PowerShell):
    - `Invoke-WebRequest -Uri https://kroki.io/mermaid/png -Method Post -ContentType 'text/plain' -InFile 'docs/diagrams/ollama_gguf_llama_flow.mmd' -OutFile 'docs/diagrams/ollama_gguf_llama_flow.png'`

Once generated, the PNG is referenced from the status doc.

