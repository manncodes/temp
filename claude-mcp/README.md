# MCP for Claude.ai

A browser extension that enables MCP (Model Control Protocol) capabilities in Claude.ai, allowing you to connect Claude to external tools and services directly from the browser. This enables functionality that's already existing in claude.ai but not enabled. 

<img width="400" alt="Screenshot 2025-04-12 at 3 53 10 PM" src="https://github.com/user-attachments/assets/65e69843-58d4-4686-80d9-1f6bb000e015" />
<img width="400" alt="Screenshot 2025-04-12 at 3 53 33 PM" src="https://github.com/user-attachments/assets/ba4b9b62-1cae-41db-ad58-8a824b6f861a" />

## Features

- Connect Claude.ai to MCP-compatible servers
- Manage multiple server connections
- Configure environment variables and command-line arguments
- Debug logging options
- Dark mode support

## Installation

### From Source

1. Clone this repository
```bash
git clone https://github.com/dnakov/claude-mcp.git
cd claude-mcp
```

2. Install dependencies
```bash
npm install
# or
pnpm install
```

3. Build the extension
```bash
npm run build
# or 
pnpm build
```

4. Load the extension in your browser:

**Chrome/Edge**:
- Go to `chrome://extensions/`
- Enable "Developer mode"
- Click "Load unpacked"
- Select the `dist` folder from this repository

**Firefox**:
- Go to `about:debugging#/runtime/this-firefox`
- Click "Load Temporary Add-on"
- Select the `manifest.json` file from the `dist` folder

## Usage

1. Click on the extension icon in your browser toolbar when on claude.ai
2. Add a new MCP server connection with the following details:
   - Name: A friendly name for the server
   - URL: The endpoint URL for the MCP server
   - Command (optional): The command to execute on the server
   - Arguments (optional): Command-line arguments
   - Environment Variables (optional): Key-value pairs for environment configuration

3. Once configured, the extension will establish connections to your MCP servers when you visit Claude.ai
4. Claude will be able to use the tools provided by your MCP servers during conversations

## Development

- Run in development mode with hot reloading:
```bash
npm run dev
# or
pnpm dev
```

## Technical Details

The extension uses a Server-Sent Events (SSE) connection to communicate with MCP servers. It consists of:

- A background script that manages storage and extension state
- Content scripts that inject MCP capabilities into Claude.ai
- An isolated content script for secure communication between contexts
- A popup UI for server management

## License

[MIT License](LICENSE) 
