import { spawn } from 'child_process';

const serverName = process.argv[2] || 'arxiv';
const configs = {
  arxiv: {
    command: 'arxiv-mcp-server',
    args: ['--storage-path', '/home/user/temp/.arxiv-papers']
  },
  fetch: {
    command: 'python3',
    args: ['-m', 'mcp_server_fetch']
  }
};

const config = configs[serverName];
if (!config) {
  console.error(`Unknown server: ${serverName}. Available: ${Object.keys(configs).join(', ')}`);
  process.exit(1);
}

console.log(`\n=== Testing MCP Server: ${serverName} ===\n`);

const proc = spawn(config.command, config.args, {
  stdio: ['pipe', 'pipe', 'pipe']
});

let buffer = '';

proc.stdout.on('data', (data) => {
  buffer += data.toString();
  const lines = buffer.split('\n');
  buffer = lines.pop();
  for (const line of lines) {
    if (line.trim()) {
      try {
        const msg = JSON.parse(line);
        console.log(`[RESPONSE] ${JSON.stringify(msg, null, 2)}`);
      } catch {
        console.log(`[RAW] ${line}`);
      }
    }
  }
});

proc.stderr.on('data', (data) => {
  const text = data.toString().trim();
  if (text) console.log(`[SERVER LOG] ${text}`);
});

function send(msg) {
  const str = JSON.stringify(msg);
  console.log(`[SEND] ${str.slice(0, 200)}${str.length > 200 ? '...' : ''}`);
  proc.stdin.write(str + '\n');
}

// Initialize
setTimeout(() => {
  send({
    jsonrpc: '2.0', id: 1, method: 'initialize',
    params: {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'test-client', version: '1.0.0' }
    }
  });
}, 500);

setTimeout(() => {
  send({ jsonrpc: '2.0', method: 'notifications/initialized' });
}, 1500);

// List tools
setTimeout(() => {
  send({ jsonrpc: '2.0', id: 2, method: 'tools/list', params: {} });
}, 2500);

// Call a tool
setTimeout(() => {
  if (serverName === 'arxiv') {
    console.log('\n--- Searching arxiv for "attention is all you need" ---\n');
    send({
      jsonrpc: '2.0', id: 3, method: 'tools/call',
      params: {
        name: 'search_papers',
        arguments: { query: 'attention is all you need', max_results: 3 }
      }
    });
  } else if (serverName === 'fetch') {
    console.log('\n--- Fetching arxiv abstract page ---\n');
    send({
      jsonrpc: '2.0', id: 3, method: 'tools/call',
      params: {
        name: 'fetch',
        arguments: { url: 'https://arxiv.org/abs/1706.03762' }
      }
    });
  }
}, 3500);

setTimeout(() => {
  console.log('\n=== Test Complete ===');
  proc.kill();
  process.exit(0);
}, 15000);
