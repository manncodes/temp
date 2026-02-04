import { spawn } from 'child_process';

const serverName = process.argv[2] || 'filesystem';
const configs = {
  filesystem: {
    command: 'npx',
    args: ['-y', '@modelcontextprotocol/server-filesystem', '/home/user/temp']
  },
  memory: {
    command: 'npx',
    args: ['-y', '@modelcontextprotocol/server-memory']
  },
  'sequential-thinking': {
    command: 'npx',
    args: ['-y', '@modelcontextprotocol/server-sequential-thinking']
  }
};

const config = configs[serverName];
if (!config) {
  console.error(`Unknown server: ${serverName}`);
  process.exit(1);
}

console.log(`\n=== Testing MCP Server: ${serverName} ===\n`);

const proc = spawn(config.command, config.args, {
  stdio: ['pipe', 'pipe', 'pipe']
});

let buffer = '';

proc.stdout.on('data', (data) => {
  buffer += data.toString();
  // Try to parse complete JSON-RPC messages
  const lines = buffer.split('\n');
  buffer = lines.pop(); // keep incomplete line in buffer
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
  console.log(`[SERVER LOG] ${data.toString().trim()}`);
});

function send(msg) {
  const str = JSON.stringify(msg);
  console.log(`[SEND] ${str}`);
  proc.stdin.write(str + '\n');
}

// MCP Protocol sequence
setTimeout(() => {
  send({
    jsonrpc: '2.0',
    id: 1,
    method: 'initialize',
    params: {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'test-client', version: '1.0.0' }
    }
  });
}, 1000);

setTimeout(() => {
  send({ jsonrpc: '2.0', method: 'notifications/initialized' });
}, 2000);

setTimeout(() => {
  send({ jsonrpc: '2.0', id: 2, method: 'tools/list', params: {} });
}, 3000);

setTimeout(() => {
  if (serverName === 'filesystem') {
    send({
      jsonrpc: '2.0', id: 3, method: 'tools/call',
      params: { name: 'list_directory', arguments: { path: '/home/user/temp' } }
    });
  } else if (serverName === 'memory') {
    send({
      jsonrpc: '2.0', id: 3, method: 'tools/call',
      params: {
        name: 'create_entities',
        arguments: {
          entities: [
            { name: 'TestEntity', entityType: 'test', observations: ['This is a test entity created by Claude'] }
          ]
        }
      }
    });
  } else if (serverName === 'sequential-thinking') {
    send({
      jsonrpc: '2.0', id: 3, method: 'tools/call',
      params: {
        name: 'sequentialthinking',
        arguments: {
          thought: 'What are the files in this project?',
          nextThoughtNeeded: false,
          thoughtNumber: 1,
          totalThoughts: 1
        }
      }
    });
  }
}, 4000);

// For memory server, read back entities
setTimeout(() => {
  if (serverName === 'memory') {
    send({
      jsonrpc: '2.0', id: 4, method: 'tools/call',
      params: { name: 'read_graph', arguments: {} }
    });
  }
}, 5000);

setTimeout(() => {
  console.log('\n=== Test Complete ===');
  proc.kill();
  process.exit(0);
}, 7000);
