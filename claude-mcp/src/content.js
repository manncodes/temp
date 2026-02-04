// Content script running in world: MAIN on claude.ai
// This means we're running in the page's JavaScript context, not the extension context
// We can access Claude's React components but NOT extension APIs

// import 'bippy';
import { MCPConnection } from './mcp-connection';
import { log, setStorage } from './utils';

let mcpServers = [];

// Set up a connection to a server
function createServerConnection(server, handler) {
  log(`[MCP Client] Creating connection to ${server.name}`);
  
  try {
    // Create configuration for the MCPConnection
    const connectionConfig = {
      connectionType: 'sse',
      transportType: 'stdio',
      url: server.url,
      command: server.command,
      args: server.args,
      env: server.env
    };
    
    // Create the MCPConnection
    const connection = new MCPConnection(server.name, connectionConfig);
    
    // Start connecting in the background
    connection.connect(handler)
      .then(result => {
        log(`[MCP Client] Successfully connected to ${server.name}:`, result);
        // Trigger an update of the injected clients to refresh with the new connection
        // updateInjectedClients();
      })
      .catch(error => {
        console.error(`[MCP Client] Failed to connect to ${server.name}:`, error);
      });
    
    return connection;
  } catch (e) {
    console.error(`[MCP Client] Error creating connection to ${server.name}:`, e);
    return null;
  }
}

// Listen for messages from other scripts
window.addEventListener('message', (event) => {
  // Log messages for debugging
  log('[MCP Client] Received window message:', event.data);

  // Only accept messages from the same frame
  if (event.source !== window) {
    return;
  }
  
  // Special case: filter out our own messages
  if (event.data && event.data.source === 'main-content') {
    log('[MCP Client] Ignoring our own message');
    return;
  }
  
  // Process messages by type
  if (event.data && event.data.type) {    
    switch (event.data.type) {
      case 'mcp-ext-storage-updated':
        log('[MCP Client] Received storage from extension:', event.data.data);
        setStorage(event.data.data);
        break;
      
      case 'mcp-servers-updated':
        if (Array.isArray(event.data?.data?.servers)) {
          log('[MCP Client] Received servers from extension:', event.data.data.servers);
          mcpServers = event.data.data.servers;
          for(let server of mcpServers) {
            const channel = new MessageChannel();
            const port1 = channel.port1;
            const port2 = channel.port2;
            port1.onmessage = (event) => {
              log('[MCP Client] Received message from port1:', event);
              connection.sendRequest(event.data);
            };
            const handler = (message) => {
              log('[MCP Client] Received message from server:', message);
              port1.postMessage(message);
            };  
            port1.start();
            const connection = createServerConnection(server, handler);
            log(server, connection)
            setTimeout(() => {  
              window.postMessage({
                source: 'main-content',
                type: 'mcp-server-connected',
                serverName: server.name
              }, '*', [port2]);
            }, 1000);
          }
        
        } else {
          console.warn('[MCP Client] Received malformed servers data:', event.data);
        }
        break;
    }
  }
});