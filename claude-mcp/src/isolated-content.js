const extensionAPI = globalThis.chrome || globalThis.browser;

let initialLoad = true;

async function getAndSendStorage() {
  try {
    const data = await extensionAPI.storage.local.get();
    
    // Send to MAIN world
    window.postMessage({
      type: 'mcp-ext-storage-updated',
      source: 'isolated-content',
      data: data
    }, '*');
    if(initialLoad) {
      initialLoad = false;
      window.postMessage({
        type: 'mcp-servers-updated',
        source: 'isolated-content',
        data: {
          servers: Object.values(data.mcpServers || {})
        }
      }, '*');
    }
  } catch (error) {
    console.error('[Claude MCP Manager] Error getting storage:', error);
  }
}

// Listen for requests from MAIN world
window.addEventListener('message', (event) => {
  // console.log('[Claude MCP Manager] Got message in ISOLATED world:', event.data);
  
  // if (event.source !== window) return;
  
  // if (event.data && event.data.type === 'mcp-ext-storage-updated') {
  //   console.log('[Claude MCP Manager] Got request for storage, sending from storage');
  //   getAndSendStorage();
  // }
});

extensionAPI.storage.onChanged.addListener((changes, area) => {
  if (area === 'local') {
    getAndSendStorage();
  }
  for(let key in changes) {
    if(key === 'mcpServers') {
      window.postMessage({
        type: 'mcp-servers-updated',
        source: 'isolated-content',
        data: {
          servers: Object.values(changes[key].newValue || {})
        }
      }, '*');
    }
  }
});

getAndSendStorage();
