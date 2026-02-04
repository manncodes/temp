const STORAGE = {}

export const log = (...args) => {
  if(STORAGE.mcpDebugLog) {
    console.log(...args)
  }
}

export const setStorage = (value) => {
  for (const key in value) {
    STORAGE[key] = value[key];
  }
}
