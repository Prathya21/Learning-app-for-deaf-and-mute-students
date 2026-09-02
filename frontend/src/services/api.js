const API_BASE = '/api';

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  if (options.body && typeof options.body === 'object') {
    config.body = JSON.stringify(options.body);
  }

  const response = await fetch(url, config);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new ApiError(error.detail || `HTTP ${response.status}`, response.status);
  }

  return response.json();
}

export const api = {
  health: () => request('/health'),

  getVideo: (word) => request(`/videos/${encodeURIComponent(word)}`),

  translateTextToIsl: (text) => request('/translate/text-to-isl', {
    method: 'POST',
    body: { text },
  }),
};

export { ApiError };