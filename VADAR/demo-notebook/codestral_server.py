import pandas as pd
import asyncio
import aiohttp
import json
import time
import threading
from aiohttp import web, ClientSession, ClientTimeout
from aiohttp.web_response import StreamResponse
import signal
import sys

# --- Configuration ---
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/14qS9A8XzpAT62iCUZMMbFz-SVmZ0ubK1uIhjc6z8PwI/export?format=csv&gid=0"
ROTATION_INTERVAL = 2 
HOST = "0.0.0.0"
PORT = 8000
MODEL_TO_PROXY = "codestral-latest"
CODESTRAL_BASE_URL = "https://codestral.mistral.ai/v1"

# Performance settings
MAX_CONNECTIONS = 1000
CONNECTION_TIMEOUT = 30
READ_TIMEOUT = 300

# --- Helper Function to Load Keys ---
def load_keys_from_google_sheet(url: str) -> list[str]:
    """Loads API keys from the first column of a public Google Sheet CSV link, skipping empty and header rows."""
    print(f"🔄 Loading API Keys from Google Sheet...")
    try:
        # Read the CSV and skip the first row (header or dummy row)
        df = pd.read_csv(
            url, 
            skiprows=1,  # Skip the first row
            on_bad_lines='skip',
            encoding='utf-8',
            usecols=[0],
            header=None
        )

        # Drop empty cells, strip whitespace, and filter out blanks
        api_keys = df[0].dropna().astype(str).map(str.strip).tolist()
        api_keys = [key for key in api_keys if key]  # Remove empty strings

        if not api_keys:
            print("⚠️ Warning: No valid API keys found after cleaning.")
            return []

        print(f"✅ Successfully loaded {len(api_keys)} API keys.")
        return api_keys

    except Exception as e:
        print(f"❌ Failed to load keys from Google Sheet: {e}")
        return []

async def validate_key(session: ClientSession, key: str) -> tuple[str, bool]:
    """Checks if an API key works by sending a dummy chat completion request."""
    url = f"{CODESTRAL_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    dummy_payload = {
        "model": "codestral-latest",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "max_tokens": 1
    }
    try:
        async with session.post(url, headers=headers, json=dummy_payload, timeout=10) as resp:
            # Accept anything like 200/201 as valid
            return key, resp.status < 300
    except Exception as e:
        return key, False

async def check_api_keys_validity(keys: list[str], concurrency: int = 10) -> tuple[list[str], list[str]]:
  """Checks all keys concurrently and returns (valid_keys, invalid_keys)."""
  connector = aiohttp.TCPConnector(limit=concurrency)
  timeout = aiohttp.ClientTimeout(total=15)

  async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
    tasks = [validate_key(session, key) for key in keys]
    results = await asyncio.gather(*tasks)

  valid_keys = [key for key, is_valid in results if is_valid]
  invalid_keys = [key for key, is_valid in results if not is_valid]

  print(f"✅ Valid keys: {len(valid_keys)}")
  print(f"❌ Invalid keys: {len(invalid_keys)}")
  return valid_keys, invalid_keys

# --- Ultra-Fast Thread-Safe Key Rotator ---
class UltraFastKeyRotator:
    __slots__ = ('keys', 'rotate_every', '_request_count', '_current_key_index', '_lock')
    
    def __init__(self, keys: list, rotate_every_n_requests: int):
        if not keys:
            raise ValueError("API key list cannot be empty.")
        self.keys = keys
        self.rotate_every = rotate_every_n_requests
        self._request_count = 0
        self._current_key_index = 0
        self._lock = threading.Lock()

    def get_current_key_and_increment(self) -> tuple[str, int]:
        """Get current key and increment counter in one atomic operation for max speed"""
        with self._lock:
            current_count = self._request_count
            self._request_count += 1
            
            # Calculate key index
            block_index = current_count // self.rotate_every
            key_index = block_index % len(self.keys)
            
            return self.keys[key_index], current_count + 1

# --- Global state ---
key_rotator = None
client_session = None

# --- Ultra-Fast Request Handler ---
async def handle_chat_completions(request):
    """Lightning-fast chat completions handler"""
    global key_rotator, client_session
    
    # Get API key and request number atomically
    current_key, request_num = key_rotator.get_current_key_and_increment()
    
    # Log without blocking
    print(f"➡️ Request #{request_num} | Key: ...{current_key[-4:]}")
    
    try:
        # Parse request body once
        request_data = await request.json()
        request_data['model'] = MODEL_TO_PROXY  # <-- Force model here
        
        # Prepare headers
        headers = {
            'Authorization': f'Bearer {current_key}',
            'Content-Type': 'application/json'
        }
        
        # Handle streaming vs non-streaming
        is_streaming = request_data.get('stream', False)
        
        if is_streaming:
            return await handle_streaming_request(request_data, headers)
        else:
            return await handle_non_streaming_request(request_data, headers)
            
    except Exception as e:
        print(f"❌ Request #{request_num} failed: {e}")
        return web.json_response(
            {"error": {"message": str(e), "type": "proxy_error"}}, 
            status=500
        )

async def handle_non_streaming_request(request_data, headers):
    """Handle non-streaming requests with maximum speed"""
    global client_session
    
    async with client_session.post(
        f"{CODESTRAL_BASE_URL}/chat/completions",
        headers=headers,
        json=request_data
    ) as response:
        response_data = await response.json()
        return web.json_response(response_data, status=response.status)

async def handle_streaming_request(request_data, headers):
    """Handle streaming requests with zero-copy streaming"""
    global client_session
    
    response = web.StreamResponse(
        status=200,
        headers={
            'Content-Type': 'text/plain; charset=utf-8',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
    )
    
    await response.prepare(request)
    
    try:
        async with client_session.post(
            f"{CODESTRAL_BASE_URL}/chat/completions",
            headers=headers,
            json=request_data
        ) as upstream_response:
            
            # Stream data with minimal overhead
            async for chunk in upstream_response.content.iter_chunked(8192):
                await response.write(chunk)
                
    except Exception as e:
        await response.write(f"data: {json.dumps({'error': str(e)})}\n\n".encode())
    
    await response.write_eof()
    return response

async def handle_models(request):
    """Ultra-fast models endpoint"""
    return web.json_response({
        "object": "list",
        "data": [{
            "id": "codestral-latest",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "mistral"
        }]
    })

async def handle_health(request):
    """Ultra-fast health check"""
    global key_rotator
    return web.json_response({
        "status": "healthy",
        "total_requests": key_rotator._request_count,
        "available_keys": len(key_rotator.keys),
        "current_key_suffix": f"...{key_rotator.keys[0][-4:]}"
    })

# --- Application Setup ---
async def create_app():
    """Create optimized aiohttp application"""
    app = web.Application()
    
    # Add routes
    app.router.add_post('/v1/chat/completions', handle_chat_completions)
    app.router.add_get('/v1/models', handle_models)
    app.router.add_get('/health', handle_health)
    
    return app

async def init_client_session():
    """Initialize high-performance HTTP client session"""
    global client_session
    
    timeout = ClientTimeout(
        total=CONNECTION_TIMEOUT,
        connect=10,
        sock_read=READ_TIMEOUT
    )
    
    connector = aiohttp.TCPConnector(
        limit=MAX_CONNECTIONS,
        limit_per_host=100,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=30,
        enable_cleanup_closed=True
    )
    
    client_session = ClientSession(
        connector=connector,
        timeout=timeout,
        headers={'User-Agent': 'CodestralProxy/1.0'}
    )
    
    print(f"🚀 HTTP Client initialized with {MAX_CONNECTIONS} max connections")

async def cleanup():
    """Cleanup resources"""
    global client_session
    if client_session:
        await client_session.close()

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"\n🛑 Received signal {signum}, shutting down...")
    asyncio.create_task(cleanup())
    sys.exit(0)

# --- Main Execution ---
async def main():
    global key_rotator
    
    print("⚡ Starting Ultra-Fast Codestral Proxy Server...")
    print("💡 Using asyncio + aiohttp for maximum performance")
    
    # Load API keys
    api_keys = load_keys_from_google_sheet(GOOGLE_SHEET_URL)

    if not api_keys:
        print("❌ No API keys loaded. Exiting.")
        return
    valid_keys, invalid_keys = await check_api_keys_validity(api_keys)
    print('VALID TOKENS')
    print(valid_keys)
    print('INVALID TOKENS')
    print(invalid_keys)

    # Initialize key rotator
    key_rotator = UltraFastKeyRotator(keys=valid_keys, rotate_every_n_requests=ROTATION_INTERVAL)
    
    # Initialize HTTP client
    await init_client_session()
    
    # Create application
    app = await create_app()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Performance info
    print(f"\n⚡ ULTRA-FAST PROXY SERVER READY ⚡")
    print(f"🌐 Listening on: http://{HOST}:{PORT}")
    print(f"🔄 Key rotation: Every {ROTATION_INTERVAL} requests")
    print(f"🔑 Loaded keys: {len(api_keys)}")
    print(f"🚀 Max connections: {MAX_CONNECTIONS}")
    print(f"⏱️  Connection timeout: {CONNECTION_TIMEOUT}s")
    print(f"📊 Health: http://{HOST}:{PORT}/health")
    print(f"📋 Models: http://{HOST}:{PORT}/v1/models")
    
    print("\n🔧 CURL TEST:")
    print(f"""curl -X POST http://{HOST}:{PORT}/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{{"model": "codestral-latest", "messages": [{{"role": "user", "content": "Hello!"}}]}}'""")
    
    print("\n" + "="*80)
    
    # Start server
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, HOST, PORT)
    await site.start()
    
    # Keep running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        await cleanup()
        await runner.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)