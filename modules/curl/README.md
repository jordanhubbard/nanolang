# libcurl Module for nanolang

HTTP/HTTPS client for making web requests, REST API calls, and file downloads.

## Installation

**macOS:**
```bash
brew install curl
```

**Ubuntu/Debian:**
```bash
sudo apt install libcurl4-openssl-dev
```

## Usage

```nano
import "modules/curl/curl.nano"

fn main() -> int {
    # Simple GET request
    let response: string = (nl_curl_simple_get "https://api.github.com")
    (println response)
    
    # Simple POST request
    let data: string = "{\"key\": \"value\"}"
    let post_response: string = (nl_curl_simple_post "https://httpbin.org/post" data)
    (println post_response)
    
    # Download file
    let result: int = (nl_curl_download_file "https://example.com/file.txt" "output.txt")
    
    return 0
}

shadow main {
    # Skip - uses extern functions
}
```

## Features

- **Simple HTTP requests**: GET and POST with one function call
- **File downloads**: Direct URL to file downloads
- **Advanced configuration**: Custom headers, timeouts, redirects
- **Response codes**: Get HTTP status codes
- **HTTPS support**: Secure connections built-in

## Example

See `examples/curl_example.nano` for comprehensive usage examples.

## API Reference

### Simple Functions
- `nl_curl_simple_get(url: string) -> string` - Perform GET request
- `nl_curl_simple_post(url: string, data: string) -> string` - Perform POST request
- `nl_curl_download_file(url: string, output: string) -> int` - Download file

### Advanced Functions
- `nl_curl_global_init() -> int` - Initialize curl library
- `nl_curl_global_cleanup() -> void` - Cleanup curl library
- `nl_curl_easy_init() -> int` - Create curl handle
- `nl_curl_easy_cleanup(handle: int) -> void` - Destroy curl handle
- `nl_curl_easy_setopt_*()` - Configuration functions
- `nl_curl_easy_perform(handle: int) -> int` - Execute request
- `nl_curl_easy_getinfo_response_code(handle: int) -> int` - Get HTTP code
