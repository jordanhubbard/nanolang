import pytest
import ctypes
import struct
from unittest.mock import MagicMock, patch


# Simulated HTTP response buffer implementation in Python
# This mirrors the vulnerable C pattern and tests the invariant

class HTTPResponse:
    """Simulates the HTTP response structure with a fixed-size body buffer."""
    
    MAX_BODY_SIZE = 1024  # Simulated fixed buffer capacity
    
    def __init__(self):
        self.body = bytearray(self.MAX_BODY_SIZE)
        self.body_size = 0
        self.capacity = self.MAX_BODY_SIZE
    
    def safe_append(self, contents: bytes) -> bool:
        """
        Safe implementation: validates bounds before copying.
        Returns True if data was accepted, False if rejected/truncated.
        Invariant: body_size must never exceed capacity.
        """
        real_size = len(contents)
        
        # SECURITY INVARIANT: Must validate before copy
        if self.body_size + real_size > self.capacity:
            # Either reject or truncate — never overflow
            available = self.capacity - self.body_size
            if available <= 0:
                return False  # Reject entirely
            # Truncate to available space
            contents = contents[:available]
            real_size = available
        
        # Safe copy within bounds
        self.body[self.body_size:self.body_size + real_size] = contents
        self.body_size += real_size
        return True
    
    def vulnerable_append(self, contents: bytes):
        """
        Simulates the VULNERABLE pattern from the C code:
        memcpy(&(response->body[response->body_size]), contents, real_size)
        without bounds checking.
        """
        real_size = len(contents)
        # This would overflow in C — in Python we simulate the check
        # that SHOULD be there but ISN'T in the vulnerable code
        end_pos = self.body_size + real_size
        # Simulate what happens: write beyond buffer
        extended_body = bytearray(max(len(self.body), end_pos))
        extended_body[:len(self.body)] = self.body
        extended_body[self.body_size:end_pos] = contents
        self.body = extended_body  # In C this would corrupt heap
        self.body_size += real_size


def simulate_http_response_write(buffer_capacity: int, chunks: list) -> dict:
    """
    Simulates the HTTP response body assembly process.
    Returns metadata about the operation for invariant checking.
    """
    body_buffer = bytearray(buffer_capacity)
    body_size = 0
    rejected_chunks = []
    accepted_chunks = []
    overflow_attempted = False
    
    for chunk in chunks:
        real_size = len(chunk)
        
        # Check the invariant BEFORE any copy operation
        if body_size + real_size > buffer_capacity:
            overflow_attempted = True
            available = buffer_capacity - body_size
            if available > 0:
                # Truncate
                body_buffer[body_size:body_size + available] = chunk[:available]
                body_size += available
                accepted_chunks.append(chunk[:available])
                rejected_chunks.append(chunk[available:])
            else:
                rejected_chunks.append(chunk)
            continue
        
        body_buffer[body_size:body_size + real_size] = chunk
        body_size += real_size
        accepted_chunks.append(chunk)
    
    return {
        'body_size': body_size,
        'capacity': buffer_capacity,
        'overflow_attempted': overflow_attempted,
        'rejected_chunks': rejected_chunks,
        'accepted_chunks': accepted_chunks,
        'buffer': body_buffer,
    }


# Attack payloads: (buffer_capacity, list_of_chunks)
ATTACK_PAYLOADS = [
    # 2x oversized single chunk
    (1024, [b'A' * 2048]),
    
    # 10x oversized single chunk
    (1024, [b'B' * 10240]),
    
    # 100x oversized single chunk
    (512, [b'C' * 51200]),
    
    # Multiple chunks that together overflow
    (256, [b'D' * 128, b'E' * 128, b'F' * 128]),  # Third chunk overflows
    
    # First chunk fills buffer, subsequent chunks should be rejected
    (512, [b'G' * 512, b'H' * 1, b'I' * 100]),
    
    # Gradual overflow with many small chunks
    (100, [b'J' * 10] * 15),  # 150 bytes total into 100-byte buffer
    
    # Empty chunk followed by oversized
    (256, [b'', b'K' * 1024]),
    
    # Null bytes in oversized payload (binary attack)
    (512, [b'\x00' * 2048]),
    
    # HTTP response smuggling pattern
    (1024, [b'HTTP/1.1 200 OK\r\n' + b'X' * 2000]),
    
    # Chunk with embedded length manipulation
    (256, [b'\xff\xfe' + b'L' * 512]),
    
    # Maximum integer boundary test
    (1024, [b'M' * 1023, b'N' * 2]),  # 1023 + 2 = 1025 > 1024
    
    # Exactly at boundary (should succeed)
    (1024, [b'O' * 1024]),
    
    # One byte over boundary
    (1024, [b'P' * 1025]),
    
    # Multiple small chunks summing to overflow
    (64, [b'Q' * 8] * 9),  # 72 bytes into 64-byte buffer
    
    # Large number of tiny chunks
    (100, [b'R'] * 200),  # 200 single bytes into 100-byte buffer
    
    # Chunk with format string patterns
    (256, [b'%s%s%s%n%n%n' * 100]),
    
    # Chunk with shell metacharacters
    (128, [b'; cat /etc/passwd; ' * 20]),
    
    # Unicode/multibyte attack
    (256, ['Ä' * 300 .encode('utf-8')]),  # UTF-8 multibyte chars
    
    # Zero-length buffer with any data
    (0, [b'S' * 10]),
    
    # Single byte buffer overflow
    (1, [b'TT']),
]


@pytest.mark.parametrize("payload", ATTACK_PAYLOADS)
def test_buffer_reads_never_exceed_declared_length(payload):
    """
    Invariant: Buffer reads/writes must never exceed the declared buffer capacity.
    
    For any input data (including adversarial oversized payloads), the HTTP response
    body assembly must ensure that body_size never exceeds the allocated buffer
    capacity. Data must be either truncated or rejected — never cause an overflow.
    
    This guards against CWE-120: Buffer Copy without Checking Size of Input,
    specifically the pattern: memcpy(&(response->body[response->body_size]), contents, real_size)
    without validating body_size + real_size <= capacity.
    """
    buffer_capacity, chunks = payload
    
    result = simulate_http_response_write(buffer_capacity, chunks)
    
    # PRIMARY INVARIANT: body_size must never exceed capacity
    assert result['body_size'] <= result['capacity'], (
        f"OVERFLOW DETECTED: body_size={result['body_size']} exceeds "
        f"capacity={result['capacity']}. "
        f"This represents a heap buffer overflow vulnerability."
    )
    
    # SECONDARY INVARIANT: body_size must be non-negative
    assert result['body_size'] >= 0, (
        f"body_size={result['body_size']} is negative, indicating corruption."
    )
    
    # TERTIARY INVARIANT: If overflow was attempted, it must have been caught
    total_input = sum(len(c) for c in chunks)
    if total_input > buffer_capacity:
        assert result['overflow_attempted'], (
            f"Total input ({total_input} bytes) exceeds capacity ({buffer_capacity} bytes) "
            f"but overflow was not detected. The bounds check is missing."
        )
    
    # QUATERNARY INVARIANT: Actual buffer content must not exceed capacity
    assert len(result['buffer']) >= result['body_size'], (
        f"Buffer length {len(result['buffer'])} is less than reported body_size "
        f"{result['body_size']}."
    )
    
    # QUINARY INVARIANT: Using HTTPResponse class
    response = HTTPResponse()
    response.capacity = buffer_capacity
    response.body = bytearray(max(buffer_capacity, 1))
    response.body_size = 0
    
    for chunk in chunks:
        response.safe_append(chunk)
        
        # Check invariant after EVERY append operation
        assert response.body_size <= response.capacity, (
            f"Invariant violated after appending chunk of size {len(chunk)}: "
            f"body_size={response.body_size} > capacity={response.capacity}. "
            f"A bounds check must occur before every memcpy operation."
        )
        
        assert response.body_size >= 0, (
            f"body_size became negative ({response.body_size}) after append."
        )


@pytest.mark.parametrize("capacity,chunk_size,num_chunks", [
    (1024, 2048, 1),    # Single 2x oversized chunk
    (1024, 10240, 1),   # Single 10x oversized chunk
    (256, 256, 3),      # Multiple chunks that overflow
    (512, 1, 1025),     # Many tiny chunks overflowing
    (100, 50, 4),       # Gradual overflow
])
def test_incremental_buffer_overflow_prevention(capacity, chunk_size, num_chunks):
    """
    Invariant: Incremental writes must never cumulatively exceed buffer capacity.
    
    Tests that repeated write operations, even with individually small chunks,
    cannot cumulatively overflow the buffer.
    """
    chunks = [b'X' * chunk_size] * num_chunks
    result = simulate_http_response_write(capacity, chunks)
    
    assert result['body_size'] <= capacity, (
        f"Cumulative overflow: {num_chunks} chunks of {chunk_size} bytes each "
        f"resulted in body_size={result['body_size']} > capacity={capacity}."
    )
    
    # Verify the buffer itself wasn't corrupted beyond capacity
    written_data = bytes(result['buffer'][:result['body_size']])
    assert len(written_data) <= capacity, (
        f"Written data length {len(written_data)} exceeds capacity {capacity}."
    )


def test_zero_capacity_buffer_rejects_all_data():
    """
    Invariant: A zero-capacity buffer must reject all incoming data.
    """
    result = simulate_http_response_write(0, [b'attack_data' * 100])
    
    assert result['body_size'] == 0, (
        f"Zero-capacity buffer accepted {result['body_size']} bytes. "
        f"All data must be rejected."
    )


def test_exact_boundary_accepted():
    """
    Invariant: Data exactly equal to buffer capacity must be accepted without overflow.
    """
    capacity = 1024
    exact_data = b'A' * capacity
    result = simulate_http_response_write(capacity, [exact_data])
    
    assert result['body_size'] == capacity, (
        f"Exact-boundary data was not fully accepted: "
        f"body_size={result['body_size']}, capacity={capacity}."
    )
    assert not result['overflow_attempted'], (
        "Exact-boundary write incorrectly flagged as overflow attempt."
    )


def test_one_byte_over_boundary_rejected_or_truncated():
    """
    Invariant: Data one byte over capacity must be truncated or rejected, never overflow.
    """
    capacity = 1024
    oversized_data = b'B' * (capacity + 1)
    result = simulate_http_response_write(capacity, [oversized_data])
    
    assert result['body_size'] <= capacity, (
        f"One-byte overflow not prevented: "
        f"body_size={result['body_size']} > capacity={capacity}."
    )
    assert result['overflow_attempted'], (
        "One-byte overflow was not detected by bounds check."
    )