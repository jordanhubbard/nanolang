"""I retain exact structured shadow identities despite progress interleaving."""
import json
import unittest
from tests.test_token_value_bytes import shadow_trace_names


class ShadowTraceNames(unittest.TestCase):
    def test_interleaved_progress_preserves_duplicate_events(self):
        output=(b'Testing tok{"event":"shadow-start","name":"token_in","source":"file.nano"}\n'
                b'en_in... PASSED\n{"event":"shadow-start","name":"token_in"}\n'
                b'{"event":"shadow-complete","name":"token_in"}\n')
        self.assertEqual(shadow_trace_names(output), [b'token_in', b'token_in'])

    def test_progress_is_not_execution_evidence(self):
        self.assertEqual(shadow_trace_names(b'Testing token_in... PASSED\n'), [])

    def test_malformed_event_refuses(self):
        for output in (b'{"event":"shadow-start","name":"bad name"}',
                       b'{"event":"shadow-start","name":',
                       b'{"event":"shadow-start"}'):
            with self.subTest(output=output), self.assertRaises((AssertionError,json.JSONDecodeError,KeyError)):
                shadow_trace_names(output)
