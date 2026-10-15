import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from load_trajectories import NotPrefixStable, clean_messages, drop_nulls, session_turns  # noqa: E402


def toy_encode(messages, add_generation_prompt):
    text = "".join(f"<{m['role']}>{m.get('content') or ''}{json.dumps(m.get('tool_calls', ''))}</>" for m in messages)
    if add_generation_prompt:
        text += "<assistant>"
    return [ord(c) for c in text]


def marks_last_message(messages, add_generation_prompt):
    return toy_encode(messages, add_generation_prompt) + [ord("*")]


RAW = [
    {"role": "system", "content": "sys", "name": None, "tool_call_id": None, "tool_calls": None},
    {"role": "user", "content": "fix it", "name": None, "tool_call_id": None, "tool_calls": None},
    {"role": "assistant", "content": "look", "name": None, "tool_call_id": None,
     "tool_calls": [{"function": {"arguments": '{"command": "ls"}', "name": "execute_bash"}, "id": "c1", "type": "function"}]},
    {"role": "tool", "content": "a.py", "name": "execute_bash", "tool_call_id": "c1", "tool_calls": None},
    {"role": "assistant", "content": "done", "name": None, "tool_call_id": None, "tool_calls": None},
]


class Loader(unittest.TestCase):
    def test_clean_messages(self):
        msgs = clean_messages(RAW)
        self.assertEqual(set(msgs[0]), {"role", "content"})
        self.assertEqual(set(msgs[3]), {"role", "content", "name", "tool_call_id"})
        self.assertEqual(msgs[2]["tool_calls"][0]["function"]["arguments"], {"command": "ls"})
        self.assertNotIn("tool_calls", msgs[4])

    def test_turns_share_one_sequence(self):
        msgs = clean_messages(RAW)
        seq, turns = session_turns(msgs, toy_encode, max_context=10_000, max_answer=1024)
        self.assertEqual(len(turns), 2)
        p0, p1 = turns[0]["prompt_len"], turns[1]["prompt_len"]
        self.assertEqual(seq[:p0], toy_encode(msgs[:2], True))
        self.assertEqual(seq[:p1], toy_encode(msgs[:4], True))
        self.assertEqual(len(seq), p1 + turns[1]["answer_len"])

    def test_context_cap_truncates(self):
        msgs = clean_messages(RAW)
        _, full = session_turns(msgs, toy_encode, 10_000, 1024)
        cap = full[1]["prompt_len"] + full[1]["answer_len"] - 1
        _, turns = session_turns(msgs, toy_encode, cap, 1024)
        self.assertEqual(len(turns), 1)

    def test_answer_cap_counts_toward_context(self):
        msgs = clean_messages(RAW)
        _, full = session_turns(msgs, toy_encode, 10_000, 1024)
        cap = full[1]["prompt_len"] + 1
        self.assertEqual(len(session_turns(msgs, toy_encode, cap, 1)[1]), 2)
        self.assertEqual(len(session_turns(msgs, toy_encode, cap, 2)[1]), 1)

    def test_rejects_unstable_template(self):
        with self.assertRaises(NotPrefixStable):
            session_turns(clean_messages(RAW), marks_last_message, 10_000, 1024)

    def test_drop_nulls(self):
        self.assertEqual(drop_nulls({"a": None, "b": [{"c": None, "d": 1}]}), {"b": [{"d": 1}]})


if __name__ == "__main__":
    unittest.main()
