"""HintBus unit tests."""

from __future__ import annotations

import unittest

from antihydra.hints import HintBus


class HintBusTests(unittest.TestCase):
    def test_publish_and_since(self) -> None:
        bus = HintBus()
        first = bus.publish(source="lane-1", text="offset is 40")
        second = bus.publish(source="lane-2", text="canary present")

        self.assertEqual(
            [hint.text for hint in bus.since(0)], ["offset is 40", "canary present"]
        )
        self.assertEqual(
            [hint.text for hint in bus.since(first.id)], ["canary present"]
        )
        self.assertEqual(bus.last_id(), second.id)

    def test_exclude_source_filters_own_hints(self) -> None:
        bus = HintBus()
        bus.publish(source="lane-1", text="a")
        bus.publish(source="lane-2", text="b")

        self.assertEqual(
            [hint.text for hint in bus.since(0, exclude_source="lane-1")], ["b"]
        )

    def test_duplicate_and_cap(self) -> None:
        bus = HintBus(max_per_source=2)
        bus.publish(source="lane-1", text="first")
        with self.assertRaises(ValueError):
            bus.publish(source="lane-1", text="first")
        bus.publish(source="lane-1", text="second")
        with self.assertRaises(ValueError):
            bus.publish(source="lane-1", text="third")

    def test_text_is_normalized_and_clipped(self) -> None:
        bus = HintBus(max_text=10)
        hint = bus.publish(source="lane-1", text="  a   b\n\nc  ")
        self.assertEqual(hint.text, "a b c")
        long_hint = bus.publish(source="lane-1", text="x" * 50)
        self.assertEqual(len(long_hint.text), 10)

    def test_empty_text_is_rejected(self) -> None:
        bus = HintBus()
        with self.assertRaises(ValueError):
            bus.publish(source="lane-1", text="   ")

    def test_stats(self) -> None:
        bus = HintBus()
        bus.publish(source="lane-1", text="a")
        bus.publish(source="lane-2", text="b")

        bus.note_read()
        self.assertEqual(
            bus.stats(), {"count": 2, "sources": {"lane-1": 1, "lane-2": 1}, "reads": 1}
        )

    def test_history_is_bounded(self) -> None:
        bus = HintBus(max_entries=2)
        bus.publish(source="lane-1", text="a")
        bus.publish(source="lane-1", text="b")
        bus.publish(source="lane-1", text="c")

        self.assertEqual([hint.text for hint in bus.since(0)], ["b", "c"])


if __name__ == "__main__":
    unittest.main()
