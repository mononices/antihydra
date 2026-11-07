"""Dataset overrides: corrections live in our repo, never in the dataset."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import antihydra.challenge as challenge_module
from antihydra.challenge import ChallengeError, discover, load, load_overrides

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None

REPO_ROOT = Path(__file__).resolve().parent.parent


def write_challenge(root: Path, category: str, directory: str, metadata: dict) -> Path:
    path = root / category / directory
    path.mkdir(parents=True)
    (path / "challenge.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


class OverrideTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_override_corrects_name_and_files(self) -> None:
        directory = write_challenge(
            self.root,
            "web",
            "gradebook",
            {"name": "power-up", "category": "web", "files": ["dist/missing"]},
        )

        challenge = load(
            directory, overrides={"web/gradebook": {"name": "gradebook", "files": []}}
        )

        self.assertEqual(challenge.name, "gradebook")
        self.assertEqual(challenge.files, ())

    def test_override_cannot_set_the_flag(self) -> None:
        directory = write_challenge(self.root, "crypto", "x", {"category": "crypto"})

        with self.assertRaises(ChallengeError):
            load(directory, overrides={"crypto/x": {"flag": "nope"}})

    def test_missing_file_fails_loudly_with_override_hint(self) -> None:
        directory = write_challenge(
            self.root,
            "pwn",
            "mooneys-bookstore",
            {"category": "pwn", "files": ["infra/chal"]},
        )
        challenge = load(directory, overrides={})

        with self.assertRaises(ChallengeError) as caught:
            challenge.require_delivered_files()

        message = str(caught.exception)
        self.assertIn("infra/chal", message)
        self.assertIn("overrides.toml", message)

    def test_override_repairs_a_missing_file(self) -> None:
        directory = write_challenge(
            self.root,
            "pwn",
            "mooneys-bookstore",
            {"category": "pwn", "files": ["infra/chal"]},
        )
        (directory / "infra").mkdir()
        (directory / "infra" / "overflow_me").write_bytes(b"\x7fELF")

        challenge = load(
            directory,
            overrides={"pwn/mooneys-bookstore": {"files": ["infra/overflow_me"]}},
        )

        challenge.require_delivered_files()
        self.assertEqual(challenge.missing_files(), [])

    @unittest.skipIf(tomllib is None, "tomllib needs Python 3.11+")
    def test_load_overrides_rejects_unknown_fields(self) -> None:
        path = self.root / "overrides.toml"
        path.write_text('["a/b"]\nflag = "x"\n', encoding="utf-8")

        with self.assertRaises(ChallengeError):
            load_overrides(path)

    @unittest.skipIf(tomllib is None, "tomllib needs Python 3.11+")
    def test_discover_applies_file_and_warns_about_stale_keys(self) -> None:
        write_challenge(
            self.root, "crypto", "x", {"name": "wrong", "category": "crypto"}
        )
        write_challenge(self.root, "misc", "y", {"category": "misc"})
        overrides_path = self.root / "overrides.toml"
        overrides_path.write_text(
            '["crypto/x"]\nname = "x"\n\n["ghost/missing"]\nfiles = []\n',
            encoding="utf-8",
        )

        original = challenge_module.OVERRIDES_FILE
        challenge_module.OVERRIDES_FILE = overrides_path
        try:
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                challenges = discover(self.root)
        finally:
            challenge_module.OVERRIDES_FILE = original

        self.assertIn("x", {challenge.name for challenge in challenges})
        self.assertIn("ghost/missing", stderr.getvalue())

    def test_shipped_dataset_matches_its_overrides(self) -> None:
        challenges = discover(REPO_ROOT / "competition" / "2025")

        names = [challenge.name.lower() for challenge in challenges]
        self.assertEqual(len(names), len(set(names)), "duplicate challenge names")
        for challenge in challenges:
            self.assertEqual(
                challenge.missing_files(), [], f"{challenge.category}/{challenge.name}"
            )


if __name__ == "__main__":
    unittest.main()
