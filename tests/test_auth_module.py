import tempfile
import unittest
from pathlib import Path

from app.observability.audit import AuditLog
from app.security.auth import (
    AccessDeniedThrottle,
    AuthenticatedUser,
    load_api_keys,
    resolve_user,
)


class ApiKeyTests(unittest.TestCase):
    def test_keys_are_parsed_into_owner_mapping(self) -> None:
        keys = load_api_keys("owner-a:key-a, owner-b:key-b")

        self.assertEqual(keys, {"key-a": "owner-a", "key-b": "owner-b"})

    def test_malformed_entries_are_ignored(self) -> None:
        keys = load_api_keys("owner-a:key-a,bad-entry,:missing-user,user:")

        self.assertEqual(keys, {"key-a": "owner-a"})

    def test_resolve_user_uses_exact_secret(self) -> None:
        keys = load_api_keys("owner-a:key-a")

        self.assertEqual(resolve_user("key-a", keys), AuthenticatedUser("owner-a"))
        self.assertIsNone(resolve_user("key-b", keys))
        self.assertIsNone(resolve_user(None, keys))
        self.assertIsNone(resolve_user("", keys))

    def test_missing_configuration_yields_no_identities(self) -> None:
        self.assertEqual(load_api_keys(""), {})

    def test_malformed_configuration_yields_no_identities(self) -> None:
        self.assertEqual(load_api_keys("garbage,no-colon,:owner,owner:"), {})

    def test_ambiguous_secret_is_dropped_entirely(self) -> None:
        keys = load_api_keys("owner-a:shared,owner-b:shared,owner-c:other")

        self.assertEqual(keys, {"other": "owner-c"})
        self.assertIsNone(resolve_user("shared", keys))

    def test_repeated_secret_for_same_owner_is_kept(self) -> None:
        keys = load_api_keys("owner-a:key-a,owner-a:key-a")

        self.assertEqual(keys, {"key-a": "owner-a"})

    def test_malformed_configuration_logs_counts_without_secrets(self) -> None:
        with self.assertLogs("app.security.auth", level="WARNING") as captured:
            keys = load_api_keys(
                "owner-a:secret-a,bad-entry,owner-b:secret-b,owner-b:secret-b"
            )

        self.assertEqual(keys, {"secret-a": "owner-a", "secret-b": "owner-b"})
        message = "\n".join(captured.output)
        self.assertIn("malformed", message)
        self.assertNotIn("secret-a", message)
        self.assertNotIn("secret-b", message)
        self.assertNotIn("owner-a", message)
        self.assertNotIn("owner-b", message)

    def test_ambiguous_configuration_logs_counts_without_secrets(self) -> None:
        with self.assertLogs("app.security.auth", level="WARNING") as captured:
            keys = load_api_keys("owner-a:shared-secret,owner-b:shared-secret")

        self.assertEqual(keys, {})
        message = "\n".join(captured.output)
        self.assertIn("ambiguous", message)
        self.assertNotIn("shared-secret", message)
        self.assertNotIn("owner-a", message)
        self.assertNotIn("owner-b", message)


class AccessDeniedThrottleTests(unittest.TestCase):
    def test_denials_are_limited_to_one_per_interval(self) -> None:
        now = {"value": 100.0}
        throttle = AccessDeniedThrottle(
            interval_seconds=5.0,
            clock=lambda: now["value"],
        )

        self.assertTrue(throttle.allow("/documents"))
        self.assertFalse(throttle.allow("/documents"))
        self.assertFalse(throttle.allow("/sessions"))

        now["value"] = 104.999
        self.assertFalse(throttle.allow("/documents"))

        now["value"] = 105.0
        self.assertTrue(throttle.allow("/documents"))


class AuditLogTests(unittest.TestCase):
    def test_events_are_recorded_with_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit = AuditLog(Path(directory) / "audit.db")
            event = audit.record(
                actor="owner-a",
                action="document.delete",
                resource_id="doc-1",
                outcome="success",
            )
            events = audit.list_events(actor="owner-a")
            other_actor_events = audit.list_events(actor="owner-b")
            audit.close()

        self.assertTrue(event["event_id"])
        self.assertTrue(event["created_at"])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["action"], "document.delete")
        self.assertEqual(events[0]["resource_id"], "doc-1")
        self.assertEqual(other_actor_events, [])

    def test_count_tracks_recorded_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit = AuditLog(Path(directory) / "audit.db")
            audit.record(actor="a", action="one")
            audit.record(actor="b", action="two")
            count = audit.count()
            audit.close()

        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
