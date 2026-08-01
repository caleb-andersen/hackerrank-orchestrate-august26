"""Acceptance-focused tests for deterministic dossier features."""

import sys
import unittest
from collections import Counter
from datetime import datetime
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config import (  # noqa: E402
    BRAND_MAX_REPORTS,
    BRAND_MIN_AGE_DAYS,
    BRAND_MIN_DOMAIN_AGE_DAYS,
)
from context.features import brand_integrity, build_dossier  # noqa: E402
from context.index import build_feature_index  # noqa: E402
from context.timewindow import dnd_state, parse_dnd_window  # noqa: E402
from data.loader import Dataset, load_dataset  # noqa: E402


class FeatureTests(unittest.TestCase):
    dataset: Dataset

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(REPO_ROOT / "dataset")
        cls.index = build_feature_index(cls.dataset)
        cls.dossiers = tuple(
            build_dossier(cls.dataset, cls.index, message)
            for message in cls.dataset.messages
        )

    def test_peer_with_history_has_observed_rates(self) -> None:
        dossier = next(
            built for built in self.dossiers
            if built.relationship.peer_engagement.n > 0
        )
        engagement = dossier.relationship.peer_engagement
        self.assertGreater(engagement.n, 0)
        self.assertEqual(dossier.relationship.evidence_state, "peer")
        self.assertIsNotNone(engagement.open_rate)
        self.assertIsNotNone(engagement.reply_rate)
        self.assertIsNotNone(engagement.dismiss_rate)
        self.assertIsNotNone(engagement.mute_rate)
        self.assertIsNotNone(engagement.report_rate)

    def test_peer_with_no_history_has_null_rates(self) -> None:
        dossier = next(
            built for built in self.dossiers
            if built.relationship.peer_engagement.n == 0
        )
        engagement = dossier.relationship.peer_engagement
        self.assertEqual(engagement.n, 0)
        self.assertEqual(engagement.n_reacted, 0)
        self.assertIsNone(engagement.open_rate)
        self.assertIsNone(engagement.reply_rate)
        self.assertIsNone(engagement.dismiss_rate)
        self.assertIsNone(engagement.mute_rate)
        self.assertIsNone(engagement.report_rate)
        self.assertIsNone(engagement.median_reaction_minutes)
        self.assertIn(
            dossier.relationship.evidence_state,
            {"global_fallback", "none"},
        )

    def test_empty_message_text_keeps_nonlexical_evidence(self) -> None:
        dossier = next(
            built for built in self.dossiers
            if not built.content_signals.raw_text.strip()
        )
        signals = dossier.content_signals
        self.assertEqual(signals.raw_text, "")
        self.assertEqual(signals.normalised_text, "")
        self.assertEqual(signals.text_length, 0)
        self.assertTrue(signals.is_empty_text)
        self.assertFalse(signals.text_scanned)
        self.assertIsNone(signals.injection_match)
        self.assertIsNone(signals.credential_request)
        self.assertIsNone(signals.payment_pressure)
        self.assertEqual(signals.url_domains, ())
        self.assertEqual(dossier.repetition.near_duplicate_history, ())
        self.assertIsNone(dossier.repetition.max_jaccard)
        self.assertEqual(dossier.repetition.duplicate_count_at_threshold, 0)
        self.assertTrue(dossier.media.requires_transcription)
        self.assertGreater(len(dossier.evidence_candidates), 0)

    def test_absent_official_domain_is_not_comparable_or_impersonation(self) -> None:
        accounts = tuple(
            account
            for account in self.dataset.business_accounts
            if account.official_domain is None
        )
        self.assertEqual(len(accounts), 5)
        for account in accounts:
            with self.subTest(account=account.display_name):
                integrity = brand_integrity(account)
                self.assertIsNone(integrity.official_domain)
                self.assertIsNone(integrity.domain_mismatch)
                self.assertNotEqual(integrity.verdict, "impersonation")
                self.assertIn("official_domain_absent", integrity.verdict_basis)

    def test_three_precision_traps_test_negative_for_impersonation(self) -> None:
        traps = tuple(
            (account, integrity)
            for account in self.dataset.business_accounts
            if (integrity := brand_integrity(account)).domain_mismatch is True
            and not account.verified
            and account.account_age_days < BRAND_MIN_AGE_DAYS
            and account.domain_used_by_sender_age_days < BRAND_MIN_DOMAIN_AGE_DAYS
            and account.user_reports_30d <= BRAND_MAX_REPORTS
        )
        self.assertEqual(len(traps), 3)
        for account, integrity in traps:
            with self.subTest(account=account.display_name):
                self.assertNotEqual(integrity.verdict, "impersonation")
                self.assertEqual(integrity.verdict, "suspect")

    def test_brand_verdict_partition_matches_spec(self) -> None:
        partition = Counter(
            brand_integrity(account).verdict
            for account in self.dataset.business_accounts
        )
        self.assertEqual(
            partition,
            Counter({"clean": 82, "suspect": 10, "impersonation": 18}),
        )
        self.assertFalse(
            any(
                integrity.domain_mismatch is True and integrity.verdict == "clean"
                for account in self.dataset.business_accounts
                if (integrity := brand_integrity(account))
            )
        )

    def test_dataset_wide_degenerate_counts_and_rate_denominators(self) -> None:
        no_peer = tuple(
            dossier
            for dossier in self.dossiers
            if dossier.relationship.peer_engagement.n == 0
        )
        self.assertEqual(len(no_peer), 6)
        self.assertEqual(
            Counter(dossier.relationship.evidence_state for dossier in no_peer),
            Counter({"global_fallback": 3, "none": 3}),
        )
        unscanned = tuple(
            dossier
            for dossier in self.dossiers
            if not dossier.content_signals.text_scanned
        )
        self.assertEqual(len(unscanned), 8)
        self.assertTrue(all(dossier.media.media_type == "voice" for dossier in unscanned))

        for dossier in self.dossiers:
            for engagement in (
                dossier.relationship.peer_engagement,
                dossier.relationship.peer_global,
            ):
                rates = (
                    engagement.open_rate,
                    engagement.reply_rate,
                    engagement.dismiss_rate,
                    engagement.mute_rate,
                    engagement.report_rate,
                )
                if engagement.n == 0:
                    self.assertTrue(all(rate is None for rate in rates))
            baseline = dossier.relationship.user_baseline
            if baseline.notifications_sent_30d == 0:
                self.assertIsNone(baseline.baseline_dismiss_rate)
            if baseline.n_summary_days == 0:
                self.assertIsNone(baseline.mean_daily_notifications)

    def test_dnd_wrapping_boundaries_and_dataset_invariant(self) -> None:
        wrapped_users = tuple(
            user
            for user in self.dataset.users
            if (
                (window := parse_dnd_window(user.do_not_disturb_window))
                is not None
                and window[0] > window[1]
            )
        )
        self.assertEqual(len(wrapped_users), 49)
        window = parse_dnd_window("22:00-07:00")
        self.assertEqual(
            dnd_state(datetime.fromisoformat("2026-08-01T23:00:00"), window),
            (True, 480),
        )
        self.assertEqual(
            dnd_state(datetime.fromisoformat("2026-08-01T06:00:00"), window),
            (True, 60),
        )
        self.assertEqual(
            dnd_state(datetime.fromisoformat("2026-08-01T22:00:00"), window),
            (True, 540),
        )
        self.assertEqual(
            dnd_state(datetime.fromisoformat("2026-08-01T07:00:00"), window),
            (False, None),
        )
        self.assertEqual(
            dnd_state(datetime.fromisoformat("2026-08-01T06:59:00"), window),
            (True, 1),
        )
        for dossier in self.dossiers:
            if dossier.timing.in_dnd:
                self.assertIsNotNone(dossier.timing.minutes_until_dnd_ends)
                self.assertGreaterEqual(dossier.timing.minutes_until_dnd_ends or 0, 1)
                self.assertLessEqual(dossier.timing.minutes_until_dnd_ends or 1440, 1439)
            else:
                self.assertIsNone(dossier.timing.minutes_until_dnd_ends)

    def test_brand_threshold_partition_is_robust(self) -> None:
        baseline = frozenset(
            account.business_id
            for account in self.dataset.business_accounts
            if brand_integrity(account).verdict == "impersonation"
        )

        def partition(max_reports: int, min_age: int, min_domain_age: int) -> frozenset[str]:
            return frozenset(
                account.business_id
                for account in self.dataset.business_accounts
                if brand_integrity(account).domain_mismatch is True
                and not account.verified
                and account.account_age_days < min_age
                and account.domain_used_by_sender_age_days < min_domain_age
                and account.user_reports_30d > max_reports
            )

        variants = (
            (22, BRAND_MIN_AGE_DAYS, BRAND_MIN_DOMAIN_AGE_DAYS),
            (36, BRAND_MIN_AGE_DAYS, BRAND_MIN_DOMAIN_AGE_DAYS),
            (BRAND_MAX_REPORTS, 274, BRAND_MIN_DOMAIN_AGE_DAYS),
            (BRAND_MAX_REPORTS, 456, BRAND_MIN_DOMAIN_AGE_DAYS),
            (BRAND_MAX_REPORTS, BRAND_MIN_AGE_DAYS, 135),
            (BRAND_MAX_REPORTS, BRAND_MIN_AGE_DAYS, 225),
        )
        self.assertEqual(len(baseline), 18)
        for variant in variants:
            with self.subTest(thresholds=variant):
                self.assertEqual(partition(*variant), baseline)

    def test_dossier_construction_is_deterministic(self) -> None:
        message = self.dataset.messages[0]
        self.assertEqual(
            build_dossier(self.dataset, self.index, message),
            build_dossier(self.dataset, self.index, message),
        )


if __name__ == "__main__":
    unittest.main()
