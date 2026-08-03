from __future__ import annotations

import pytest

from ckp.privacy import PrivacyClass
from ckp.writer.errors import WriterErrorCode, WriterRefusal
from ckp.writer.target import TargetRoute, TargetZone, WriterTargetPolicy


@pytest.mark.parametrize(
    "path,privacy",
    [
        ("Core/_inbox/c7-synthetic/synthetic-public.md", PrivacyClass.PUBLIC),
        (
            "Core/_inbox/c7-synthetic/synthetic-internal.md",
            PrivacyClass.INTERNAL,
        ),
        (
            "Private/_inbox/c7-synthetic/synthetic-private.md",
            PrivacyClass.SENSITIVE,
        ),
    ],
)
def test_only_frozen_synthetic_inbox_pairs_are_authorized(
    path: str,
    privacy: PrivacyClass,
) -> None:
    policy = WriterTargetPolicy.c7_synthetic()
    target = policy.canonicalize(path)

    policy.authorize(target, privacy)
    assert target.path == path


@pytest.mark.parametrize(
    "path",
    [
        "Core/project-formal.md",
        "Core/Publication/synthetic.md",
        "Core/_inbox/direct-formal/synthetic.md",
        "Private/Life/synthetic.md",
        "Other/_inbox/c7-synthetic/synthetic.md",
        "/Core/_inbox/c7-synthetic/synthetic.md",
        "Core/_inbox/c7-synthetic/../escape.md",
        "Core/_inbox/c7-synthetic/./synthetic.md",
        "Core//_inbox/c7-synthetic/synthetic.md",
        "Core\\_inbox\\c7-synthetic\\synthetic.md",
        "Core/_inbox/c7-synthetic/nested/synthetic.md",
        "Core/_inbox/c7-synthetic/.hidden.md",
        "Core/_inbox/c7-synthetic/Unsafe.md",
        "Core/_inbox/c7-synthetic/synthetic name.md",
        "Core/_inbox/c7-synthetic/synthetic\ntrailer.md",
        "Core/_inbox/c7-synthetic/synthetic.txt",
    ],
)
def test_formal_arbitrary_traversal_and_unsafe_targets_are_refused(path: str) -> None:
    with pytest.raises(WriterRefusal) as caught:
        WriterTargetPolicy.c7_synthetic().canonicalize(path)
    assert caught.value.code is WriterErrorCode.TARGET_DENIED


@pytest.mark.parametrize(
    "path,privacy,expected",
    [
        (
            "Core/_inbox/c7-synthetic/synthetic.md",
            PrivacyClass.SENSITIVE,
            WriterErrorCode.CROSS_ZONE_DENIED,
        ),
        (
            "Private/_inbox/c7-synthetic/synthetic.md",
            PrivacyClass.PUBLIC,
            WriterErrorCode.CROSS_ZONE_DENIED,
        ),
        (
            "Private/_inbox/c7-synthetic/synthetic.md",
            PrivacyClass.STUDENT_PRIVATE,
            WriterErrorCode.STUDENT_PRIVATE_DENIED,
        ),
    ],
)
def test_cross_zone_and_student_private_pairs_fail_closed(
    path: str,
    privacy: PrivacyClass,
    expected: WriterErrorCode,
) -> None:
    policy = WriterTargetPolicy.c7_synthetic()
    target = policy.canonicalize(path)
    with pytest.raises(WriterRefusal) as caught:
        policy.authorize(target, privacy)
    assert caught.value.code is expected


def test_injected_policy_cannot_broaden_c7_to_formal_routes() -> None:
    with pytest.raises(WriterRefusal) as caught:
        TargetRoute(
            prefix="Core/Publication/",
            zone=TargetZone.CORE,
            privacy_classes=frozenset({PrivacyClass.PUBLIC, PrivacyClass.INTERNAL}),
        )
    assert caught.value.code is WriterErrorCode.TARGET_DENIED

    with pytest.raises(WriterRefusal) as caught:
        TargetRoute(
            prefix="Core/_inbox/c7-synthetic/",
            zone=TargetZone.CORE,
            privacy_classes=frozenset(
                {
                    PrivacyClass.PUBLIC,
                    PrivacyClass.INTERNAL,
                    PrivacyClass.SENSITIVE,
                }
            ),
        )
    assert caught.value.code is WriterErrorCode.TARGET_DENIED
