"""C7's closed synthetic ``_inbox`` target policy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from ckp.privacy import PrivacyClass
from ckp.writer.errors import WriterErrorCode, WriterRefusal


class TargetZone(StrEnum):
    CORE = "core"
    PRIVATE = "private"


_SAFE_LEAF = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}\.md$")
_C7_PREFIXES = {
    TargetZone.CORE: "Core/_inbox/c7-synthetic/",
    TargetZone.PRIVATE: "Private/_inbox/c7-synthetic/",
}
_C7_PRIVACY_CLASSES = {
    TargetZone.CORE: frozenset({PrivacyClass.PUBLIC, PrivacyClass.INTERNAL}),
    TargetZone.PRIVATE: frozenset({PrivacyClass.INTERNAL, PrivacyClass.SENSITIVE}),
}


@dataclass(frozen=True, slots=True)
class TargetRoute:
    prefix: str
    zone: TargetZone
    privacy_classes: frozenset[PrivacyClass]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.zone, TargetZone)
            or not isinstance(self.privacy_classes, frozenset)
            or any(
                not isinstance(privacy, PrivacyClass)
                for privacy in self.privacy_classes
            )
            or not self.prefix.endswith("/")
            or not self.privacy_classes
        ):
            raise WriterRefusal(WriterErrorCode.TARGET_DENIED)
        if PrivacyClass.STUDENT_PRIVATE in self.privacy_classes:
            raise WriterRefusal(WriterErrorCode.STUDENT_PRIVATE_DENIED)
        if self.prefix != _C7_PREFIXES.get(self.zone):
            raise WriterRefusal(WriterErrorCode.TARGET_DENIED)
        if self.privacy_classes != _C7_PRIVACY_CLASSES[self.zone]:
            raise WriterRefusal(WriterErrorCode.TARGET_DENIED)


@dataclass(frozen=True, slots=True)
class CanonicalTarget:
    path: str
    route: TargetRoute


class WriterTargetPolicy:
    """Authorize only exact, server-owned synthetic route prefixes."""

    def __init__(self, routes: tuple[TargetRoute, ...]) -> None:
        if not routes:
            raise WriterRefusal(WriterErrorCode.TARGET_DENIED)
        prefixes = [route.prefix for route in routes]
        if len(prefixes) != len(set(prefixes)):
            raise WriterRefusal(WriterErrorCode.TARGET_DENIED)
        self._routes = routes

    @classmethod
    def c7_synthetic(cls) -> WriterTargetPolicy:
        """The two routes frozen by issue #10; still passed explicitly."""
        return cls(
            (
                TargetRoute(
                    prefix=_C7_PREFIXES[TargetZone.CORE],
                    zone=TargetZone.CORE,
                    privacy_classes=_C7_PRIVACY_CLASSES[TargetZone.CORE],
                ),
                TargetRoute(
                    prefix=_C7_PREFIXES[TargetZone.PRIVATE],
                    zone=TargetZone.PRIVATE,
                    privacy_classes=_C7_PRIVACY_CLASSES[TargetZone.PRIVATE],
                ),
            )
        )

    def canonicalize(self, value: str) -> CanonicalTarget:
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 240
            or "\x00" in value
            or "\\" in value
            or value.startswith("/")
            or not value.endswith(".md")
        ):
            raise WriterRefusal(WriterErrorCode.TARGET_DENIED)
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.as_posix() != value
        ):
            raise WriterRefusal(WriterErrorCode.TARGET_DENIED)
        matching = [route for route in self._routes if value.startswith(route.prefix)]
        if len(matching) != 1 or value == matching[0].prefix:
            raise WriterRefusal(WriterErrorCode.TARGET_DENIED)
        relative_name = value.removeprefix(matching[0].prefix)
        if _SAFE_LEAF.fullmatch(relative_name) is None:
            # C7 keeps the leaf shape narrow; nested arbitrary routes are C8+.
            raise WriterRefusal(WriterErrorCode.TARGET_DENIED)
        return CanonicalTarget(path=value, route=matching[0])

    def authorize(
        self,
        target: CanonicalTarget,
        privacy: PrivacyClass,
    ) -> None:
        if privacy is PrivacyClass.STUDENT_PRIVATE:
            raise WriterRefusal(WriterErrorCode.STUDENT_PRIVATE_DENIED)
        if privacy not in target.route.privacy_classes:
            raise WriterRefusal(WriterErrorCode.CROSS_ZONE_DENIED)


__all__ = [
    "CanonicalTarget",
    "TargetRoute",
    "TargetZone",
    "WriterTargetPolicy",
]
