"""Format-conversion adapters that normalize foreign input into Cyclone Profile.

Contract §2.6: OpenWiki is a knowledge-organizing *engine*, not a second
knowledge base. Its output only becomes a Cyclone Profile note after it has
passed through an adapter here -- nothing upstream of this package is treated
as already-Profile-shaped.
"""

from ckp.adapters.openwiki_v01 import (
    FieldDrop,
    FieldMapping,
    GapDeclaration,
    OpenWikiConversionError,
    convert_note_text,
    convert_openwiki_v01,
)

__all__ = [
    "FieldDrop",
    "FieldMapping",
    "GapDeclaration",
    "OpenWikiConversionError",
    "convert_note_text",
    "convert_openwiki_v01",
]
