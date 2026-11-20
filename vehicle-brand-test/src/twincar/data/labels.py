"""Parse Stanford Cars class names into make / model / year.

Stanford class names follow the loose convention ``<make> <model...> <year>``,
e.g. ``"AM General Hummer SUV 2000"`` or ``"Acura RL Sedan 2012"``. Most makes
are a single token, but a few span multiple tokens (``"AM General"``,
``"Aston Martin"``, ``"Land Rover"``), so we resolve the make with a
longest-prefix match against a known list before falling back to the first
token.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Known makes used for longest-prefix matching. Multi-word makes must win over a
# bare first-token fallback, so we sort by token count (then length) at import
# time rather than relying on declaration order here.
KNOWN_MAKES: tuple[str, ...] = (
    # Multi-word makes (the only ones in Stanford Cars that need prefix logic).
    "AM General",
    "Aston Martin",
    "Land Rover",
    # Single-token / hyphenated makes present in Stanford Cars.
    "Acura",
    "Audi",
    "BMW",
    "Bentley",
    "Bugatti",
    "Buick",
    "Cadillac",
    "Chevrolet",
    "Chrysler",
    "Daewoo",
    "Dodge",
    "Eagle",
    "FIAT",
    "Ferrari",
    "Fisker",
    "Ford",
    "GMC",
    "Geo",
    "HUMMER",
    "Honda",
    "Hyundai",
    "Infiniti",
    "Isuzu",
    "Jaguar",
    "Jeep",
    "Lamborghini",
    "Lincoln",
    "MINI",
    "Maybach",
    "Mazda",
    "McLaren",
    "Mercedes-Benz",
    "Mitsubishi",
    "Nissan",
    "Plymouth",
    "Porsche",
    "Ram",
    "Rolls-Royce",
    "Scion",
    "Spyker",
    "Suzuki",
    "Tesla",
    "Toyota",
    "Volkswagen",
    "Volvo",
    "smart",
)

# Longest prefix first so e.g. "Aston Martin" matches before any single token.
_MAKES_BY_LENGTH: tuple[str, ...] = tuple(
    sorted(KNOWN_MAKES, key=lambda m: (m.count(" "), len(m)), reverse=True)
)

_YEAR_RE = re.compile(r"(\d{4})\s*$")


@dataclass(frozen=True)
class ParsedClassName:
    """Structured view of a Stanford Cars class name."""

    fine_label: str
    make: str
    model: str
    year: int | None


def parse_year(class_name: str) -> int | None:
    """Return the trailing 4-digit year, or ``None`` if absent."""
    match = _YEAR_RE.search(class_name.strip())
    return int(match.group(1)) if match else None


def split_make_model(core: str) -> tuple[str, str]:
    """Split a year-stripped class name into ``(make, model)``.

    Uses a longest-prefix match against :data:`KNOWN_MAKES`; unknown makes fall
    back to using the first whitespace-delimited token as the make.
    """
    core = core.strip()
    if not core:
        return "", ""
    for make in _MAKES_BY_LENGTH:
        if core == make or core.startswith(make + " "):
            model = core[len(make):].strip()
            return make, model
    # Fallback: first token is the make, remainder is the model.
    make, _, model = core.partition(" ")
    return make, model.strip()


def parse_class_name(class_name: str) -> ParsedClassName:
    """Parse a Stanford class name into make / model / year."""
    name = class_name.strip()
    year = parse_year(name)
    core = _YEAR_RE.sub("", name).strip() if year is not None else name
    make, model = split_make_model(core)
    return ParsedClassName(fine_label=name, make=make, model=model, year=year)
