import pytest

from lib.types.other import Direction, PetAction, normalize_direction


class TestNormalizeDirection:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("north", Direction.north),
            ("northeast", Direction.northeast),
            ("east", Direction.east),
            ("southeast", Direction.southeast),
            ("south", Direction.south),
            ("southwest", Direction.southwest),
            ("west", Direction.west),
            ("northwest", Direction.northwest),
        ],
    )
    def test_canonical(self, raw: str, expected: Direction):
        assert normalize_direction(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("NW", Direction.northwest),
            ("north-west", Direction.northwest),
            ("north west", Direction.northwest),
            ("NorthWest", Direction.northwest),
            ("NORTH_WEST", Direction.northwest),
            ("ne", Direction.northeast),
            ("NE", Direction.northeast),
            ("north-east", Direction.northeast),
            ("se", Direction.southeast),
            ("South-East", Direction.southeast),
            ("sw", Direction.southwest),
            ("s_w", Direction.southwest),
            ("n", Direction.north),
            ("S", Direction.south),
            ("e", Direction.east),
            ("E", Direction.east),
            ("w", Direction.west),
            ("W", Direction.west),
            ("  northwest  ", Direction.northwest),
        ],
    )
    def test_alternate_spellings(self, raw: str, expected: Direction):
        assert normalize_direction(raw) == expected

    def test_passthrough_enum(self):
        assert normalize_direction(Direction.north) is Direction.north

    @pytest.mark.parametrize("raw", ["up", "northwestt", "", "diagonal"])
    def test_unknown_raises(self, raw: str):
        with pytest.raises(ValueError, match="unknown direction"):
            normalize_direction(raw)

    def test_non_string_raises(self):
        with pytest.raises(ValueError, match="must be a string"):
            normalize_direction(42)


class TestPetActionDirection:
    def test_canonical_parses(self):
        action = PetAction(thought="t", direction="northwest", distance=5)
        assert action.direction == Direction.northwest

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("NW", "northwest"),
            ("north-west", "northwest"),
            ("north west", "northwest"),
            ("NorthWest", "northwest"),
            ("ne", "northeast"),
            ("s_w", "southwest"),
            ("E", "east"),
            ("W", "west"),
        ],
    )
    def test_alternate_spellings_accepted(self, raw: str, expected: str):
        action = PetAction(thought="t", direction=raw, distance=5)
        # use_enum_values stores the canonical string value
        assert action.direction == expected

    @pytest.mark.parametrize("raw", ["up", "northwestt", ""])
    def test_invalid_rejected(self, raw: str):
        with pytest.raises(ValueError, match="unknown direction"):
            PetAction(thought="t", direction=raw, distance=5)

    def test_schema_only_lists_canonical_values(self):
        schema = PetAction.model_json_schema()
        defs = schema.get("defs") or schema.get("$defs")
        enum = defs["Direction"]["enum"]
        assert enum == [d.value for d in Direction]
        # none of the alternate spellings leak into the schema
        for alias in ("NW", "north-west", "north west", "ne", "n"):
            assert alias not in enum
