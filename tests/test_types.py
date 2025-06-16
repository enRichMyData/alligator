from alligator.types import Candidate, Entity, RowData


class TestEntity:
    def test_entity_creation(self):
        """Test Entity creation with basic parameters."""
        entity = Entity(value="Brad Pitt", row_index=0, col_index="0")
        assert entity.value == "Brad Pitt"
        assert entity.row_index == 0
        assert entity.col_index == "0"
        assert entity.correct_qids is None
        assert entity.fuzzy is False

    def test_entity_with_qids(self):
        """Test Entity creation with correct QIDs."""
        entity = Entity(
            value="Brad Pitt", row_index=0, col_index="0", correct_qids=["Q35332"], fuzzy=True
        )
        assert entity.correct_qids == ["Q35332"]
        assert entity.fuzzy is True

    def test_entity_with_multiple_qids(self):
        """Test Entity with multiple correct QIDs."""
        entity = Entity(
            value="John Smith", row_index=1, col_index="1", correct_qids=["Q12345", "Q67890"]
        )
        assert entity.correct_qids is not None
        assert len(entity.correct_qids) == 2
        assert "Q12345" in entity.correct_qids
        assert "Q67890" in entity.correct_qids


class TestRowData:
    def test_rowdata_creation(self):
        """Test RowData creation."""
        row_data = RowData(
            doc_id="test_doc_1",
            row=["Brad Pitt", "Fight Club", "1999"],
            ne_columns={"0": "PERSON", "1": "OTHER"},
            lit_columns={"2": "NUMBER"},
            context_columns=["0", "1", "2"],
            correct_qids={"0-0": ["Q35332"]},
            row_index=0,
        )

        assert row_data.doc_id == "test_doc_1"
        assert len(row_data.row) == 3
        assert row_data.ne_columns["0"] == "PERSON"
        assert row_data.lit_columns["2"] == "NUMBER"
        assert row_data.correct_qids["0-0"] == ["Q35332"]
        assert row_data.row_index == 0

    def test_rowdata_empty_collections(self):
        """Test RowData with empty collections."""
        row_data = RowData(
            doc_id="test_doc_2",
            row=["data"],
            ne_columns={},
            lit_columns={},
            context_columns=[],
            correct_qids={},
            row_index=None,
        )

        assert len(row_data.ne_columns) == 0
        assert len(row_data.lit_columns) == 0
        assert len(row_data.context_columns) == 0
        assert len(row_data.correct_qids) == 0
        assert row_data.row_index is None


class TestCandidate:
    def test_candidate_creation(self):
        """Test Candidate creation with basic parameters."""
        candidate = Candidate(id="Q35332", name="Brad Pitt")
        assert candidate.id == "Q35332"
        assert candidate.name == "Brad Pitt"
        assert candidate.score == 0.0
        assert candidate.kind == ""
        assert candidate.NERtype == ""
        assert candidate.description == ""
        assert isinstance(candidate.features, dict)
        assert len(candidate.features) == 0

    def test_candidate_with_all_fields(self):
        """Test Candidate creation with all fields."""
        features = {"similarity": 0.95, "popularity": 0.8}
        types = [{"id": "Q5", "name": "human"}]

        candidate = Candidate(
            id="Q35332",
            name="Brad Pitt",
            score=0.9,
            kind="entity",
            NERtype="PERSON",
            description="American actor",
            features=features,
            types=types,
        )

        assert candidate.score == 0.9
        assert candidate.kind == "entity"
        assert candidate.NERtype == "PERSON"
        assert candidate.description == "American actor"
        assert candidate.features["similarity"] == 0.95
        assert len(candidate.types) == 1
        assert candidate.types[0]["id"] == "Q5"

    def test_candidate_to_dict(self):
        """Test Candidate to_dict method."""
        candidate = Candidate(
            id="Q35332", name="Brad Pitt", score=0.9, description="American actor"
        )

        candidate_dict = candidate.to_dict()

        assert isinstance(candidate_dict, dict)
        assert candidate_dict["id"] == "Q35332"
        assert candidate_dict["name"] == "Brad Pitt"
        assert candidate_dict["score"] == 0.9
        assert candidate_dict["description"] == "American actor"
        assert "features" in candidate_dict
        assert "types" in candidate_dict
        assert "matches" in candidate_dict
        assert "predicates" in candidate_dict

    def test_candidate_from_dict(self):
        """Test Candidate from_dict class method."""
        data = {
            "id": "Q35332",
            "name": "Brad Pitt",
            "score": 0.9,
            "description": "American actor",
            "kind": "entity",
            "NERtype": "PERSON",
            "features": {"similarity": 0.95},
            "types": [{"id": "Q5", "name": "human"}],
            "matches": {"exact": [{"match": "Brad Pitt"}]},
            "predicates": {"P106": {"actor": 0.9}},
        }

        candidate = Candidate.from_dict(data)

        assert candidate.id == "Q35332"
        assert candidate.name == "Brad Pitt"
        assert candidate.score == 0.9
        assert candidate.description == "American actor"
        assert candidate.kind == "entity"
        assert candidate.NERtype == "PERSON"
        assert candidate.features["similarity"] == 0.95
        assert len(candidate.types) == 1
        assert candidate.types[0]["id"] == "Q5"
        assert "exact" in candidate.matches
        assert "P106" in candidate.predicates

    def test_candidate_from_dict_minimal(self):
        """Test Candidate from_dict with minimal data."""
        data = {"id": "Q123", "name": "Test Entity"}

        candidate = Candidate.from_dict(data)

        assert candidate.id == "Q123"
        assert candidate.name == "Test Entity"
        assert candidate.score == 0.0
        assert candidate.description == ""
        assert candidate.kind == ""
        assert candidate.NERtype == ""
        assert len(candidate.features) == 0
        assert len(candidate.types) == 0

    def test_candidate_from_dict_empty_dict(self):
        """Test Candidate from_dict with empty dictionary."""
        candidate = Candidate.from_dict({})

        assert candidate.id == ""
        assert candidate.name == ""
        assert candidate.score == 0.0
        assert candidate.description == ""

    def test_candidate_roundtrip_conversion(self):
        """Test converting Candidate to dict and back."""
        original = Candidate(
            id="Q35332",
            name="Brad Pitt",
            score=0.9,
            description="American actor",
            features={"similarity": 0.95},
        )

        # Convert to dict and back
        candidate_dict = original.to_dict()
        reconstructed = Candidate.from_dict(candidate_dict)

        assert reconstructed.id == original.id
        assert reconstructed.name == original.name
        assert reconstructed.score == original.score
        assert reconstructed.description == original.description
        assert reconstructed.features == original.features
