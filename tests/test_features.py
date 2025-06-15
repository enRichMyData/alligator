from collections import defaultdict
from typing import Dict, List
from unittest.mock import Mock, patch

import pytest

from alligator.feature import Feature
from alligator.types import Candidate, ObjectsData


class TestFeature:
    @pytest.fixture()
    def feature(self) -> Feature:
        """Setup method to patch MongoWrapper and initialize Feature."""
        with patch("alligator.fetchers.MongoWrapper"):
            feature = Feature(dataset_name="test_dataset", table_name="table_name")
            feature.logger = Mock()

        return feature

    @pytest.fixture()
    def candidates(self):
        c1 = {
            "id": "Q163872",
            "name": "The Dark Knight",
            "description": "2008 film directed by Christopher Nolan",
            "features": {
                "ambiguity_mention": 0.241,
                "corrects_tokens": 1.0,
                "ntoken_mention": 3,
                "ntoken_entity": 3,
                "length_mention": 15,
                "length_entity": 15,
                "popularity": 0.12,
                "pos_score": 0.06,
                "es_score": 1.0,
                "ed_score": 1.0,
                "jaccard_score": 1.0,
                "jaccardNgram_score": 1.0,
            },
            "kind": "entity",
            "NERtype": "OTHERS",
            "types": [{"id": "Q11424", "name": "film"}],
        }
        c2 = {
            "id": "Q2695156",
            "name": "the Dark Knight",
            "description": "fictional character in DC Comics",
            "features": {
                "ambiguity_mention": 0.241,
                "corrects_tokens": 1.0,
                "ntoken_mention": 3,
                "ntoken_entity": 3,
                "length_mention": 15,
                "length_entity": 15,
                "popularity": 0.13,
                "pos_score": 0.04,
                "es_score": 1.0,
                "ed_score": 1.0,
                "jaccard_score": 1.0,
                "jaccardNgram_score": 1.0,
            },
            "kind": "entity",
            "NERtype": "OTHERS",
            "score": 0.9573338031768799,
            "types": [
                {"id": "Q43845", "name": "businessperson"},
                {"id": "Q842782", "name": "detective"},
                {"id": "Q3190387", "name": "vigilante"},
                {"id": "Q484876", "name": "chief executive officer"},
                {"id": "Q188784", "name": "superhero"},
                {"id": "Q12362622", "name": "philanthropist"},
                {"id": "Q11124885", "name": "martial artist"},
                {"id": "Q1114461", "name": "comics character"},
                {"id": "Q15632617", "name": "fictional human"},
                {"id": "Q15773347", "name": "film character"},
                {"id": "Q15773317", "name": "television character"},
                {"id": "Q15711870", "name": "animated character"},
                {"id": "Q3656924", "name": "fictional detective"},
                {"id": "Q80447738", "name": "anime character"},
                {"id": "Q20085850", "name": "fictional vigilante"},
                {"id": "Q76450109", "name": "mutant"},
                {"id": "Q117089018", "name": "Faunus"},
            ],
        }
        candidates = [Candidate.from_dict(c1), Candidate.from_dict(c2)]
        return candidates

    @pytest.fixture()
    def mock_db_collection(self):
        """Mock database collection for global frequencies tests."""
        collection = Mock()

        # Mock count_documents to return various document counts
        collection.count_documents = Mock(return_value=10)

        # Mock aggregation pipeline results
        sample_docs = [
            {
                "classified_columns": {"NE": {"0": "OTHER", "1": "OTHER"}},
                "candidates_by_column": {
                    "0": [
                        {
                            "types": [
                                {"id": "Q5", "name": "human"},
                                {"id": "Q215627", "name": "person"},
                            ],
                            "predicates": {
                                "1": {"P31": 0.8, "P106": 0.6}  # instance of  # occupation
                            },
                        },
                        {
                            "types": [{"id": "Q5", "name": "human"}],
                            "predicates": {
                                "1": {"P31": 0.9, "P27": 0.7}  # country of citizenship
                            },
                        },
                    ],
                    "1": [
                        {
                            "types": [
                                {"id": "Q515", "name": "city"},
                                {"id": "Q486972", "name": "human settlement"},
                            ],
                            "predicates": {"0": {"P17": 0.9, "P31": 0.8}},  # country
                        }
                    ],
                },
            },
            {
                "classified_columns": {"NE": {"0": "OTHER", "2": "OTHER"}},
                "candidates_by_column": {
                    "0": [
                        {
                            "types": [{"id": "Q5", "name": "human"}],
                            "predicates": {"2": {"P31": 0.7, "P569": 0.5}},  # date of birth
                        }
                    ],
                    "2": [
                        {
                            "types": [
                                {"id": "Q216353", "name": "title"},
                                {"id": "Q1914636", "name": "academic degree"},
                            ],
                            "predicates": {"0": {"P31": 0.6, "P828": 0.4}},  # has cause
                        }
                    ],
                },
            },
        ]

        collection.aggregate = Mock(return_value=iter(sample_docs))
        return collection

    @pytest.fixture()
    def empty_db_collection(self):
        """Mock empty database collection."""
        collection = Mock()
        collection.count_documents = Mock(return_value=0)
        collection.aggregate = Mock(return_value=iter([]))
        return collection

    @pytest.fixture()
    def no_candidates_db_collection(self):
        """Mock database collection with documents but no candidates."""
        collection = Mock()
        collection.count_documents = Mock(return_value=5)

        sample_docs = [
            {"classified_columns": {"NE": {"0": "OTHER"}}, "candidates_by_column": {}},
            {"classified_columns": {"NE": {"1": "OTHER"}}, "candidates_by_column": {"1": []}},
        ]

        collection.aggregate = Mock(return_value=iter(sample_docs))
        return collection

    def test_feature_initialization(self, feature: Feature):
        assert feature.dataset_name == "test_dataset"
        assert feature.table_name == "table_name"
        assert len(feature.selected_features) > 0
        assert feature.logger is not None

    def test_process_candidates(self, feature: Feature, candidates: List[Candidate]):
        """Test processing candidates with mocked data."""
        feature.process_candidates(candidates, row="Batman Begins 2005")

        for feature_name in feature.selected_features:
            assert feature_name in candidates[0].features
            assert feature_name in candidates[1].features

    def test_compute_global_frequencies_basic(self, feature: Feature, mock_db_collection):
        """Test basic functionality of compute_global_frequencies."""
        with patch.object(feature, "get_db") as mock_get_db:
            mock_db = Mock()
            mock_db.__getitem__ = Mock(return_value=mock_db_collection)
            mock_get_db.return_value = mock_db

            type_freq, pred_freq, pred_pair_freq = feature.compute_global_frequencies()

            # Verify MongoDB query was called correctly
            expected_query = {
                "dataset_name": "test_dataset",
                "table_name": "table_name",
                "status": "DONE",
                "rank_status": "DONE",
            }
            mock_db_collection.count_documents.assert_called_once_with(expected_query)

            # Verify aggregation pipeline was called
            mock_db_collection.aggregate.assert_called_once()

            # Check that frequencies were computed
            assert isinstance(type_freq, defaultdict)
            assert isinstance(pred_freq, defaultdict)
            assert isinstance(pred_pair_freq, dict)

            # Verify type frequencies are normalized (divided by number of docs)
            assert "0" in type_freq
            assert "Q5" in type_freq["0"]  # human type should be present
            assert type_freq["0"]["Q5"] == 1.0  # Should appear in both docs, so 2/2 = 1.0

            # Verify predicate frequencies
            assert "0" in pred_freq
            assert "P31" in pred_freq["0"]  # instance of predicate

            # Verify predicate pair frequencies
            assert "0" in pred_pair_freq
            assert "1" in pred_pair_freq["0"]
            assert "P31" in pred_pair_freq["0"]["1"]

    def test_compute_global_frequencies_with_sampling(self, feature: Feature, mock_db_collection):
        """Test compute_global_frequencies with document sampling."""
        with patch.object(feature, "get_db") as mock_get_db:
            mock_db = Mock()
            mock_db.__getitem__ = Mock(return_value=mock_db_collection)
            mock_get_db.return_value = mock_db

            # Test with random sampling
            type_freq, pred_freq, pred_pair_freq = feature.compute_global_frequencies(
                docs_to_process=0.5, random_sample=True
            )

            # Verify aggregation was called
            mock_db_collection.aggregate.assert_called_once()

            # Check the pipeline includes sampling
            call_args = mock_db_collection.aggregate.call_args[0][0]
            pipeline_stages = [stage for stage in call_args]

            # Should have $match, $sample, and $lookup stages
            assert any("$match" in stage for stage in pipeline_stages)
            assert any("$sample" in stage for stage in pipeline_stages)
            assert any("$lookup" in stage for stage in pipeline_stages)

    def test_compute_global_frequencies_no_sampling(self, feature: Feature, mock_db_collection):
        """Test compute_global_frequencies without sampling (using limit)."""
        with patch.object(feature, "get_db") as mock_get_db:
            mock_db = Mock()
            mock_db.__getitem__ = Mock(return_value=mock_db_collection)
            mock_get_db.return_value = mock_db

            # Test without random sampling
            type_freq, pred_freq, pred_pair_freq = feature.compute_global_frequencies(
                docs_to_process=0.3, random_sample=False
            )

            # Check the pipeline includes limit instead of sample
            call_args = mock_db_collection.aggregate.call_args[0][0]
            pipeline_stages = [stage for stage in call_args]

            # Should have $match, $limit, and $lookup stages
            assert any("$match" in stage for stage in pipeline_stages)
            assert any("$limit" in stage for stage in pipeline_stages)
            assert not any("$sample" in stage for stage in pipeline_stages)

    def test_compute_global_frequencies_empty_database(
        self, feature: Feature, empty_db_collection
    ):
        """Test compute_global_frequencies with empty database."""
        with patch.object(feature, "get_db") as mock_get_db:
            mock_db = Mock()
            mock_db.__getitem__ = Mock(return_value=empty_db_collection)
            mock_get_db.return_value = mock_db

            type_freq, pred_freq, pred_pair_freq = feature.compute_global_frequencies()

            # Should return empty defaultdicts when no documents
            assert len(type_freq) == 0
            assert len(pred_freq) == 0
            assert len(pred_pair_freq) == 0

            # Should log warning
            feature.logger.warning.assert_called_with(
                "No documents match the criteria for computing global frequencies."
            )

    def test_compute_global_frequencies_no_candidates(
        self, feature: Feature, no_candidates_db_collection
    ):
        """Test compute_global_frequencies when documents exist but have no candidates."""
        with patch.object(feature, "get_db") as mock_get_db:
            mock_db = Mock()
            mock_db.__getitem__ = Mock(return_value=no_candidates_db_collection)
            mock_get_db.return_value = mock_db

            type_freq, pred_freq, pred_pair_freq = feature.compute_global_frequencies()

            # Should return empty frequencies
            assert len(type_freq) == 0
            assert len(pred_freq) == 0
            assert len(pred_pair_freq) == 0

    def test_compute_global_frequencies_top_n_limit(self, feature: Feature):
        """Test that compute_global_frequencies respects the top_n_cta_cpa_freq limit."""
        # Create a feature with top_n_cta_cpa_freq = 1
        feature.top_n_cta_cpa_freq = 1

        # Mock collection with multiple candidates per column
        collection = Mock()
        collection.count_documents = Mock(return_value=1)

        sample_docs = [
            {
                "classified_columns": {"NE": {"0": "OTHER"}},
                "candidates_by_column": {
                    "0": [
                        {
                            "types": [{"id": "Q5", "name": "human"}],
                            "predicates": {"1": {"P31": 0.8}},
                        },
                        {
                            "types": [{"id": "Q215627", "name": "person"}],
                            "predicates": {"1": {"P106": 0.6}},
                        },
                        {
                            "types": [{"id": "Q488383", "name": "object"}],
                            "predicates": {"1": {"P27": 0.7}},
                        },
                    ]
                },
            }
        ]

        collection.aggregate = Mock(return_value=iter(sample_docs))

        with patch.object(feature, "get_db") as mock_get_db:
            mock_db = Mock()
            mock_db.__getitem__ = Mock(return_value=collection)
            mock_get_db.return_value = mock_db

            type_freq, pred_freq, pred_pair_freq = feature.compute_global_frequencies()

            # Should only process the first candidate (top_n_cta_cpa_freq = 1)
            assert "Q5" in type_freq["0"]  # First candidate's type
            assert "Q215627" not in type_freq["0"]  # Second candidate's type should be ignored
            assert "Q488383" not in type_freq["0"]  # Third candidate's type should be ignored

    def test_compute_global_frequencies_deduplication(self, feature: Feature):
        """Test that compute_global_frequencies properly deduplicates
        types and predicates within a document."""
        collection = Mock()
        collection.count_documents = Mock(return_value=1)

        # Create candidates with duplicate types and predicates
        sample_docs = [
            {
                "classified_columns": {"NE": {"0": "OTHER"}},
                "candidates_by_column": {
                    "0": [
                        {
                            "types": [{"id": "Q5", "name": "human"}],
                            "predicates": {"1": {"P31": 0.8}},
                        },
                        {
                            "types": [{"id": "Q5", "name": "human"}],  # Duplicate type
                            "predicates": {"1": {"P31": 0.6}},  # Duplicate predicate
                        },
                    ]
                },
            }
        ]

        collection.aggregate = Mock(return_value=iter(sample_docs))

        with patch.object(feature, "get_db") as mock_get_db:
            mock_db = Mock()
            mock_db.__getitem__ = Mock(return_value=collection)
            mock_get_db.return_value = mock_db

            type_freq, pred_freq, pred_pair_freq = feature.compute_global_frequencies()

            # Type should only be counted once per document
            # despite appearing in multiple candidates
            assert type_freq["0"]["Q5"] == 1.0  # Should be 1/1 = 1.0, not 2/1 = 2.0

            # Predicate should only be counted once per document
            assert pred_freq["0"]["P31"] == 0.8  # Should be 0.8/1 = 0.8, not (0.8+0.6)/1 = 1.4

    def test_compute_global_frequencies_full_processing(
        self, feature: Feature, mock_db_collection
    ):
        """Test compute_global_frequencies with docs_to_process >= 1.0."""
        with patch.object(feature, "get_db") as mock_get_db:
            mock_db = Mock()
            mock_db.__getitem__ = Mock(return_value=mock_db_collection)
            mock_get_db.return_value = mock_db

            # Test with 1.5 (should process all since >= 1.0)
            type_freq, pred_freq, pred_pair_freq = feature.compute_global_frequencies(
                docs_to_process=1.5
            )

            # Verify aggregation was called
            mock_db_collection.aggregate.assert_called_once()

            # Test with 1.0 (should process all)
            mock_db_collection.reset_mock()
            type_freq, pred_freq, pred_pair_freq = feature.compute_global_frequencies(
                docs_to_process=1.0
            )

            # Verify aggregation was called again
            mock_db_collection.aggregate.assert_called_once()

    def test_compute_global_frequencies_warning_on_no_results(self, feature: Feature):
        """Test warning when aggregation pipeline returns no results despite matching documents."""
        collection = Mock()
        collection.count_documents = Mock(return_value=5)  # Documents exist
        collection.aggregate = Mock(return_value=iter([]))  # But aggregation returns nothing

        with patch.object(feature, "get_db") as mock_get_db:
            mock_db = Mock()
            mock_db.__getitem__ = Mock(return_value=collection)
            mock_get_db.return_value = mock_db

            type_freq, pred_freq, pred_pair_freq = feature.compute_global_frequencies()

            # Should log warning about pipeline returning no documents
            warning_calls = [
                call
                for call in feature.logger.warning.call_args_list
                if "aggregation pipeline" in str(call)
            ]
            assert len(warning_calls) > 0

    def test_compute_entity_entity_relationships_empty_input(self, feature: Feature):
        """Test compute_entity_entity_relationships with empty input."""
        # Test with empty candidates
        feature.compute_entity_entity_relationships({}, {})

        # Test with single column (should return early)
        single_col_candidates = {
            "0": [
                Candidate.from_dict(
                    {
                        "id": "Q1",
                        "name": "Entity1",
                        "features": {"p_subj_ne": 0.0, "p_obj_ne": 0.0},
                    }
                )
            ]
        }
        feature.compute_entity_entity_relationships(single_col_candidates, {})

        # Features should remain unchanged since method returns early
        assert single_col_candidates["0"][0].features["p_subj_ne"] == 0.0

    def test_compute_entity_entity_relationships_no_objects_data(self, feature: Feature):
        """Test compute_entity_entity_relationships when candidates have no objects data."""
        candidates_by_col = {
            "0": [
                Candidate.from_dict(
                    {
                        "id": "Q1",
                        "name": "Entity1",
                        "features": {
                            "p_obj_ne": 0.0,
                            "p_subj_ne": 0.0,
                            "ed_score": 0.8,
                            "jaccard_score": 0.9,
                        },
                    }
                )
            ],
            "1": [
                Candidate.from_dict(
                    {
                        "id": "Q2",
                        "name": "Entity2",
                        "features": {
                            "p_obj_ne": 0.0,
                            "p_subj_ne": 0.0,
                            "ed_score": 0.7,
                            "jaccard_score": 0.85,
                        },
                    }
                )
            ],
        }

        # Empty objects data
        objects_data: Dict[str, ObjectsData] = {}

        feature.compute_entity_entity_relationships(candidates_by_col, objects_data)

        # Features should remain at 0 since no objects data
        assert candidates_by_col["0"][0].features["p_subj_ne"] == 0.0
        assert candidates_by_col["0"][0].features["p_obj_ne"] == 0.0
        assert candidates_by_col["1"][0].features["p_subj_ne"] == 0.0
        assert candidates_by_col["1"][0].features["p_obj_ne"] == 0.0

    def test_compute_entity_entity_relationships_basic_functionality(self, feature: Feature):
        """Test basic functionality of compute_entity_entity_relationships."""
        candidates_by_col = {
            "0": [
                Candidate.from_dict(
                    {
                        "id": "Q1",
                        "name": "Brad Pitt",
                        "features": {
                            "p_obj_ne": 0.0,
                            "p_subj_ne": 0.0,
                            "ed_score": 0.8,
                            "jaccard_score": 0.9,
                            "jaccardNgram_score": 0.85,
                        },
                    }
                )
            ],
            "1": [
                Candidate.from_dict(
                    {
                        "id": "Q2",
                        "name": "Fight Club",
                        "features": {
                            "p_obj_ne": 0.0,
                            "p_subj_ne": 0.0,
                            "ed_score": 0.7,
                            "jaccard_score": 0.85,
                            "jaccardNgram_score": 0.8,
                        },
                    }
                )
            ],
        }

        # Objects data showing that Q1 (Brad Pitt) is connected to Q2 (Fight Club)
        objects_data: Dict[str, ObjectsData] = {
            "Q1": {"objects": {"Q2": ["P161"]}}  # P161 = "cast member"
        }

        feature.compute_entity_entity_relationships(candidates_by_col, objects_data)

        # Subject candidate should have updated p_subj_ne feature
        subj_candidate = candidates_by_col["0"][0]
        assert subj_candidate.features["p_subj_ne"] > 0.0

        # Object candidate should have updated p_obj_ne feature
        obj_candidate = candidates_by_col["1"][0]
        assert obj_candidate.features["p_obj_ne"] > 0.0

        # Check that matches and predicates were recorded
        assert "1" in subj_candidate.matches
        assert len(subj_candidate.matches["1"]) > 0
        assert subj_candidate.matches["1"][0]["p"] == "P161"
        assert subj_candidate.matches["1"][0]["o"] == "Q2"

        assert "1" in subj_candidate.predicates
        assert "P161" in subj_candidate.predicates["1"]

    def test_compute_entity_entity_relationships_multiple_columns(self, feature: Feature):
        """Test compute_entity_entity_relationships with multiple columns."""
        candidates_by_col = {
            "0": [
                Candidate.from_dict(
                    {
                        "id": "Q1",
                        "name": "Brad Pitt",
                        "features": {
                            "p_obj_ne": 0.0,
                            "p_subj_ne": 0.0,
                            "ed_score": 0.8,
                            "jaccard_score": 0.9,
                            "jaccardNgram_score": 0.85,
                        },
                    }
                )
            ],
            "1": [
                Candidate.from_dict(
                    {
                        "id": "Q2",
                        "name": "Fight Club",
                        "features": {
                            "p_obj_ne": 0.0,
                            "p_subj_ne": 0.0,
                            "ed_score": 0.7,
                            "jaccard_score": 0.85,
                            "jaccardNgram_score": 0.8,
                        },
                    }
                )
            ],
            "2": [
                Candidate.from_dict(
                    {
                        "id": "Q3",
                        "name": "Helena Bonham Carter",
                        "features": {
                            "p_obj_ne": 0.0,
                            "p_subj_ne": 0.0,
                            "ed_score": 0.75,
                            "jaccard_score": 0.8,
                            "jaccardNgram_score": 0.82,
                        },
                    }
                )
            ],
        }

        # Objects data showing relationships between entities
        objects_data: Dict[str, ObjectsData] = {
            "Q1": {  # Brad Pitt
                "objects": {
                    "Q2": ["P161"],  # cast member of Fight Club
                    "Q3": ["P26"],  # spouse of Helena Bonham Carter (example)
                }
            },
            "Q2": {  # Fight Club
                "objects": {
                    "Q1": ["P161"],  # has cast member Brad Pitt
                    "Q3": ["P161"],  # has cast member Helena Bonham Carter
                }
            },
        }

        feature.compute_entity_entity_relationships(candidates_by_col, objects_data)

        # With 3 columns, normalization factor is 1/3
        subj_candidate = candidates_by_col["0"][0]

        # Should have relationships with both other columns
        assert "1" in subj_candidate.matches
        assert "2" in subj_candidate.matches
        assert len(subj_candidate.matches["1"]) > 0
        assert len(subj_candidate.matches["2"]) > 0

        # p_subj_ne should be sum of normalized scores from both relationships
        assert subj_candidate.features["p_subj_ne"] > 0.0

        # Object candidates should also have updated features
        assert candidates_by_col["1"][0].features["p_obj_ne"] > 0.0
        assert candidates_by_col["2"][0].features["p_obj_ne"] > 0.0

    def test_compute_entity_entity_relationships_multiple_predicates(self, feature: Feature):
        """Test compute_entity_entity_relationships with multiple predicates."""
        candidates_by_col = {
            "0": [
                Candidate.from_dict(
                    {
                        "id": "Q1",
                        "name": "Brad Pitt",
                        "features": {
                            "p_subj_ne": 0.0,
                            "ed_score": 0.8,
                            "jaccard_score": 0.9,
                            "jaccardNgram_score": 0.85,
                        },
                    }
                )
            ],
            "1": [
                Candidate.from_dict(
                    {
                        "id": "Q2",
                        "name": "Fight Club",
                        "features": {
                            "p_obj_ne": 0.0,
                            "ed_score": 0.7,
                            "jaccard_score": 0.85,
                            "jaccardNgram_score": 0.8,
                        },
                    }
                )
            ],
        }

        # Multiple predicates connecting the same entities
        objects_data: Dict[str, ObjectsData] = {
            "Q1": {"objects": {"Q2": ["P161", "P57", "P170"]}}  # cast member, director, creator
        }

        feature.compute_entity_entity_relationships(candidates_by_col, objects_data)

        subj_candidate = candidates_by_col["0"][0]

        # Should have multiple matches recorded
        assert len(subj_candidate.matches["1"]) == 3

        # Should have multiple predicates recorded
        assert len(subj_candidate.predicates["1"]) == 3
        assert "P161" in subj_candidate.predicates["1"]
        assert "P57" in subj_candidate.predicates["1"]
        assert "P170" in subj_candidate.predicates["1"]

    def test_compute_entity_entity_relationships_no_intersection(self, feature: Feature):
        """Test compute_entity_entity_relationships when there's no object intersection."""
        candidates_by_col = {
            "0": [
                Candidate.from_dict(
                    {
                        "id": "Q1",
                        "name": "Entity1",
                        "features": {
                            "p_obj_ne": 0.0,
                            "p_subj_ne": 0.0,
                            "ed_score": 0.8,
                            "jaccard_score": 0.9,
                            "jaccardNgram_score": 0.85,
                        },
                    }
                )
            ],
            "1": [
                Candidate.from_dict(
                    {
                        "id": "Q2",
                        "name": "Entity2",
                        "features": {
                            "p_obj_ne": 0.0,
                            "p_subj_ne": 0.0,
                            "ed_score": 0.7,
                            "jaccard_score": 0.85,
                            "jaccardNgram_score": 0.8,
                        },
                    }
                )
            ],
        }

        # Q1 has objects, but not Q2
        objects_data: Dict[str, ObjectsData] = {
            "Q1": {"objects": {"Q99": ["P31"]}}  # Connected to different entity
        }

        feature.compute_entity_entity_relationships(candidates_by_col, objects_data)

        # No intersection, so features should remain 0
        assert candidates_by_col["0"][0].features["p_subj_ne"] == 0.0
        assert candidates_by_col["1"][0].features["p_obj_ne"] == 0.0

    def test_compute_entity_entity_relationships_multiple_candidates_per_column(
        self, feature: Feature
    ):
        """Test compute_entity_entity_relationships with multiple candidates per column."""
        candidates_by_col = {
            "0": [
                Candidate.from_dict(
                    {
                        "id": "Q1",
                        "name": "Brad Pitt",
                        "features": {
                            "p_obj_ne": 0.0,
                            "p_subj_ne": 0.0,
                            "ed_score": 0.8,
                            "jaccard_score": 0.9,
                            "jaccardNgram_score": 0.85,
                        },
                    }
                ),
                Candidate.from_dict(
                    {
                        "id": "Q10",
                        "name": "William Bradley Pitt",
                        "features": {
                            "p_obj_ne": 0.0,
                            "p_subj_ne": 0.0,
                            "ed_score": 0.75,
                            "jaccard_score": 0.85,
                            "jaccardNgram_score": 0.8,
                        },
                    }
                ),
            ],
            "1": [
                Candidate.from_dict(
                    {
                        "id": "Q2",
                        "name": "Fight Club",
                        "features": {
                            "p_obj_ne": 0.0,
                            "p_subj_ne": 0.0,
                            "ed_score": 0.7,
                            "jaccard_score": 0.85,
                            "jaccardNgram_score": 0.8,
                        },
                    }
                ),
                Candidate.from_dict(
                    {
                        "id": "Q20",
                        "name": "Fight Club (film)",
                        "features": {
                            "p_obj_ne": 0.0,
                            "p_subj_ne": 0.0,
                            "ed_score": 0.65,
                            "jaccard_score": 0.8,
                            "jaccardNgram_score": 0.75,
                        },
                    }
                ),
            ],
        }

        # Both Q1 and Q10 connected to Q2
        objects_data: Dict[str, ObjectsData] = {
            "Q1": {"objects": {"Q2": ["P161"]}},
            "Q10": {"objects": {"Q2": ["P161"]}},
        }

        feature.compute_entity_entity_relationships(candidates_by_col, objects_data)

        # Both subject candidates should have updated features
        assert candidates_by_col["0"][0].features["p_subj_ne"] > 0.0
        assert candidates_by_col["0"][1].features["p_subj_ne"] > 0.0

        # First object candidate should have updated feature (it's in the objects)
        assert candidates_by_col["1"][0].features["p_obj_ne"] > 0.0

        # Second object candidate should remain 0 (not in the objects data)
        assert candidates_by_col["1"][1].features["p_obj_ne"] == 0.0

    def test_compute_entity_entity_relationships_score_maximization(self, feature: Feature):
        """Test that compute_entity_entity_relationships uses maximum scores correctly."""
        candidates_by_col = {
            "0": [
                Candidate.from_dict(
                    {
                        "id": "Q1",
                        "name": "Entity1",
                        "features": {
                            "p_obj_ne": 0.0,
                            "p_subj_ne": 0.0,
                            "ed_score": 0.5,
                            "jaccard_score": 0.6,
                            "jaccardNgram_score": 0.7,
                        },
                    }
                ),
                Candidate.from_dict(
                    {
                        "id": "Q1",
                        "name": "Entity1",
                        "features": {
                            "p_obj_ne": 0.0,
                            "p_subj_ne": 0.0,
                            "ed_score": 0.8,
                            "jaccard_score": 0.9,
                            "jaccardNgram_score": 0.95,
                        },
                    }
                ),
            ],
            "1": [
                Candidate.from_dict(
                    {
                        "id": "Q2",
                        "name": "Entity2",
                        "features": {
                            "p_obj_ne": 0.0,
                            "p_subj_ne": 0.0,
                            "ed_score": 0.7,
                            "jaccard_score": 0.8,
                            "jaccardNgram_score": 0.75,
                        },
                    }
                )
            ],
        }

        objects_data: Dict[str, ObjectsData] = {"Q1": {"objects": {"Q2": ["P31"]}}}

        feature.compute_entity_entity_relationships(candidates_by_col, objects_data)

        expected_p_obj_ne = (0.8 + 0.9 + 0.95) / 3 / 2
        expected_p_subj_ne = (0.7 + 0.8 + 0.75) / 3 / 2

        assert candidates_by_col["0"][0].features["p_obj_ne"] == 0.0
        assert candidates_by_col["0"][1].features["p_obj_ne"] == 0.0
        assert candidates_by_col["0"][0].features["p_subj_ne"] == expected_p_subj_ne
        assert candidates_by_col["0"][1].features["p_subj_ne"] == expected_p_subj_ne

        assert candidates_by_col["1"][0].features["p_obj_ne"] == expected_p_obj_ne
        assert candidates_by_col["1"][0].features["p_subj_ne"] == 0.0
