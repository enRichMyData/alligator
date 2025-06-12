import multiprocessing as mp
from functools import partial
from typing import Tuple

from alligator.config import AlligatorConfig
from alligator.feature import Feature
from alligator.logging import get_logger
from alligator.ml import MLWorker


class MLManager:
    """Manages machine learning pipeline for ranking and reranking."""

    def __init__(self, config: AlligatorConfig):
        self.config = config
        self.logger = get_logger("ml_manager")

    def run_ml_pipeline(self, feature: Feature) -> None:
        """Run the complete ML pipeline with ranking and reranking stages."""
        num_workers = self.config.worker.num_workers or 1
        pool = mp.Pool(processes=num_workers)

        try:
            # First ML ranking stage
            rank_stage_partial_func = partial(
                self._ml_worker,
                stage="rank",
                global_frequencies=(None, None, None),
            )
            pool.map(rank_stage_partial_func, range(self.config.ml.num_ml_workers))
            self.logger.info("ML Rank stage complete.")

            # Compute global frequencies (this happens in the main process)
            self.logger.info("Computing global frequencies...")
            (
                type_frequencies,
                predicate_frequencies,
                predicate_pair_frequencies,
            ) = feature.compute_global_frequencies(
                docs_to_process=self.config.feature.doc_percentage_type_features,
                random_sample=False,
            )
            self.logger.info("Global frequencies computed.")

            # Second ML ranking stage with global frequencies
            rerank_stage_partial_func = partial(
                self._ml_worker,
                stage="rerank",
                global_frequencies=(
                    type_frequencies,
                    predicate_frequencies,
                    predicate_pair_frequencies,
                ),
            )
            pool.map(rerank_stage_partial_func, range(self.config.ml.num_ml_workers))
            self.logger.info("ML Rerank stage complete.")

        finally:
            pool.close()
            pool.join()

    def _ml_worker(self, rank: int, stage: str, global_frequencies: Tuple):
        """ML worker process implementation."""
        dataset_name = self.config.data.dataset_name or "default_dataset"
        table_name = self.config.data.table_name or "default_table"
        mongo_uri = self.config.database.mongo_uri or "mongodb://gator-mongodb:27017/"
        db_name = self.config.database.db_name or "alligator_db"

        # Determine model path based on stage
        model_path = (
            self.config.ml.ranker_model_path
            if stage == "rank"
            else self.config.ml.reranker_model_path
        )
        max_candidates_for_stage = (
            -1 if stage == "rank" else self.config.retrieval.max_candidates_in_result
        )

        ml_worker = MLWorker(
            rank,
            table_name=table_name,
            dataset_name=dataset_name,
            stage=stage,
            model_path=model_path,
            batch_size=self.config.ml.ml_worker_batch_size,
            max_candidates_in_result=max_candidates_for_stage,
            top_n_cta_cpa_freq=self.config.feature.top_n_cta_cpa_freq,
            features=self.config.ml.selected_features,
            mongo_uri=mongo_uri,
            db_name=db_name,
            input_collection=self.config.database.input_collection,
        )

        # Run the ML worker with global frequencies
        return ml_worker.run(global_frequencies=global_frequencies)
