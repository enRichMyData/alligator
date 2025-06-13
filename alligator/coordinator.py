"""
Coordinator and manager classes for the Alligator entity linking system.

This module implements the coordinator pattern to orchestrate the entity linking
pipeline through specialized managers, replacing the monolithic Alligator class.
"""

from typing import Any, Dict, List

from alligator.config import AlligatorConfig
from alligator.feature import Feature
from alligator.log import get_logger
from alligator.manager import DataManager, MLManager, OutputManager, WorkerManager


class AlligatorCoordinator:
    """
    Main coordinator that orchestrates the entity linking pipeline.

    This class coordinates the different managers to execute the complete
    entity linking workflow while maintaining clean separation of concerns.
    """

    def __init__(self, config: AlligatorConfig):
        self.config = config
        self.logger = get_logger("coordinator")

        # Initialize managers
        self.data_manager = DataManager(config)
        self.worker_manager = WorkerManager(config)
        self.ml_manager = MLManager(config)
        self.output_manager = OutputManager(config)

        # Initialize feature computation
        dataset_name = config.data.dataset_name or "default_dataset"
        table_name = config.data.table_name or "default_table"
        db_name = config.database.db_name or "alligator_db"
        mongo_uri = config.database.mongo_uri or "mongodb://gator-mongodb:27017/"

        self.feature = Feature(
            dataset_name,
            table_name,
            top_n_cta_cpa_freq=config.feature.top_n_cta_cpa_freq,
            features=config.ml.selected_features,
            db_name=db_name,
            mongo_uri=mongo_uri,
            input_collection=config.database.input_collection,
        )

    def run(self) -> List[Dict[str, Any]]:
        """Execute the complete entity linking pipeline."""
        self.logger.info("Starting Alligator entity linking pipeline...")

        # Step 1: Data onboarding
        self.logger.info("Step 1: Data onboarding...")
        self.data_manager.onboard_data()

        # Step 2: Worker-based processing
        self.logger.info("Step 2: Running workers for candidate retrieval and processing...")
        self.worker_manager.run_workers(self.feature)

        # Step 3: ML pipeline
        self.logger.info("Step 3: Running ML pipeline...")
        self.ml_manager.run_ml_pipeline(self.feature)

        # Step 4: Output generation
        self.logger.info("Step 4: Generating output...")
        extracted_rows = self.output_manager.save_output()

        self.logger.info("Alligator entity linking pipeline completed successfully!")
        return extracted_rows

    def close_connections(self):
        """Cleanup resources and close connections."""
        from alligator.mongo import MongoConnectionManager

        try:
            MongoConnectionManager.close_connection()
        except Exception:
            pass
        self.logger.info("Connections closed.")
