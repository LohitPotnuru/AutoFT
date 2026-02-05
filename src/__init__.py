"""AutoFT - Automated Fine-Tuning Data Generation."""

from src.data_generation.ticket_generator import create_ticket_data
from src.data_ingestion.ingest_pipeline import ingest_data

__all__ = ["create_ticket_data", "ingest_data"]
