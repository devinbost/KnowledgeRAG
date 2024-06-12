print("Initializing dual_encoder package")
try:
    from .t5_contrastive import DualEncoderT5Contrastive, ContrastiveSentencePairDataset, ContrastiveDataModule, InfoNCELoss
    from .indexing_strategy import IndexingStrategy
    from .astradb_indexing import AstraDBIndexing
    from .faiss_indexing import FaissIndexing
except Exception as e:
    print("Error importing modules:", e)

__all__ = [DualEncoderT5Contrastive, ContrastiveSentencePairDataset, ContrastiveDataModule, InfoNCELoss, IndexingStrategy, AstraDBIndexing, FaissIndexing]