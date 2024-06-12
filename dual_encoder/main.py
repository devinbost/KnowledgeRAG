from dual_encoder.ContrastiveDataModule import ContrastiveDataModule
from dual_encoder.DualEncoderT5Contrastive import DualEncoderT5Contrastive
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
#from dual_encoder.FaissIndexing import FaissIndexing
from dual_encoder.AstraDBIndexing import AstraDBIndexing
import torch
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import os

def main():
    filename = "coarse_dual_T5_math_test_no_hard_neg"
    suffix = "simple"
    dataset_prefix = f"symbolic_math_example_{suffix}"
    # Used at eval
    log_name = dataset_prefix + "___" + filename + "__" + suffix
    base_path = '/teamspace/studios/this_studio/experiments/dual-encoder/'
    save_path = base_path + 'saves'
    data_path = base_path + 'data/' + dataset_prefix + '.parquet'

    torch.set_float32_matmul_precision("medium")

    vector_length = 512

    # Initialize
    #data_module = ContrastiveDataModule.,
    data_module = ContrastiveDataModule(
        data_path=data_path, 
        dataset_prefix=dataset_prefix,
        save_path=save_path,
        tokenizer_name='t5-small', 
        batch_size=2, 
        max_length=1024, 
        match_behavior = "omit",
        train_size=0.80, 
        val_size=0.10, 
        test_size=0.10)

        # train_size=0.90, 
        # val_size=0.05, 
        # test_size=0.05)
    os.environ["ASTRA_TOKEN"] = "test"
    os.environ["ASTRA_API_ENDPOINT"] = "test"
    os.environ["ASTRA_COLLECTION"] = "test"

    data_module.setup()
    #faiss_indexing_strategy = FaissIndexing(vector_length=vector_length, index_m=32, efConstruction=200)
    astradb_indexing_strategy = AstraDBIndexing(
        token=os.getenv("ASTRA_TOKEN"), 
        api_endpoint=os.getenv("ASTRA_API_ENDPOINT"), 
        collection_name=os.getenv("ASTRA_COLLECTION"), 
        vector_length=vector_length, 
        default_limit=10
    )

    #model = DualEncoderT5Contrastive.load_from_checkpoint("/teamspace/jobs/.../epoch=0-step=1473.ckpt",
    model = DualEncoderT5Contrastive(
        data_module=data_module, 
        model_name='t5-small',
        indexing_strategy=astradb_indexing_strategy, 
        learning_rate=1e-3, 
        temperature=0.2,
        margin=1.0)
    # Initialize trainer

    logger = TensorBoardLogger(filename, name=log_name)

    trainer = pl.Trainer(max_epochs=50, accelerator="gpu", devices=-1,
                        enable_progress_bar=True,
                        #limit_train_batches=3,
                        # limit_val_batches=0,
                        #enable_checkpointing=False,
                        #callbacks=[EarlyStopping(monitor="val_loss", patience=2)],
                        callbacks=[ModelCheckpoint(monitor="ndcg10_predict_all", mode="max"),
                                    EarlyStopping(monitor="ndcg10_predict_all", patience=3, mode="max")],
                        logger=logger,
                        precision="16-mixed")

    

    # Train the model
    trainer.fit(model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)



if __name__ == "__main__":
    main()