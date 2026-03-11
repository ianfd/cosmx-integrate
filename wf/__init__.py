from typing import List, Optional

from latch.resources.workflow import workflow
from latch.resources.tasks import small_task
from latch.resources.launch_plan import LaunchPlan
from latch.types.file import LatchFile
from latch.types.directory import LatchDir, LatchOutputDir
from latch.types.metadata import LatchAuthor, LatchMetadata, LatchParameter

from .integrate import IntegrationInput, scvi_integration_task


metadata = LatchMetadata(
    display_name="scVI Spatial Integration (Multi-Sample)",
    author=LatchAuthor(name="Ian"),
    parameters={
        "sample_h5ads": LatchParameter(
            display_name="Sample H5AD files",
            description=(
                "List of per-sample H5AD files (post-QC). Raw counts should be "
                "in .X or .layers['counts']. Sample name is derived from filename stem. "
                "Works with one or more samples."
            ),
            batch_table_column=True,
        ),
        "n_latent": LatchParameter(
            display_name="Latent Dimensions",
            description="Dimensionality of the scVI latent space.",
        ),
        "n_layers": LatchParameter(
            display_name="Encoder/Decoder Layers",
            description="Number of hidden layers in scVI encoder/decoder.",
        ),
        "max_epochs": LatchParameter(
            display_name="Max Training Epochs",
            description="Maximum scVI training epochs (early stopping is enabled).",
        ),
        "n_top_genes": LatchParameter(
            display_name="Number of HVGs (0 = skip)",
            description=(
                "Highly variable genes to select before training. "
                "Set to 0 to use all genes (typical for small CosMx panels)."
            ),
        ),
        "gene_likelihood": LatchParameter(
            display_name="Gene Likelihood",
            description="'nb' (negative binomial) or 'zinb' (zero-inflated NB).",
        ),
        "output_dir": LatchParameter(
            display_name="Output Directory",
            description="Latch path for integrated H5AD, scVI model, and metrics.",
            batch_table_column=True,
        ),
    },
)


@small_task
def prep_integration_args(
    sample_h5ads: List[LatchFile], #does this make a download here?
    n_latent: int,
    n_layers: int,
    max_epochs: int,
    n_top_genes: int,
    gene_likelihood: str,
    output_dir: LatchDir,
) -> IntegrationInput:
    if len(sample_h5ads) < 1:
        raise ValueError("At least one sample h5ad is required.")

    if gene_likelihood not in ("nb", "zinb"):
        raise ValueError(
            f"gene_likelihood must be 'nb' or 'zinb', got '{gene_likelihood}'."
        )

    return IntegrationInput(
        sample_h5ads=sample_h5ads,
        n_latent=n_latent,
        n_layers=n_layers,
        max_epochs=max_epochs,
        n_top_genes=n_top_genes if n_top_genes > 0 else 0,
        gene_likelihood=gene_likelihood,
        output_dir=output_dir,
    )


@workflow(metadata)
def scvi_integration(
    sample_h5ads: List[LatchFile],
    n_latent: int = 30,
    n_layers: int = 2,
    max_epochs: int = 400,
    n_top_genes: int = 0,
    gene_likelihood: str = "nb",
    output_dir: LatchDir = LatchDir("latch://40726.account/cosmx-test/out-dir/integration/"),
) -> LatchOutputDir:
    integration_input = prep_integration_args(
        sample_h5ads=sample_h5ads,
        n_latent=n_latent,
        n_layers=n_layers,
        max_epochs=max_epochs,
        n_top_genes=n_top_genes,
        gene_likelihood=gene_likelihood,
        output_dir=output_dir,
    )

    return scvi_integration_task(input=integration_input)


LaunchPlan(
    scvi_integration,
    "Test Data (2 samples)",
    {
        "sample_h5ads": [
            LatchFile("latch://40726.account/cosmx-test/out-dir/conversion_out/GSE282193_Slide1/GSE282193_Slide1.h5ad"),
            LatchFile("latch://40726.account/cosmx-test/out-dir/qc/GSE282193_Slide2/GSE282193_Slide2.h5ad"),
        ],
        "n_latent": 30,
        "n_layers": 2,
        "max_epochs": 400,
        "n_top_genes": 0,
        "gene_likelihood": "nb",
        "output_dir": LatchDir("latch://40726.account/cosmx-test/out-dir/integration/"),
    },
)