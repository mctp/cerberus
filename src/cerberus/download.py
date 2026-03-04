
import logging
from pathlib import Path
import urllib.request
import shutil
import gzip
import pyfaidx

logger = logging.getLogger(__name__)

def _download_file(url: str, dest: Path):
    """Downloads a file from a URL to a destination path."""
    # Use a custom user agent to avoid potential 403s from some servers
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req) as response, open(dest, "wb") as out_file:
        shutil.copyfileobj(response, out_file)

def download_dataset(output_dir: Path | str, name: str) -> dict[str, Path]:
    """
    Downloads a specified dataset.

    Args:
        output_dir: Directory where the dataset files will be saved.
        name: Name of the dataset to download. Currently supported:
              - 'mdapca2b_ar': MDA-PCA-2b AR ChIP-seq data (BigWig and narrowPeak)
              - 'kidney_scatac': Human kidney 10x scATAC-seq from CellxGene
                (fragments, tabix index, and gene activity h5ad;
                 27,034 cells, 14 cell types, 5 donors, GRCh38)

    Returns:
        Dictionary mapping file types to their local paths.
        For 'mdapca2b_ar': 'bigwig', 'narrowPeak'.
        For 'kidney_scatac': 'fragments', 'fragments_index', 'h5ad'.

    Raises:
        ValueError: If the dataset name is not supported.
    """
    out_dir = Path(output_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    if name == "mdapca2b_ar":
        files = {
            "bigwig": "mdapca2b-ar.bigwig",
            "narrowPeak": "mdapca2b-ar.narrowPeak.gz"
        }

        urls = {
            "bigwig": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM8605979&format=file&file=GSM8605979%5F05%5F0FP2%5F0255Genen%5FDMSO%2D3a%5FAR%5Fhs%5Fi37%5FR1%2Esrt%2Enodup%5Fx%5F00%5F0FPL%5F0255Genen%5FPooled%5FInput%5Fhs%5Fi67%5FR1%2Esrt%2Enodup%2Efc%2Esignal%2Ebigwig",
            "narrowPeak": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM8605979&format=file&file=GSM8605979%5F05%5F0FP2%5F0255Genen%5FDMSO%2D3a%5FAR%5Fhs%5Fi37%5FR1%2Esrt%2Enodup%5Fx%5F00%5F0FPL%5F0255Genen%5FPooled%5FInput%5Fhs%5Fi67%5FR1%2Esrt%2Enodup%2Epval0%2E01%2E500K%2Ebfilt%2EnarrowPeak%2Egz"
        }

        for key, filename in files.items():
            filepath = out_dir / filename
            if not filepath.exists():
                logger.info(f"Downloading {filename}...")
                _download_file(urls[key], filepath)
            results[key] = filepath

    elif name == "kidney_scatac":
        files = {
            "fragments": "fragments.tsv.bgz",
            "fragments_index": "fragments.tsv.bgz.tbi",
            "h5ad": "gene_activity.h5ad",
        }

        _CELLXGENE_BASE = "https://datasets.cellxgene.cziscience.com"
        urls = {
            "fragments": f"{_CELLXGENE_BASE}/214d3c72-6dfc-4806-9077-e24060eba55a-fragment.tsv.bgz",
            "fragments_index": f"{_CELLXGENE_BASE}/214d3c72-6dfc-4806-9077-e24060eba55a-fragment.tsv.bgz.tbi",
            "h5ad": f"{_CELLXGENE_BASE}/43513175-baf7-4881-9564-c4daa2416026.h5ad",
        }

        for key, filename in files.items():
            filepath = out_dir / filename
            if not filepath.exists():
                logger.info(f"Downloading {filename}...")
                _download_file(urls[key], filepath)
            results[key] = filepath

    else:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Supported: ['mdapca2b_ar', 'kidney_scatac']"
        )
        
    return results

GENOME_RESOURCES = {
    "hg38": {
        "fasta_url": "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
        "blacklist_url": "https://github.com/Boyle-Lab/Blacklist/raw/master/lists/hg38-blacklist.v2.bed.gz",
        "gaps_url": "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/gap.txt.gz",
        "mappability_url": "https://hgdownload.soe.ucsc.edu/gbdb/hg38/hoffmanMappability/k100.Umap.MultiTrackMappability.bw",
        "ccre_url": "https://hgdownload.soe.ucsc.edu/gbdb/hg38/encode3/ccre/encodeCcreCombined.bb",
    },
    "mm10": {
        "fasta_url": "http://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz",
        "blacklist_url": "https://github.com/Boyle-Lab/Blacklist/raw/master/lists/mm10-blacklist.v2.bed.gz",
        "gaps_url": "http://hgdownload.soe.ucsc.edu/goldenPath/mm10/database/gap.txt.gz",
        "mappability_url": None,
        "ccre_url": "https://hgdownload.soe.ucsc.edu/gbdb/mm10/encode3/ccre/encodeCcreCombined.bb",
    },
}


def download_reference_genome(output_dir: Path | str, genome: str = "hg38") -> dict[str, Path]:
    """
    Downloads and prepares reference genome resources.

    Downloads (depending on genome):
    1. FASTA
    2. ENCODE Blacklist v2
    3. Gap locations
    4. Mappability track
    5. ENCODE cCREs

    Args:
        output_dir: Directory to save the resources.
        genome: Name of the genome (e.g., 'hg38', 'mm10').

    Returns:
        Dictionary mapping resource names to their file paths.
    """
    if genome not in GENOME_RESOURCES:
        raise ValueError(f"Unknown genome: {genome}. Supported: {list(GENOME_RESOURCES.keys())}")

    resources = GENOME_RESOURCES[genome]
    out_dir = Path(output_dir) / genome
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. FASTA
    fasta_gz = out_dir / f"{genome}.fa.gz"
    fasta_final = out_dir / f"{genome}.fa"
    if not fasta_final.exists():
        logger.info(f"Downloading FASTA from {resources['fasta_url']}...")
        _download_file(resources["fasta_url"], fasta_gz)
        logger.info("Decompressing FASTA...")
        with gzip.open(fasta_gz, "rb") as f_in, open(fasta_final, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        fasta_gz.unlink()  # cleanup
    results["fasta"] = fasta_final

    # 1b. FASTA Index
    fai_final = out_dir / f"{genome}.fa.fai"
    if not fai_final.exists():
        logger.info("Generating FASTA Index using pyfaidx...")
        _ = pyfaidx.Fasta(str(fasta_final))
    results["fai"] = fai_final

    # 2. Blacklist
    blacklist_gz = out_dir / "blacklist.bed.gz"
    blacklist_final = out_dir / "blacklist.bed"
    if not blacklist_final.exists():
        logger.info(f"Downloading Blacklist from {resources['blacklist_url']}...")
        _download_file(resources["blacklist_url"], blacklist_gz)
        logger.info("Decompressing Blacklist...")
        with gzip.open(blacklist_gz, "rb") as f_in, open(blacklist_final, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        blacklist_gz.unlink()
    results["blacklist"] = blacklist_final

    # 3. Gaps
    gap_gz = out_dir / "gap.txt.gz"
    gaps_path = out_dir / "gaps.bed"

    if not gaps_path.exists():
        if not gap_gz.exists():
            logger.info(f"Downloading Gap tracks from {resources['gaps_url']}...")
            _download_file(resources["gaps_url"], gap_gz)

        logger.info("Processing gap tracks...")
        with gzip.open(gap_gz, "rt") as f, open(gaps_path, "w") as f_gaps:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 8:
                    continue

                chrom = parts[1]
                try:
                    start = int(parts[2])
                    end = int(parts[3])
                except ValueError:
                    continue

                # Filter length > 3
                if (end - start) > 3:
                    f_gaps.write(f"{chrom}\t{start}\t{end}\n")

        if gap_gz.exists():
            gap_gz.unlink()

    results["gaps"] = gaps_path

    # 4. Mappability (Optional)
    if resources["mappability_url"]:
        mappability_path = out_dir / "mappability.bw"
        if not mappability_path.exists():
            logger.info(f"Downloading Mappability from {resources['mappability_url']}...")
            _download_file(resources["mappability_url"], mappability_path)
        results["mappability"] = mappability_path

    # 5. ENCODE cCREs
    if resources["ccre_url"]:
        encode_cre_path = out_dir / "encode_cre.bb"
        if not encode_cre_path.exists():
            logger.info(f"Downloading ENCODE cCREs from {resources['ccre_url']}...")
            _download_file(resources["ccre_url"], encode_cre_path)
        results["encode_cre"] = encode_cre_path

    return results


def download_human_reference(output_dir: Path | str, name: str = "hg38") -> dict[str, Path]:
    """
    Wrapper for download_reference_genome to maintain backward compatibility.
    """
    return download_reference_genome(output_dir, genome=name)
