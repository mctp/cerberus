
from pathlib import Path
import urllib.request
import shutil
import gzip
import pyfaidx
from typing import Dict

def _download_file(url: str, dest: Path):
    """Downloads a file from a URL to a destination path."""
    # Use a custom user agent to avoid potential 403s from some servers
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req) as response, open(dest, "wb") as out_file:
        shutil.copyfileobj(response, out_file)

def download_dataset(output_dir: Path | str, name: str) -> Dict[str, Path]:
    """
    Downloads a specified dataset.

    Args:
        output_dir: Directory where the dataset files will be saved.
        name: Name of the dataset to download. Currently supported:
              - 'mdapca2b_ar': MDA-PCA-2b AR ChIP-seq data (BigWig and narrowPeak)

    Returns:
        Dictionary mapping file types (e.g., 'bigwig', 'narrowPeak') to their local paths.
    
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
                print(f"Downloading {filename}...")
                _download_file(urls[key], filepath)
            results[key] = filepath
            
    else:
        raise ValueError(f"Unknown dataset: {name}. Supported: ['mdapca2b_ar']")
        
    return results

def download_human_reference(output_dir: Path | str, name: str = "hg38") -> dict[str, Path]:
    """
    Downloads and prepares human reference genome resources (hg38).

    Downloads:
    1. hg38 FASTA
    2. ENCODE Blacklist v2
    3. Gap locations (telomeres, centromeres, etc.)

    Generates:
    - hg38.fa
    - blacklist.bed
    - gaps.bed (all assembly gaps, suitable for 'unmappable' exclusion)
    - mappability.bw (mappability track)
    - encode_cre.bb (ENCODE cCREs)

    Args:
        output_dir: Directory to save the resources.
        name: Name of the genome directory (default: 'hg38').

    Returns:
        Dictionary mapping resource names to their file paths.
    """
    out_dir = Path(output_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # URLs
    HG38_FASTA_URL = (
        "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
    )
    HG38_BLACKLIST_URL = "https://github.com/Boyle-Lab/Blacklist/raw/master/lists/hg38-blacklist.v2.bed.gz"
    HG38_GAP_URL = (
        "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/gap.txt.gz"
    )

    # 1. FASTA
    fasta_gz = out_dir / "hg38.fa.gz"
    fasta_final = out_dir / "hg38.fa"
    if not fasta_final.exists():
        print(f"Downloading FASTA from {HG38_FASTA_URL}...")
        _download_file(HG38_FASTA_URL, fasta_gz)
        print("Decompressing FASTA...")
        with gzip.open(fasta_gz, "rb") as f_in, open(fasta_final, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        fasta_gz.unlink()  # cleanup
    results["fasta"] = fasta_final

    # 1b. FASTA Index
    fai_final = out_dir / "hg38.fa.fai"
    if not fai_final.exists():
        print("Generating FASTA Index using pyfaidx...")
        # Accessing the Fasta object triggers index generation
        _ = pyfaidx.Fasta(str(fasta_final))
    results["fai"] = fai_final

    # 2. Blacklist
    blacklist_gz = out_dir / "blacklist.bed.gz"
    blacklist_final = out_dir / "blacklist.bed"
    if not blacklist_final.exists():
        print(f"Downloading Blacklist from {HG38_BLACKLIST_URL}...")
        _download_file(HG38_BLACKLIST_URL, blacklist_gz)
        print("Decompressing Blacklist...")
        with gzip.open(blacklist_gz, "rb") as f_in, open(blacklist_final, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        blacklist_gz.unlink()
    results["blacklist"] = blacklist_final

    # 3. Gaps
    gap_gz = out_dir / "gap.txt.gz"
    gaps_path = out_dir / "gaps.bed"

    if not gaps_path.exists():
        if not gap_gz.exists():
            print(f"Downloading Gap tracks from {HG38_GAP_URL}...")
            _download_file(HG38_GAP_URL, gap_gz)

        print("Processing gap tracks...")
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

    # 4. Mappability
    mappability_url = "https://hgdownload.soe.ucsc.edu/gbdb/hg38/hoffmanMappability/k100.Umap.MultiTrackMappability.bw"
    mappability_path = out_dir / "mappability.bw"

    if not mappability_path.exists():
        print(f"Downloading Mappability from {mappability_url}...")
        _download_file(mappability_url, mappability_path)

    results["mappability"] = mappability_path

    # 5. ENCODE cCREs
    encode_cre_url = "https://hgdownload.soe.ucsc.edu/gbdb/hg38/encode3/ccre/encodeCcreCombined.bb"
    encode_cre_path = out_dir / "encode_cre.bb"

    if not encode_cre_path.exists():
        print(f"Downloading ENCODE cCREs from {encode_cre_url}...")
        _download_file(encode_cre_url, encode_cre_path)
    
    results["encode_cre"] = encode_cre_path

    return results
