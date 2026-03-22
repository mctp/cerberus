"""Coverage tests for cerberus.genome — untested code paths."""
import pytest

from cerberus.genome import _create_folds_chrom_partition, create_genome_folds

# ---------------------------------------------------------------------------
# create_genome_folds
# ---------------------------------------------------------------------------

class TestCreateGenomeFolds:

    def test_unknown_fold_type_raises(self):
        chrom_sizes = {"chr1": 1000, "chr2": 2000}
        with pytest.raises(ValueError, match="Unknown fold_type"):
            create_genome_folds(chrom_sizes, fold_type="unknown_strategy", fold_args={"k": 3})


# ---------------------------------------------------------------------------
# _create_folds_chrom_partition
# ---------------------------------------------------------------------------

class TestChromPartition:

    def test_distributes_chroms_into_k_folds(self):
        chrom_sizes = {
            "chr1": 100000,
            "chr2": 80000,
            "chr3": 60000,
            "chr4": 40000,
            "chr5": 20000,
        }
        folds = _create_folds_chrom_partition(chrom_sizes, k=3)
        assert len(folds) == 3

        # Each fold should be a dict of chrom -> InterLap
        all_chroms = set()
        for fold in folds:
            for chrom in fold:
                all_chroms.add(chrom)
        # All chromosomes should be assigned
        assert all_chroms == set(chrom_sizes.keys())

    def test_roughly_equal_fold_sizes(self):
        """Greedy allocation should produce roughly equal total sizes."""
        chrom_sizes = {
            "chr1": 248956422,
            "chr2": 242193529,
            "chr3": 198295559,
            "chr4": 190214555,
            "chr5": 181538259,
            "chrX": 156040895,
        }
        k = 3
        folds = _create_folds_chrom_partition(chrom_sizes, k=k)
        assert len(folds) == k

        fold_totals = []
        for fold in folds:
            total = sum(chrom_sizes[c] for c in fold)
            fold_totals.append(total)

        avg = sum(fold_totals) / k
        for total in fold_totals:
            # Each fold should be within ~35% of average for this distribution
            assert abs(total - avg) / avg < 0.35, f"Fold total {total} too far from avg {avg}"

    def test_k_equals_num_chroms(self):
        """When k equals the number of chromosomes, each fold has exactly one chrom."""
        chrom_sizes = {"chr1": 100, "chr2": 200, "chr3": 300}
        folds = _create_folds_chrom_partition(chrom_sizes, k=3)
        assert len(folds) == 3
        for fold in folds:
            assert len(fold) == 1

    def test_k_greater_than_chroms(self):
        """When k > chromosomes, some folds will be empty."""
        chrom_sizes = {"chr1": 100, "chr2": 200}
        folds = _create_folds_chrom_partition(chrom_sizes, k=5)
        assert len(folds) == 5
        non_empty = [f for f in folds if len(f) > 0]
        assert len(non_empty) == 2

    def test_single_fold(self):
        """k=1 puts all chroms in one fold."""
        chrom_sizes = {"chr1": 100, "chr2": 200}
        folds = _create_folds_chrom_partition(chrom_sizes, k=1)
        assert len(folds) == 1
        assert len(folds[0]) == 2
