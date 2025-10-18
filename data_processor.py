import os
import numpy as np
import pickle
from tqdm import tqdm


class EEGDataset:
    def __init__(self, data_dirs, metadata_file=None, rebuild=False):
        """
        data_dirs: list of directories, each containing spectrogram .npy files
        metadata_file: optional path to save/load precomputed index_map and file_pairs
        rebuild: if True, force rebuild metadata even if metadata_file exists
        """
        self.data_dirs = data_dirs

        if metadata_file and os.path.exists(metadata_file) and not rebuild:
            # Load precomputed metadata
            with open(metadata_file, "rb") as f:
                meta = pickle.load(f)
            self.file_pairs = meta['file_pairs']
            self.index_map = meta['index_map']
            self.length = len(self.index_map)
            print(f"Loaded metadata from {metadata_file}. Total items: {self.length}")
        else:
            # Build metadata from scratch
            all_files = []
            file_to_path = {}
            for spec_dir in data_dirs:
                fold_items = os.listdir(spec_dir)
                for fim in fold_items:
                    all_files.append(fim)
                    file_to_path[fim] = os.path.join(spec_dir, fim)

            # Separate bipolar vs referential
            bipolar_files = sorted([f for f in all_files if "bipolar" in f])
            referential_files = sorted([f for f in all_files if "referential" in f])

            # Match files by timestamp
            self.file_pairs = []
            for bf in bipolar_files:
                ts = bf.split("_")[-1].split(".")[0]
                rf = f"batch_referential_{ts}.npy"
                if rf in referential_files:
                    self.file_pairs.append((
                        file_to_path[bf],
                        file_to_path[rf]
                    ))

            # Build index map
            self.index_map = []
            for pair_idx, (bipolar_file, referential_file) in tqdm(enumerate(self.file_pairs), desc="Building index map"):
                n_items = len(np.load(bipolar_file, allow_pickle=True))
                assert n_items == len(np.load(referential_file, allow_pickle=True)), \
                    f"Mismatch in file lengths: {bipolar_file} vs {referential_file}"

                for i in range(n_items):
                    self.index_map.append((pair_idx, i))

            self.length = len(self.index_map)

            # Save metadata for future use
            if metadata_file:
                meta = {
                    "file_pairs": self.file_pairs,
                    "index_map": self.index_map
                }
                with open(metadata_file, "wb") as f:
                    pickle.dump(meta, f)
                print(f"Saved metadata to {metadata_file}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pair_idx, array_idx = self.index_map[idx]
        bipolar_file, referential_file = self.file_pairs[pair_idx]

        data_bip = np.load(bipolar_file, allow_pickle=True)[array_idx]
        data_ref = np.load(referential_file, allow_pickle=True)[array_idx]

        bipolar_specs = data_bip['spectrogram']
        referential_specs = data_ref['spectrogram']
        spectrogram = np.vstack([bipolar_specs, referential_specs])

        return spectrogram, data_ref['label'], data_ref['votes']